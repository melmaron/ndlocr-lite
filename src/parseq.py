from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from onnx_batch_patcher import is_dynamic_batch_model, prepare_parseq_model
from ort_provider import (
    create_inference_session,
    normalize_device_name,
    resolve_auto_backend_name,
)


def _remove_ascii_spaces_around_japanese_punctuation(text: str) -> str:
    for punctuation in "、。":
        text = text.replace(f" {punctuation}", punctuation)
        text = text.replace(f"{punctuation} ", punctuation)
    return text


class PARSEQ:
    def __init__(
        self,
        model_path: str,
        charlist: List[str],
        original_size: Tuple[int, int] = (384, 32),
        device: str = "auto",
        device_id: int = 0,
        batch_size: int = 0,
        strict_device: bool = False,
        precision: str = "fp32",
    ) -> None:
        self.model_path = model_path
        self.charlist = charlist
        self.device = device or "auto"
        self.device_id = int(device_id or 0)
        self.requested_batch_size = int(batch_size or 0)
        self.strict_device = bool(strict_device)
        self.precision = precision
        self.image_width, self.image_height = original_size
        self.backend_profile = None
        self.dynamic_batch = False
        self.batch_size = 1
        self.effective_model_path = str(model_path)
        self.create_session()

    def create_session(self) -> None:
        requested = normalize_device_name(self.device)
        resolved_target = resolve_auto_backend_name() if requested == "auto" else requested
        want_dynamic = resolved_target != "cpu" and self.requested_batch_size != 1

        effective_path = Path(self.model_path)
        if want_dynamic:
            patched = prepare_parseq_model(
                effective_path,
                enable_dynamic_batch=True,
                precision=self.precision,
            )
            effective_path = patched.model_path
            self.dynamic_batch = patched.dynamic_batch
        else:
            self.dynamic_batch = is_dynamic_batch_model(effective_path)

        self.effective_model_path = str(effective_path)
        session, profile = create_inference_session(
            self.effective_model_path,
            device=self.device,
            device_id=self.device_id,
            strict_device=self.strict_device,
            cpu_single_thread=True,
            model_label=f"PARSEQ({Path(self.model_path).name})",
        )
        self.session = session
        self.backend_profile = profile
        self.model_inputs = self.session.get_inputs()
        self.input_names = [self.model_inputs[i].name for i in range(len(self.model_inputs))]
        self.input_shape = self.model_inputs[0].shape
        self.input_dtype = (
            np.float16 if "float16" in str(self.model_inputs[0].type) else np.float32
        )
        self.model_output = self.session.get_outputs()
        self.output_names = [self.model_output[i].name for i in range(len(self.model_output))]
        self.input_height, self.input_width = self.input_shape[2:]

        dim0 = self.input_shape[0]
        runtime_dynamic = isinstance(dim0, str) or dim0 is None or dim0 == 0
        self.dynamic_batch = bool(self.dynamic_batch or runtime_dynamic)

        if not self.dynamic_batch:
            self.batch_size = 1
        elif self.requested_batch_size > 0:
            self.batch_size = self.requested_batch_size
        else:
            self.batch_size = max(1, int(profile.default_batch_size))

    def postprocess(self, outputs):
        predictions = np.squeeze(outputs).T
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > self.conf_thresold, :]
        scores = scores[scores > self.conf_thresold]
        class_ids = np.argmax(predictions[:, 4:], axis=1)

    def _preprocess_single(self, img: np.ndarray, *, auto_rotate: bool = True) -> np.ndarray:
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        h, w = img.shape[:2]
        if auto_rotate and h > w * 0.8:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        resized = cv2.resize(
            img,
            (self.input_width, self.input_height),
            interpolation=cv2.INTER_LINEAR,
        )
        input_image = np.ascontiguousarray(resized[:, :, ::-1]).astype(np.float32)
        input_image /= 127.5
        input_image -= 1.0
        tensor = input_image.transpose(2, 0, 1)
        if self.input_dtype != np.float32:
            tensor = tensor.astype(self.input_dtype)
        return tensor

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        return self._preprocess_single(img, auto_rotate=True)[np.newaxis, :, :, :]

    def preprocess_batch(
        self,
        imgs: Sequence[np.ndarray],
        *,
        auto_rotate: bool = True,
    ) -> np.ndarray:
        if not imgs:
            return np.empty(
                (0, 3, self.input_height, self.input_width),
                dtype=self.input_dtype,
            )
        if len(imgs) >= 8:
            max_workers = min(4, len(imgs))
            with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="parseq-prep") as pool:
                tensors = list(
                    pool.map(lambda im: self._preprocess_single(im, auto_rotate=auto_rotate), imgs)
                )
        else:
            tensors = [self._preprocess_single(im, auto_rotate=auto_rotate) for im in imgs]
        return np.ascontiguousarray(np.stack(tensors, axis=0))

    def _decode_outputs(self, outputs: np.ndarray, *, clean_punctuation: bool = True) -> List[str]:
        all_indices = np.argmax(outputs, axis=2)
        texts: List[str] = []
        for row_idx in range(all_indices.shape[0]):
            indices = all_indices[row_idx]
            stop_idx = np.where(indices == 0)[0]
            end_pos = int(stop_idx[0]) if stop_idx.size > 0 else len(indices)
            resval = indices[:end_pos].tolist()
            resstr = "".join(self.charlist[i - 1] for i in resval)
            if clean_punctuation:
                resstr = _remove_ascii_spaces_around_japanese_punctuation(resstr)
            texts.append(resstr)
        return texts

    def read_batch_raw(
        self,
        imgs: Sequence[np.ndarray],
        batch_size: Optional[int] = None,
        *,
        auto_rotate: bool = True,
        clean_punctuation: bool = False,
    ) -> List[str]:
        """Run batched PARSeq recognition and return decoded strings."""
        if not imgs:
            return []

        results = [""] * len(imgs)
        valid_positions: List[int] = []
        valid_imgs: List[np.ndarray] = []
        for idx, img in enumerate(imgs):
            if img is not None and img.size > 0:
                valid_positions.append(idx)
                valid_imgs.append(img)

        if not valid_imgs:
            return results

        if not self.dynamic_batch:
            effective_bs = 1
        elif batch_size is not None and int(batch_size) > 0:
            effective_bs = int(batch_size)
        else:
            effective_bs = max(1, int(self.batch_size))

        for start in range(0, len(valid_imgs), effective_bs):
            chunk = valid_imgs[start:start + effective_bs]
            chunk_positions = valid_positions[start:start + effective_bs]
            batch_tensor = self.preprocess_batch(chunk, auto_rotate=auto_rotate)
            outputs = self.session.run(
                self.output_names,
                {self.input_names[0]: batch_tensor},
            )[0]
            decoded = self._decode_outputs(outputs, clean_punctuation=clean_punctuation)
            for pos, text in zip(chunk_positions, decoded):
                results[pos] = text

        return results

    def read_batch(
        self,
        imgs: Sequence[np.ndarray],
        batch_size: Optional[int] = None,
    ) -> List[str]:
        return self.read_batch_raw(
            imgs,
            batch_size=batch_size,
            auto_rotate=True,
            clean_punctuation=True,
        )

    def read(self, img: np.ndarray) -> str:
        if img is None or img.size == 0:
            return ""
        input_tensor = self.preprocess(img)
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})[0]
        return self._decode_outputs(outputs, clean_punctuation=True)[0]
