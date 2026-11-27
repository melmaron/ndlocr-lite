from PIL import Image, ImageDraw
import yaml
import numpy as np
from typing import Tuple, List
import xml.etree.ElementTree as ET

from ort_provider import create_inference_session


class DEIM:
    def __init__(self,
                 model_path: str,
                 class_mapping_path: str,
                 original_size: Tuple[int, int] = (1280,1280),
                 score_threshold: float = 0.1,
                 conf_threshold: float = 0.1,
                 iou_threshold: float = 0.4,
                 device: str = "auto",
                 overlap_threshold: float = 0.9,
                 device_id: int = 0,
                 strict_device: bool = False) -> None:
        if not 0.0 <= overlap_threshold <= 1.0:
            raise ValueError("overlap_threshold must be between 0 and 1")
        self.model_path = model_path
        self.class_mapping_path = class_mapping_path
        self.image_width, self.image_height = original_size
        self.device = device or "auto"
        self.device_id = int(device_id or 0)
        self.strict_device = bool(strict_device)
        self.score_threshold = score_threshold
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.overlap_threshold = overlap_threshold
        self.colorlist=[(0, 0, 0), (255, 0, 0), (0, 0, 142), (0, 0, 230), (106, 0, 228),
                        (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
                        (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30),(165, 42, 42), (255, 77, 255),(255,0,0)]
        self._mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self._inv_std = np.reciprocal(
            np.array([0.229, 0.224, 0.225], dtype=np.float32)
        )
        self.backend_profile = None
        self.create_session()

    def create_session(self) -> None:
        session, profile = create_inference_session(
            self.model_path,
            device=self.device,
            device_id=self.device_id,
            strict_device=self.strict_device,
            cpu_single_thread=False,
            model_label="DEIM",
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

        if self.class_mapping_path is not None:
            with open(self.class_mapping_path, 'r', encoding='utf-8') as file:
                yaml_file = yaml.safe_load(file)
                self.classes = yaml_file['names']
                self.markdown_styles = yaml_file.get('markdown', {})
                self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))
        else:
            self.markdown_styles = {}

    def warmup(self) -> None:
        """Run a dummy inference to trigger first-run kernel selection/setup."""
        dummy = np.zeros(
            (1, 3, int(self.input_height), int(self.input_width)),
            dtype=self.input_dtype,
        )
        self.session.run(
            self.output_names,
            {
                self.input_names[0]: dummy,
                self.input_names[1]: np.array(
                    [[int(self.input_height), int(self.input_width)]],
                    dtype=np.int64,
                ),
            },
        )

    def preprocess_with_meta(self, img: np.ndarray) -> Tuple[np.ndarray, int]:
        """Preprocess image without mutating instance dimensions (safe for prefetch)."""
        max_wh = max(img.shape[0], img.shape[1])
        paddedimg = np.zeros((max_wh, max_wh, 3), dtype=np.uint8)
        paddedimg[:img.shape[0], :img.shape[1], :] = img
        pil_image = Image.fromarray(paddedimg)
        pil_resized = pil_image.resize((self.input_width, self.input_height))
        input_image = np.asarray(pil_resized, dtype=np.float32)
        input_image *= np.float32(1.0 / 255.0)
        input_image -= self._mean
        input_image *= self._inv_std
        tensor = np.ascontiguousarray(input_image.transpose(2, 0, 1)[np.newaxis, :, :, :])
        if self.input_dtype != np.float32:
            tensor = tensor.astype(self.input_dtype)
        return tensor, max_wh

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        tensor, max_wh = self.preprocess_with_meta(img)
        self.image_width = max_wh
        self.image_height = max_wh
        return tensor

    def xywh2xyxy(self, x):
        # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
        y = np.copy(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2
        y[..., 1] = x[..., 1] - x[..., 3] / 2
        y[..., 2] = x[..., 0] + x[..., 2] / 2
        y[..., 3] = x[..., 1] + x[..., 3] / 2
        return y

    def postprocess(self, outputs):

        if len(outputs)==6:
            class_ids,bboxes,scores,char_counts,markdown_style,_=outputs
            class_ids=np.asarray(class_ids).reshape(-1)
            predictions = np.asarray(bboxes).reshape(-1, 4)
            scores=np.asarray(scores).reshape(-1)
            char_counts=np.asarray(char_counts).reshape(-1)
            markdown_style=np.asarray(markdown_style).reshape(-1)
        if len(outputs)==4:
            class_ids,bboxes,scores,char_counts=outputs
            class_ids=np.asarray(class_ids).reshape(-1)
            predictions = np.asarray(bboxes).reshape(-1, 4)
            scores=np.asarray(scores).reshape(-1)
            char_counts=np.asarray(char_counts).reshape(-1)
            markdown_style=np.array([0]*scores.shape[0])
        elif len(outputs)==3:
            class_ids,bboxes,scores=outputs
            class_ids=np.asarray(class_ids).reshape(-1)
            predictions = np.asarray(bboxes).reshape(-1, 4)
            scores=np.asarray(scores).reshape(-1)
            char_counts=np.array([100.0]*scores.shape[0])
            markdown_style=np.array([0]*scores.shape[0])
        keep = scores > self.conf_threshold
        predictions = predictions[keep, :]
        scores = scores[keep]
        class_ids = class_ids[keep]
        char_counts = char_counts[keep]
        markdown_style = markdown_style[keep]
        scales = np.array([
            self.image_width / self.input_width,
            self.image_height / self.input_width,
            self.image_width / self.input_width,
            self.image_height / self.input_width
        ], dtype=np.float32)
        boxes = (predictions[:, :4] * scales).astype(np.int32)
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, self.image_width)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, self.image_height)
        detections = []
        for bbox, score, label,char_count,m_style in zip(boxes, scores, class_ids,char_counts,markdown_style):
            class_index=int(label)-1
            detections.append({
                "class_index": class_index,
                "confidence": score,
                "box": bbox,
                "pred_char_count":char_count,
                "pred_markdown_style":m_style,
                "class_name": self.classes[class_index]#"line_main"
            })
        return self._suppress_overlapping_regions(detections)

    def _suppress_overlapping_regions(self, detections):
        """Prefer larger same-label boxes using intersection / smaller area."""
        if self.overlap_threshold == 0:
            return detections

        areas = [
            max(0, int(d["box"][2]) - int(d["box"][0]))
            * max(0, int(d["box"][3]) - int(d["box"][1]))
            for d in detections
        ]
        kept_by_class = {}
        suppressed = set()
        for index in sorted(range(len(detections)), key=lambda i: areas[i], reverse=True):
            detection = detections[index]
            area = areas[index]
            if area <= 0:
                continue
            x1, y1, x2, y2 = detection["box"]
            kept = kept_by_class.setdefault(detection["class_index"], [])
            for other_index in kept:
                if areas[other_index] <= area:
                    continue
                ox1, oy1, ox2, oy2 = detections[other_index]["box"]
                intersection = (
                    max(0, int(min(x2, ox2)) - int(max(x1, ox1)))
                    * max(0, int(min(y2, oy2)) - int(max(y1, oy1)))
                )
                if intersection / area >= self.overlap_threshold:
                    suppressed.add(index)
                    break
            else:
                kept.append(index)
        return [d for i, d in enumerate(detections) if i not in suppressed]

    def get_label_name(self, class_id: int) -> str:
        return self.classes[class_id]

    def detect_from_preprocessed(self, input_tensor: np.ndarray, max_wh: int) -> List:
        """Run inference on a pre-normalized tensor produced by :meth:`preprocess_with_meta`."""
        self.image_width = max_wh
        self.image_height = max_wh
        outputs = self.session.run(
            self.output_names,
            {
                self.input_names[0]: input_tensor,
                self.input_names[1]: np.array([[self.input_height, self.input_width]], np.int64),
            },
        )
        return self.postprocess(outputs)

    def detect(self, img: np.ndarray) -> List:
        input_tensor, max_wh = self.preprocess_with_meta(img)
        return self.detect_from_preprocessed(input_tensor, max_wh)

    def draw_detections(self, npimg: np.ndarray, detections: List):
        pil_image = Image.fromarray(npimg)
        draw = ImageDraw.Draw(pil_image)
        for detection in detections:
            # バウンディングボックスの座標を抽出
            x1, y1, x2, y2 = detection['box']
            class_id = detection['class_index']
            confidence = detection['confidence']
            # クラスIDに対する色を取得
            color = self.colorlist[class_id]  # RGB形式で青色
            # 画像にバウンディングボックスを描画
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        return pil_image

    def drawxml_detections(self, npimg: np.ndarray, xmlstr: str,categories:dict,outputimgpath:str):
        root = ET.fromstring(xmlstr)
        pil_image = Image.fromarray(npimg)
        draw = ImageDraw.Draw(pil_image)
        for child in root.iter():
            if child.tag=="LINE":
                class_id=categories[child.get("TYPE")]["id"]
                x1=int(child.get("X"))
                y1=int(child.get("Y"))
                x2=x1+int(child.get("WIDTH"))
                y2=y1+int(child.get("HEIGHT"))
            elif child.tag=="POLYGON":
                class_id=0
                xlist=[int(t) for t in child.get("POINTS").split(",") ][0::2]
                ylist=[int(t) for t in child.get("POINTS").split(",") ][1::2]
                x1,y1,x2,y2=min(xlist),min(ylist),max(xlist),max(ylist)
            else:
                continue
            color = self.colorlist[class_id]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
        save_kwargs = {"compress_level": 1} if str(outputimgpath).lower().endswith(".png") else {}
        pil_image.save(outputimgpath, **save_kwargs)
