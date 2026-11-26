from argparse import Namespace
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import ocr
import onnx_batch_patcher as patcher
from parseq import PARSEQ
from tcy_wrapper import TateChuYokoWrapper


PARSEQ30 = ROOT / "src" / "model" / "parseq-ndl-24x256-30-tiny-499epoch-261124.onnx"
PARSEQ50 = ROOT / "src" / "model" / "parseq-ndl-24x384-50-tiny-278epoch-261124.onnx"
PARSEQ100 = ROOT / "src" / "model" / "parseq-ndl-24x768-100-tiny-153epoch-261124.onnx"
DEIM_MODEL = ROOT / "src" / "model" / "deim-s-1024x1024-261124.onnx"
DET_CLASSES = ROOT / "src" / "config" / "ndl.yaml"
REC_CLASSES = ROOT / "src" / "config" / "NDLmoji.yaml"

SAMPLE_IMAGES = [
    ROOT / "resource" / "digidepo_2531162_0024.jpg",
    ROOT / "resource" / "tategaki2026-04-24-094138.png",
    ROOT / "resource" / "yokogaki2026-04-24-103617.png",
]


class _DummyRec:
    def __init__(self, outputs, batch_size=1):
        self.outputs = outputs
        self.batch_size = batch_size
        self.calls = []

    def read(self, img):
        key = int(img[0, 0, 0])
        self.calls.append(key)
        return self.outputs[key]

    def read_batch(self, imgs):
        return [self.read(img) for img in imgs]


def _make_img(tag: int, h: int = 30, w: int = 120) -> np.ndarray:
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[0, 0, 0] = tag
    return arr


class ParallelOcrTest(unittest.TestCase):
    def test_iter_prefetched_preserves_order_and_propagates_errors(self):
        items = [1, 2, 3, 4, 5]
        results = list(ocr._iter_prefetched(items, lambda x: x * 10, maxsize=2))
        self.assertEqual(results, [(1, 10), (2, 20), (3, 30), (4, 40), (5, 50)])
        self.assertEqual(list(ocr._iter_prefetched([7], lambda x: x + 1)), [(7, 8)])
        self.assertEqual(list(ocr._iter_prefetched([], lambda x: x)), [])

        def failing(x):
            if x == 3:
                raise ValueError("boom at 3")
            return x

        with self.assertRaisesRegex(ValueError, "boom at 3"):
            list(ocr._iter_prefetched(items, failing, maxsize=2))

    def test_batched_cascade_matches_sequential_with_overflow_and_split(self):
        lines = [
            ocr.RecogLine(_make_img(10), idx=0, pred_char_cnt=3),
            ocr.RecogLine(_make_img(11), idx=1, pred_char_cnt=3),
            ocr.RecogLine(_make_img(12), idx=2, pred_char_cnt=2),
            ocr.RecogLine(_make_img(13, h=30, w=200), idx=3, pred_char_cnt=1),
        ]
        r30_map = {10: "short", 11: "A" * 25}
        r50_map = {11: "B" * 45, 12: "mid"}
        r100_map = {11: "final50_to_100", 13: "X" * 98}

        # Note: targetdflist200 splits arr[:, :100] (tag 13 still at [0,0,0])
        # and arr[:, 100:] (tag 0 at [0,0,0])
        r100_map[0] = "_TAIL"

        seq = ocr.process_cascade(
            [ocr.RecogLine(l.npimg.copy(), l.idx, l.pred_char_cnt) for l in lines],
            _DummyRec(r30_map, batch_size=1),
            _DummyRec(r50_map, batch_size=1),
            _DummyRec(r100_map, batch_size=1),
        )
        batched = ocr.process_cascade(
            [ocr.RecogLine(l.npimg.copy(), l.idx, l.pred_char_cnt) for l in lines],
            _DummyRec(r30_map, batch_size=8),
            _DummyRec(r50_map, batch_size=8),
            _DummyRec(r100_map, batch_size=8),
        )
        self.assertEqual(batched, seq)
        self.assertEqual(batched, ["short", "final50_to_100", "mid", ("X" * 98) + "_TAIL"])

    def test_deim_split_preprocess_matches_detect(self):
        if not DEIM_MODEL.is_file() or not SAMPLE_IMAGES[0].is_file():
            self.skipTest("Bundled DEIM model or sample image missing")
        from deim import DEIM

        det = DEIM(
            model_path=str(DEIM_MODEL),
            class_mapping_path=str(DET_CLASSES),
            device="cpu",
        )
        img_bgr = cv2.imread(str(SAMPLE_IMAGES[0]))
        direct = det.detect(img_bgr)
        tensor, max_wh = det.preprocess_with_meta(img_bgr)
        split = det.detect_from_preprocessed(tensor, max_wh)
        self.assertEqual(len(direct), len(split))
        for d1, d2 in zip(direct, split):
            self.assertEqual(d1["class_index"], d2["class_index"])
            self.assertEqual(d1["class_name"], d2["class_name"])
            self.assertAlmostEqual(float(d1["confidence"]), float(d2["confidence"]), places=6)
            self.assertAlmostEqual(float(d1["pred_char_count"]), float(d2["pred_char_count"]), places=6)
            np.testing.assert_array_equal(d1["box"], d2["box"])

    def test_tcy_wrapper_read_batch_matches_sequential_read(self):
        if not PARSEQ50.is_file():
            self.skipTest("Bundled PARSeq-50 model missing")
        with REC_CLASSES.open("r", encoding="utf-8") as stream:
            charlist = list(yaml.safe_load(stream)["model"]["charset_train"])

        with tempfile.TemporaryDirectory() as tmpdir:
            patched = patcher.prepare_parseq_model(
                PARSEQ50, enable_dynamic_batch=True, cache_dir=Path(tmpdir), verify=False
            )
            rec_seq = PARSEQ(str(PARSEQ50), charlist, device="cpu", batch_size=1)
            rec_batched = PARSEQ(str(patched.model_path), charlist, device="cpu", batch_size=4)
            tcy_seq = TateChuYokoWrapper(rec_seq)
            tcy_batched = TateChuYokoWrapper(rec_batched)

            # Vertical block strips + horizontal strip
            img = cv2.imread(str(SAMPLE_IMAGES[1]))
            h, w = img.shape[:2]
            strips = [
                img[int(h * 0.1):int(h * 0.6), int(w * 0.7):int(w * 0.8)],
                img[int(h * 0.1):int(h * 0.6), int(w * 0.5):int(w * 0.6)],
                np.full((40, 120, 3), 255, dtype=np.uint8),
            ]
            seq_out = [tcy_seq.read(s) for s in strips]
            batched_out = tcy_batched.read_batch(strips)
            self.assertEqual(batched_out, seq_out)

    def test_end_to_end_pipeline_matches_sequential_cpu(self):
        if not (DEIM_MODEL.is_file() and PARSEQ30.is_file() and PARSEQ50.is_file() and PARSEQ100.is_file()):
            self.skipTest("Bundled models missing")

        with tempfile.TemporaryDirectory() as tmpdir:
            src_dir = Path(tmpdir) / "inputs"
            src_dir.mkdir()
            for img_path in SAMPLE_IMAGES:
                shutil.copy2(img_path, src_dir / img_path.name)

            cache_dir = Path(tmpdir) / "cache"
            out_seq = Path(tmpdir) / "out_seq"
            out_batch = Path(tmpdir) / "out_batch"

            args_seq = Namespace(
                sourcedir=str(src_dir),
                sourceimg=None,
                sourcepdf=None,
                output=str(out_seq),
                viz=False,
                det_weights=str(DEIM_MODEL),
                det_classes=str(DET_CLASSES),
                det_score_threshold=0.2,
                det_conf_threshold=0.2,
                det_iou_threshold=0.2,
                det_overlap_threshold=0.7,
                rec_weights30=str(PARSEQ30),
                rec_weights50=str(PARSEQ50),
                rec_weights=str(PARSEQ100),
                rec_classes=str(REC_CLASSES),
                device="cpu",
                device_id=0,
                batch_size=1,
                strict_device=False,
                check_device=False,
                json_only=False,
                enable_tcy=False,
                tcy_digits_only=False,
                tcy_vertical_threshold=1.5,
                tcy_max_chars=2,
                tcy_min_conf=0.35,
            )

            args_batch = Namespace(**vars(args_seq))
            args_batch.output = str(out_batch)
            args_batch.batch_size = 8

            ocr.process(args_seq)

            # Force dynamic-batch PARSeq even on CPU to test full batched + 3-stage pipeline end-to-end
            orig_get_rec = ocr.get_recognizer

            def patched_get_rec(args, weights_path=None):
                target = weights_path if weights_path is not None else args.rec_weights
                patched = patcher.prepare_parseq_model(
                    target, enable_dynamic_batch=True, cache_dir=cache_dir, verify=False
                )
                return orig_get_rec(args, weights_path=str(patched.model_path))

            with patch.object(ocr, "get_recognizer", side_effect=patched_get_rec):
                ocr.process(args_batch)

            for img_path in SAMPLE_IMAGES:
                stem = img_path.stem
                self.assertEqual(
                    (out_batch / f"{stem}.txt").read_text(encoding="utf-8"),
                    (out_seq / f"{stem}.txt").read_text(encoding="utf-8"),
                )
                self.assertEqual(
                    (out_batch / f"{stem}.xml").read_text(encoding="utf-8"),
                    (out_seq / f"{stem}.xml").read_text(encoding="utf-8"),
                )
                self.assertEqual(
                    json.loads((out_batch / f"{stem}.json").read_text(encoding="utf-8")),
                    json.loads((out_seq / f"{stem}.json").read_text(encoding="utf-8")),
                )


if __name__ == "__main__":
    unittest.main()
