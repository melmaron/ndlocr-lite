"""Deterministic output-path tests that do not require the OCR models.

They use tiny stub detectors/recognizers, bundled class definitions and a
synthetic PDF, so they stay fast in CI and contain no external data.
"""
from argparse import Namespace
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import ocr


DET_CLASSES = ROOT / "src" / "config" / "ndl.yaml"


def _class_names():
    with DET_CLASSES.open("r", encoding="utf-8") as stream:
        names = yaml.safe_load(stream)["names"]
    return [names[i] for i in sorted(names)]


class _StubDetector:
    def __init__(self, detections=None):
        self.classes = {i: name for i, name in enumerate(_class_names())}
        self.markdown_styles = {}
        self._detections = detections

    def detect(self, img):
        if self._detections is not None:
            return [dict(item) for item in self._detections]
        return [
            {
                "class_index": 2,
                "confidence": 0.9,
                "box": np.array([10, 10, 80, 30], dtype=np.int32),
                "pred_char_count": 2.0,
                "pred_markdown_style": 0.0,
                "class_name": "line_caption",
            },
            {
                "class_index": 1,
                "confidence": 0.8,
                "box": np.array([10, 40, 80, 60], dtype=np.int32),
                "pred_char_count": 100.0,
                "pred_markdown_style": 0.0,
                "class_name": "line_main",
            },
        ]


class _StubRecognizer:
    def __init__(self, text="TESTLINE"):
        self.text = text
        self.batch_size = 1
        self.backend_profile = None

    def read(self, img):
        return self.text

    def read_batch(self, imgs, batch_size=None):
        return [self.text for _ in imgs]


def _make_args(output_dir, **overrides):
    args = Namespace(
        sourcedir=None,
        sourceimg=None,
        sourcepdf=None,
        output=str(output_dir),
        viz=False,
        det_weights="det.onnx",
        det_classes=str(DET_CLASSES),
        det_score_threshold=0.2,
        det_conf_threshold=0.25,
        det_iou_threshold=0.2,
        det_overlap_threshold=0.9,
        rec_weights30="rec30.onnx",
        rec_weights50="rec50.onnx",
        rec_weights="rec100.onnx",
        rec_classes="classes.yaml",
        device="cpu",
        device_id=0,
        batch_size=1,
        strict_device=False,
        check_device=False,
        json_only=False,
        enable_tcy=False,
        pdf_output=None,
        pdf_render_dpi=100.0,
        pdf_page_output=True,
        pdf_visible_text=False,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def _write_png(path: Path):
    Image.fromarray(np.full((120, 160, 3), 255, dtype=np.uint8)).save(path)


def _write_pdf(path: Path, pages=2):
    from reportlab.pdfgen import canvas

    pdf = canvas.Canvas(str(path), pagesize=(200, 150))
    for _ in range(pages):
        pdf.setFillColorRGB(0, 0, 0)
        pdf.rect(20, 20, 120, 40, stroke=1, fill=0)
        pdf.showPage()
    pdf.save()


class OcrOutputTest(unittest.TestCase):
    def test_image_json_class_index_uses_localized_type(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            src = tmp / "input"
            src.mkdir()
            _write_png(src / "page.png")
            out = tmp / "out"
            args = _make_args(out, sourcedir=str(src))
            detector = _StubDetector()
            recognizer = _StubRecognizer()
            with patch.object(ocr, "get_detector", return_value=detector), patch.object(
                ocr, "get_recognizer", return_value=recognizer
            ):
                ocr.process(args)
            payload = json.loads((out / "page.json").read_text(encoding="utf-8"))
            indexes = sorted(line["class_index"] for line in payload["contents"][0])
            self.assertEqual(indexes, [1, 2])
            xml = (out / "page.xml").read_text(encoding="utf-8")
            self.assertIn('TYPE="キャプション"', xml)
            self.assertIn('TYPE="本文"', xml)

    def test_pdf_outputs_and_text_layer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            pdf_path = tmp / "doc.pdf"
            _write_pdf(pdf_path, pages=2)
            out = tmp / "out"
            text_pdf = out / "text.pdf"
            args = _make_args(
                out,
                sourcepdf=str(pdf_path),
                pdf_output=str(text_pdf),
            )
            with patch.object(ocr, "get_detector", return_value=_StubDetector()), patch.object(
                ocr, "get_recognizer", return_value=_StubRecognizer("TESTLINE")
            ):
                ocr.process_pdf_documents(args, [str(pdf_path)])

            for index in (1, 2):
                stem = f"doc_{index:05}"
                self.assertTrue((out / f"{stem}.xml").is_file())
                self.assertTrue((out / f"{stem}.txt").is_file())
                payload = json.loads((out / f"{stem}.json").read_text(encoding="utf-8"))
                self.assertEqual(
                    sorted(line["class_index"] for line in payload["contents"][0]),
                    [1, 2],
                )
                self.assertEqual(payload["imginfo"]["source_page_index"], index - 1)
            self.assertTrue(text_pdf.is_file())

            import pypdfium2

            generated = pypdfium2.PdfDocument(str(text_pdf))
            try:
                self.assertEqual(len(generated), 2)
                text = "".join(
                    generated[i].get_textpage().get_text_bounded() for i in range(len(generated))
                )
                self.assertIn("TESTLINE", text)
            finally:
                generated.close()

            # Re-embedding into an already text-layered PDF must not raise.
            again = tmp / "again.pdf"
            page_results = [
                {
                    "text_layer_lines": [
                        {
                            "x": 10,
                            "y": 20,
                            "width": 60,
                            "height": 12,
                            "text": "TESTLINE",
                            "is_vertical": False,
                        }
                    ],
                    "img_width": 200,
                    "img_height": 150,
                }
                for _ in range(2)
            ]
            ocr.embed_text_layer_pdf(
                str(text_pdf), str(again), page_results, visible_text=True
            )
            self.assertTrue(again.is_file())

    def test_track_write_future_bounds_and_propagates(self):
        calls = []

        class FakeFuture:
            def __init__(self, name, fail=False):
                self.name = name
                self.fail = fail

            def result(self):
                calls.append(self.name)
                if self.fail:
                    raise RuntimeError(f"write failed: {self.name}")

        futures = []
        ocr._track_write_future(futures, FakeFuture("a"), limit=3)
        ocr._track_write_future(futures, FakeFuture("b"), limit=3)
        self.assertEqual(calls, [])
        ocr._track_write_future(futures, FakeFuture("c"), limit=3)
        self.assertEqual(calls, ["a"])  # bounded backlog drains oldest
        self.assertEqual(len(futures), 2)

        failing = []
        with self.assertRaisesRegex(RuntimeError, "write failed: x"):
            ocr._track_write_future(failing, FakeFuture("x", fail=True), limit=2)
            ocr._track_write_future(failing, FakeFuture("y"), limit=2)



class XmlFilenameEscapingTest(unittest.TestCase):
    """PAGE@IMAGENAME must stay well-formed for POSIX filenames with <, >, &, "."""

    def _empty_result(self, classes):
        result = [dict(), dict()]
        result[0][0] = []
        for index in range(len(classes)):
            result[1][index] = []
        return result

    def test_convert_to_xml_string3_returns_well_formed_xml_for_special_filenames(self):
        import xml.etree.ElementTree as ET

        from ndl_parser import convert_to_xml_string3

        classes = _class_names()
        result = self._empty_result(classes)
        names = [
            "page.png",
            "a&b.png",
            "a&amp;b.png",
            "a<b.png",
            "a>b.png",
            'a"b.png',
            "日本語 ページ.png",
        ]
        for name in names:
            with self.subTest(name=name):
                xml = convert_to_xml_string3(120, 160, name, classes, result)
                root = ET.fromstring("<OCRDATASET>" + xml + "</OCRDATASET>")
                self.assertEqual(root.find("PAGE").get("IMAGENAME"), name)

    def test_process_writes_outputs_for_xml_special_filenames(self):
        import xml.etree.ElementTree as ET

        names = ["a<b.png", 'a"b.png', "a&b.png", "a&amp;b.png"]
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            src = tmp / "input"
            src.mkdir()
            try:
                for name in names:
                    _write_png(src / name)
            except OSError:
                self.skipTest("Filesystem does not support special filenames")
            out = tmp / "out"
            args = _make_args(out, sourcedir=str(src))
            with patch.object(ocr, "get_detector", return_value=_StubDetector()), patch.object(
                ocr, "get_recognizer", return_value=_StubRecognizer("TESTLINE")
            ):
                ocr.process(args)
            for name in names:
                stem = Path(name).stem
                with self.subTest(name=name):
                    xml_root = ET.parse(out / f"{stem}.xml").getroot()
                    self.assertEqual(xml_root.find("PAGE").get("IMAGENAME"), name)
                    payload = json.loads((out / f"{stem}.json").read_text(encoding="utf-8"))
                    self.assertEqual(payload["imginfo"]["img_name"], name)
                    lines = [
                        line
                        for line in (out / f"{stem}.txt").read_text(encoding="utf-8").splitlines()
                        if line
                    ]
                    self.assertTrue(lines)
                    self.assertEqual(set(lines), {"TESTLINE"})

    def test_run_ocr_on_image_array_accepts_special_inputname(self):
        import xml.etree.ElementTree as ET

        with tempfile.TemporaryDirectory() as tmpdir:
            img = np.full((120, 160, 3), 255, dtype=np.uint8)
            for inputname in ("page.png", "a<b.png", 'a"b.png', "a&b.png", "a&amp;b.png"):
                with self.subTest(inputname=inputname):
                    page_result = ocr._run_ocr_on_image_array(
                        detector=_StubDetector(),
                        recognizer30=_StubRecognizer("TESTLINE"),
                        recognizer50=_StubRecognizer("TESTLINE"),
                        recognizer100=_StubRecognizer("TESTLINE"),
                        inputname=inputname,
                        img=img,
                        outputpath=tmpdir,
                        save_viz=False,
                    )
                    root = ET.fromstring("<OCRDATASET>" + page_result["page_xml"] + "</OCRDATASET>")
                    self.assertEqual(root.find("PAGE").get("IMAGENAME"), inputname)


if __name__ == "__main__":
    unittest.main()
