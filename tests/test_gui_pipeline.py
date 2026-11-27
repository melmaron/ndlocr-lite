"""Unit and headless integration tests for GUI 3-stage pipeline & throughput fixes."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import logging
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import flet as ft
import numpy as np
from PIL import Image
import pypdfium2
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "ndlocr-lite-gui") not in sys.path:
    sys.path.insert(0, str(ROOT / "ndlocr-lite-gui"))

import ocr
import pdf_output_utils
from deim import DEIM
from ndl_parser import categories_org_name_index
from tools.ndlkoten2tei import convert_tei


def _load_gui_modules():
    spec = importlib.util.spec_from_file_location(
        "ndlocr_lite_gui", ROOT / "ndlocr-lite-gui" / "__init__.py"
    )
    pkg = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = pkg
    spec.loader.exec_module(pkg)
    from ndlocr_lite_gui import gui_config, main, pipeline_utils

    return gui_config, main, pipeline_utils


def _make_sample_pdf(path: Path, pages: int = 4, size=(200, 150)) -> None:
    from reportlab.pdfgen import canvas

    pdf = canvas.Canvas(str(path), pagesize=size)
    for idx in range(pages):
        pdf.setFillColorRGB(0.1 * ((idx % 5) + 1), 0.2, 0.3)
        pdf.rect(20, 20, size[0] - 40, size[1] - 40, fill=1)
        pdf.drawString(40, size[1] // 2, f"Page {idx + 1}")
        pdf.showPage()
    pdf.save()


def _wait_until_no_threads(name_prefix: str, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not any(t.name.startswith(name_prefix) for t in threading.enumerate()):
            return True
        time.sleep(0.02)
    return False


class _StubDetector:
    def __init__(self):
        yaml_path = ROOT / "src" / "config" / "ndl.yaml"
        names = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))["names"]
        self.classes = {i: names[i] for i in sorted(names)}
        self.markdown_styles = {}
        self.colorlist = [(0, 0, 0), (255, 0, 0)] + [(0, 0, 200)] * max(16, len(self.classes))
        self.preprocess_calls = 0
        self.detect_from_preprocessed_calls = 0
        self.detect_calls = 0

    def preprocess_with_meta(self, img):
        self.preprocess_calls += 1
        max_wh = max(img.shape[0], img.shape[1])
        return np.zeros((1, 3, 16, 16), dtype=np.float32), max_wh

    def detect_from_preprocessed(self, tensor, max_wh):
        self.detect_from_preprocessed_calls += 1
        return [
            {
                "class_index": 1,
                "confidence": 0.9,
                "box": np.array([10, 10, min(80, max_wh), min(30, max_wh)], dtype=np.int32),
                "pred_char_count": 3.0,
                "pred_markdown_style": 0.0,
                "class_name": "line_main",
            }
        ]

    def detect(self, img):
        self.detect_calls += 1
        tensor, max_wh = self.preprocess_with_meta(img)
        return self.detect_from_preprocessed(tensor, max_wh)

    drawxml_detections = DEIM.drawxml_detections
    draw_detections = DEIM.draw_detections


class _StubRecognizer:
    def __init__(self, text="テスト行"):
        self.text = text
        self.batch_size = 8
        self.effective_precision = "fp32"
        self.requested_precision = "fp32"
        self.precision_fallback_occurred = False
        self.backend_profile = SimpleNamespace(
            display_name="GPU (CUDA)",
            active_backend="cuda",
            active_providers=["CUDAExecutionProvider"],
            fallback_occurred=False,
            warning_message=None,
            concurrent_session_run=False,
        )

    def read(self, img):
        return self.text

    def read_batch(self, imgs, batch_size=None):
        return [self.text for _ in imgs]


class _FakePage:
    """Minimal headless substitute for ``ft.Page`` covering UI wiring tests."""

    def __init__(self):
        self.title = ""
        self.theme_mode = ft.ThemeMode.SYSTEM
        self.locale_configuration = None
        self.window = SimpleNamespace(
            icon=None,
            width=1024,
            height=900,
            min_width=1024,
            min_height=900,
            left=0,
            top=0,
            maximized=False,
            title_bar_hidden=False,
            title_bar_buttons_hidden=False,
            always_on_top=False,
            opacity=1.0,
            bgcolor=None,
        )
        self.on_keyboard_event = None
        self.overlay = []
        self.services = []
        self.controls = []
        self.dialogs = []
        self.media = None
        self.bgcolor = None
        self.pending_tasks = []

    def clean(self):
        self.controls.clear()

    def add(self, *controls):
        self.controls.extend(controls)

    def update(self):
        pass

    def run_task(self, fn, *args, **kwargs):
        coro = fn(*args, **kwargs)
        self.pending_tasks.append(coro)
        return coro

    def show_dialog(self, dlg):
        dlg.open = True
        if dlg not in self.dialogs:
            self.dialogs.append(dlg)

    def pop_dialog(self):
        if self.dialogs:
            dlg = self.dialogs.pop()
            dlg.open = False
            return dlg
        return None


def _collect_controls_by_key(root_controls) -> dict[str, ft.Control]:
    found: dict[str, ft.Control] = {}
    stack = list(root_controls)
    seen = set()
    while stack:
        ctrl = stack.pop()
        if ctrl is None or id(ctrl) in seen:
            continue
        seen.add(id(ctrl))
        key = getattr(ctrl, "key", None)
        if key:
            found[str(key)] = ctrl
        for attr in ("controls", "actions"):
            children = getattr(ctrl, attr, None)
            if children:
                stack.extend(children)
        content = getattr(ctrl, "content", None)
        if isinstance(content, ft.Control):
            stack.append(content)
    return found


class GuiPipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gui_config, cls.app, cls.pipeline_utils = _load_gui_modules()

    def test_default_worker_and_pending_bounds(self):
        self.assertEqual(self.pipeline_utils.default_writer_workers(1), 2)
        self.assertEqual(self.pipeline_utils.default_writer_workers(4), 2)
        self.assertEqual(self.pipeline_utils.default_writer_workers(8), 4)
        self.assertEqual(self.pipeline_utils.default_writer_workers(16), 6)
        self.assertEqual(self.pipeline_utils.default_writer_workers(64), 6)
        self.assertEqual(self.pipeline_utils.default_max_pending_writes(2), 4)
        self.assertEqual(self.pipeline_utils.default_max_pending_writes(6), 8)
        self.assertEqual(self.pipeline_utils.default_max_pending_writes(16), 8)

    def test_pdf_page_prefetcher_opens_pdf_once_and_runs_deim_preprocess(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = Path(tmpdir) / "sample.pdf"
            _make_sample_pdf(pdf_path, pages=5)

            det = _StubDetector()
            open_calls = 0
            closed_docs = 0
            orig_pdf_doc = pypdfium2.PdfDocument

            class SpyPdfDocument:
                def __init__(self, stream):
                    nonlocal open_calls
                    open_calls += 1
                    self._doc = orig_pdf_doc(stream)

                def __len__(self):
                    return len(self._doc)

                def render(self, *args, **kwargs):
                    return self._doc.render(*args, **kwargs)

                def close(self):
                    nonlocal closed_docs
                    closed_docs += 1
                    return self._doc.close()

            spy_pdfium = SimpleNamespace(
                PdfDocument=SpyPdfDocument,
                PdfBitmap=pypdfium2.PdfBitmap,
            )

            async def consume():
                prefetcher = self.pipeline_utils.PdfPagePrefetcher(
                    pdf_path=pdf_path,
                    page_count=5,
                    render_scale=100.0 / 72.0,
                    detector=det,
                    prefetch_size=3,
                    pdfium_module=spy_pdfium,
                    np_module=np,
                )
                pages = []
                try:
                    while True:
                        item = await prefetcher.next_page()
                        if item is None:
                            break
                        pages.append(item)
                finally:
                    prefetcher.close()
                return pages

            pages = asyncio.run(consume())
            self.assertEqual(open_calls, 1)
            self.assertEqual(closed_docs, 1)
            self.assertEqual([p[0] for p in pages], [0, 1, 2, 3, 4])
            self.assertEqual(det.preprocess_calls, 5)
            for _, img, det_pre in pages:
                self.assertEqual(img.ndim, 3)
                self.assertEqual(img.shape[2], 3)
                self.assertIsNotNone(det_pre)
                self.assertEqual(det_pre[0].shape, (1, 3, 16, 16))
            self.assertTrue(_wait_until_no_threads("gui-pdf-render"))

    def test_pdf_page_prefetcher_cancel_and_error_cleanup(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = Path(tmpdir) / "cancel.pdf"
            _make_sample_pdf(pdf_path, pages=8)
            cancel_event = threading.Event()
            det = _StubDetector()

            async def run_cancel():
                prefetcher = self.pipeline_utils.PdfPagePrefetcher(
                    pdf_path=pdf_path,
                    page_count=8,
                    render_scale=1.0,
                    detector=det,
                    prefetch_size=2,
                    cancel_event=cancel_event,
                    pdfium_module=pypdfium2,
                    np_module=np,
                )
                try:
                    first = await prefetcher.next_page()
                    self.assertEqual(first[0], 0)
                    cancel_event.set()
                finally:
                    prefetcher.close()

            asyncio.run(run_cancel())
            self.assertTrue(_wait_until_no_threads("gui-pdf-render"))

            # Rendering exception propagates cleanly and does not leak threads or handles.
            closed_flag = False

            class BoomPdfDocument:
                def __init__(self, stream):
                    pass

                def render(self, *args, **kwargs):
                    raise RuntimeError("boom render")

                def close(self):
                    nonlocal closed_flag
                    closed_flag = True

            boom_pdfium = SimpleNamespace(
                PdfDocument=BoomPdfDocument,
                PdfBitmap=pypdfium2.PdfBitmap,
            )

            async def run_boom():
                prefetcher = self.pipeline_utils.PdfPagePrefetcher(
                    pdf_path=pdf_path,
                    page_count=3,
                    render_scale=1.0,
                    detector=det,
                    prefetch_size=2,
                    pdfium_module=boom_pdfium,
                    np_module=np,
                )
                try:
                    await prefetcher.next_page()
                finally:
                    prefetcher.close()

            with self.assertRaisesRegex(RuntimeError, "boom render"):
                asyncio.run(run_boom())
            self.assertTrue(closed_flag)
            self.assertTrue(_wait_until_no_threads("gui-pdf-render"))

    def test_image_prefetcher_order_cancel_and_cleanup(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = []
            for idx in range(4):
                p = Path(tmpdir) / f"img_{idx}.png"
                Image.fromarray(np.full((40, 50, 3), idx * 40, dtype=np.uint8)).save(p)
                paths.append(str(p))

            det = _StubDetector()

            async def run_all():
                prefetcher = self.pipeline_utils.ImagePrefetcher(
                    paths=paths,
                    load_image_fn=self.app.load_rgb_image_array,
                    detector=det,
                    prefetch_size=3,
                )
                items = []
                try:
                    while True:
                        item = await prefetcher.next_item()
                        if item is None:
                            break
                        items.append(item)
                finally:
                    prefetcher.close()
                return items

            items = asyncio.run(run_all())
            self.assertEqual([item[0] for item in items], [0, 1, 2, 3])
            self.assertEqual([item[1] for item in items], paths)
            self.assertEqual(det.preprocess_calls, 4)
            for idx, _, img, det_pre in items:
                self.assertEqual(int(img[0, 0, 0]), idx * 40)
                self.assertIsNotNone(det_pre)

            cancel_event = threading.Event()

            async def run_cancel():
                prefetcher = self.pipeline_utils.ImagePrefetcher(
                    paths=paths,
                    load_image_fn=self.app.load_rgb_image_array,
                    detector=det,
                    prefetch_size=2,
                    cancel_event=cancel_event,
                )
                try:
                    first = await prefetcher.next_item()
                    self.assertEqual(first[0], 0)
                    cancel_event.set()
                    self.assertIsNone(await prefetcher.next_item())
                finally:
                    prefetcher.close()

            asyncio.run(run_cancel())

    def test_bounded_async_writer_concurrency_fifo_backpressure_and_errors(self):
        async def run_checks():
            active = 0
            max_active = 0
            lock = threading.Lock()
            completed_order = []

            def work(idx, delay):
                nonlocal active, max_active
                with lock:
                    active += 1
                    max_active = max(max_active, active)
                time.sleep(delay)
                with lock:
                    active -= 1
                return idx

            writer = self.pipeline_utils.BoundedAsyncWriter(
                max_workers=3,
                max_pending=2,
                thread_name_prefix="test-writer",
            )
            try:
                # Item 0 takes longer than item 1, yet callback order must stay FIFO [0, 1].
                await writer.submit(work, 0, 0.12, on_complete=completed_order.append)
                await writer.submit(work, 1, 0.01, on_complete=completed_order.append)
                # Submitting item 2 triggers backpressure because max_pending == 2 -> awaits item 0.
                await writer.submit(work, 2, 0.05, on_complete=completed_order.append)
                self.assertEqual(completed_order[0], 0)
                await writer.submit(work, 3, 0.01, on_complete=completed_order.append)
                await writer.drain()
                self.assertEqual(completed_order, [0, 1, 2, 3])
                self.assertGreaterEqual(max_active, 2)
            finally:
                await writer.close()

            # Error propagation waits for later tasks to complete before raising.
            settled = []
            writer_err = self.pipeline_utils.BoundedAsyncWriter(max_workers=2, max_pending=4)

            def maybe_fail(idx):
                if idx == 0:
                    time.sleep(0.03)
                    raise ValueError("write failure 0")
                time.sleep(0.08)
                settled.append(idx)
                return idx

            await writer_err.submit(maybe_fail, 0)
            await writer_err.submit(maybe_fail, 1)
            with self.assertRaisesRegex(ValueError, "write failure 0"):
                await writer_err.close(suppress_errors=False)
            self.assertEqual(settled, [1])

            # If an earlier task already failed before the next submit(), flush_completed surfaces it early.
            writer_eager = self.pipeline_utils.BoundedAsyncWriter(max_workers=2, max_pending=4)

            def fail_now():
                raise ValueError("eager failure")

            try:
                await writer_eager.submit(fail_now)
                time.sleep(0.05)
                with self.assertRaisesRegex(ValueError, "eager failure"):
                    await writer_eager.submit(lambda: "never")
            finally:
                await writer_eager.close(suppress_errors=True)

            # suppress_errors=True swallows worker failures safely on cancellation.
            writer_sup = self.pipeline_utils.BoundedAsyncWriter(max_workers=2, max_pending=4)
            await writer_sup.submit(maybe_fail, 0)
            await writer_sup.close(suppress_errors=True)

        asyncio.run(run_checks())

    def test_convert_tei_is_thread_safe(self):
        sample_json = [
            {
                "contents": [
                    [
                        {
                            "boundingBox": [[10, 20], [10, 50], [80, 20], [80, 50]],
                            "id": 0,
                            "text": "テスト",
                        }
                    ]
                ],
                "imginfo": {
                    "img_name": "p1.png",
                    "img_path": "/tmp/p1.png",
                    "img_width": 200,
                    "img_height": 300,
                },
            }
        ]
        expected = convert_tei(sample_json)
        with ThreadPoolExecutor(max_workers=6) as pool:
            results = list(pool.map(lambda _: convert_tei(sample_json), range(24)))
        for res in results:
            self.assertEqual(res, expected)

    def test_drawxml_detections_uses_lossless_compress_level_1_for_png(self):
        det = _StubDetector()
        img = np.full((80, 100, 3), 240, dtype=np.uint8)
        xmlstr = '<OCRDATASET><PAGE IMAGENAME="p.png" WIDTH="100" HEIGHT="80"><LINE TYPE="本文" X="10" Y="12" WIDTH="30" HEIGHT="16"/></PAGE></OCRDATASET>'

        orig_save = Image.Image.save
        recorded_kwargs = []

        def spy_save(self_img, fp, format=None, **params):
            recorded_kwargs.append((str(fp), params))
            return orig_save(self_img, fp, format=format, **params)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_png = Path(tmpdir) / "viz_page.png"
            out_jpg = Path(tmpdir) / "viz_page.jpg"
            with patch.object(Image.Image, "save", spy_save):
                det.drawxml_detections(img, xmlstr, categories_org_name_index, str(out_png))
                det.drawxml_detections(img, xmlstr, categories_org_name_index, str(out_jpg))

            self.assertEqual(recorded_kwargs[0][1].get("compress_level"), 1)
            self.assertNotIn("compress_level", recorded_kwargs[1][1])

            decoded = np.asarray(Image.open(out_png).convert("RGB"))
            # Interior box remains untouched RGB (240, 240, 240); border uses line_main color (255, 0, 0).
            self.assertEqual(tuple(decoded[20, 25]), (240, 240, 240))
            self.assertEqual(tuple(decoded[12, 20]), (255, 0, 0))

    def test_write_pdf_page_outputs_precomputed_json_matches_default(self):
        page_result = {
            "page_xml": '<PAGE IMAGENAME="s_00001.png" WIDTH="100" HEIGHT="120"><LINE TYPE="本文" X="1" Y="2" WIDTH="10" HEIGHT="12" STRING="行1"/></PAGE>',
            "text": "行1",
            "json_lines": [
                {
                    "boundingBox": [[1, 2], [1, 14], [11, 2], [11, 14]],
                    "id": 0,
                    "text": "行1",
                }
            ],
            "img_width": 100,
            "img_height": 120,
            "img_name": "s_00001.png",
        }
        with tempfile.TemporaryDirectory() as tmp_a, tempfile.TemporaryDirectory() as tmp_b:
            pdf_output_utils.write_pdf_page_outputs(
                tmp_a,
                "s",
                0,
                page_result,
                source_path="/tmp/s.pdf",
                write_json=True,
                write_xml=True,
                write_txt=True,
                write_tei=True,
                convert_tei_func=convert_tei,
                include_xml_in_json=True,
                source_dpi=150.0,
            )
            precomputed = pdf_output_utils.build_pdf_page_json_obj(
                page_result,
                source_path="/tmp/s.pdf",
                include_xml=True,
                source_page_index=0,
                source_dpi=150.0,
            )
            pdf_output_utils.write_pdf_page_outputs(
                tmp_b,
                "s",
                0,
                page_result,
                source_path="/tmp/s.pdf",
                write_json=True,
                write_xml=True,
                write_txt=True,
                write_tei=True,
                convert_tei_func=convert_tei,
                include_xml_in_json=True,
                source_dpi=150.0,
                page_json_obj=precomputed,
            )
            for name in ("s_00001.json", "s_00001.xml", "s_00001.txt", "s_00001_tei.xml"):
                self.assertEqual(
                    Path(tmp_a, name).read_bytes(),
                    Path(tmp_b, name).read_bytes(),
                )

    def test_gui_pdf_dpi_editable_visualize_persisted_and_pdf_pipeline_parity(self):
        async def run_gui_test():
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_root = Path(tmpdir)
                out_dir = tmp_root / "out"
                out_dir.mkdir()
                conf_path = tmp_root / "userconf.yaml"
                conf_path.write_text(
                    yaml.dump(
                        {
                            "langcode": "ja",
                            "pdf": False,
                            "visualize": False,
                            "pdf_resolution": 300,
                            "selected_output_path": str(out_dir),
                            "json": True,
                            "xml": True,
                            "txt": True,
                            "tei": True,
                            "markdown": True,
                            "html": False,
                            "pdf_page_output": True,
                            "table_structure": False,
                        },
                        allow_unicode=True,
                    ),
                    encoding="utf-8",
                )
                sample_pdf = tmp_root / "book.pdf"
                _make_sample_pdf(sample_pdf, pages=5, size=(200, 150))

                img_dir = tmp_root / "imgs"
                img_dir.mkdir()
                for idx in range(2):
                    Image.fromarray(np.full((120, 160, 3), 220, dtype=np.uint8)).save(
                        img_dir / f"pic_{idx}.png"
                    )

                det = _StubDetector()
                rec100 = _StubRecognizer("本文テキスト")
                rec30 = _StubRecognizer("短行")
                rec50 = _StubRecognizer("中行")

                fake_page = _FakePage()
                args = self.app.parse_gui_args(["--device", "cpu"])

                def fake_load_models(*_a, **_kw):
                    self.app.load_runtime_modules()
                    return det, rec100, rec30, rec50

                parent_logger = logging.getLogger("ndlocr")
                orig_handlers = list(parent_logger.handlers)
                orig_log_path = self.app.GUI_LOG_PATH
                try:
                    with patch.object(self.app, "USER_CONFIG_PATH", conf_path), patch.object(
                        self.app, "APP_DATA_DIR", tmp_root
                    ), patch.object(self.app, "APP_TEMP_DIR", tmp_root / "tmp"), patch.object(
                        self.app, "PDFTMPPATH", str(tmp_root / "tmp" / "pdfprev")
                    ), patch.object(
                        self.app, "load_ocr_models", side_effect=fake_load_models
                    ), patch.object(
                        self.app, "load_table_recognizer", return_value=None
                    ), patch.object(
                        ft.Control, "update", lambda self: None
                    ):
                        self.app.main(fake_page, args=args, download_models=False)
                        # Drain initial initialize_models task queued via page.run_task
                        while fake_page.pending_tasks:
                            await fake_page.pending_tasks.pop(0)

                        # Trigger UI construction lookup from main_controls + customize_btn closure
                        by_key = _collect_controls_by_key(fake_page.controls)
                        chkbx_viz = by_key["visualize"]
                        ocr_btn = by_key["run_ocr"]
                        customize_btn = by_key["customize_output"]
                        pick_file_btn = by_key["select_input_file"]

                        # 1. Visualize checkbox initializes from config (False) and persists toggle.
                        self.assertFalse(chkbx_viz.value)
                        chkbx_viz.value = True
                        chkbx_viz.on_change(None)
                        reloaded_conf = yaml.safe_load(conf_path.read_text(encoding="utf-8"))
                        self.assertTrue(reloaded_conf["visualize"])

                        # 2. PDF resolution field stays editable even when chkbx_pdf is False!
                        customize_btn.on_click(None)
                        self.assertTrue(fake_page.dialogs)
                        dlg = fake_page.dialogs[-1]
                        dlg_keys = _collect_controls_by_key([dlg.content])
                        pdf_cb = dlg_keys["output_pdf"]
                        pdf_dpi_field = dlg_keys["output_pdf_resolution"]
                        save_btn = _collect_controls_by_key(dlg.actions)["customize_output_ok"]

                        self.assertFalse(pdf_cb.value)
                        self.assertFalse(bool(pdf_dpi_field.disabled))
                        pdf_cb.value = True
                        pdf_cb.on_change(None)
                        self.assertFalse(bool(pdf_dpi_field.disabled))
                        pdf_cb.value = False
                        pdf_cb.on_change(None)
                        self.assertFalse(bool(pdf_dpi_field.disabled))

                        # Edit DPI to 120 while chkbx_pdf is still False and save!
                        pdf_dpi_field.value = "120"
                        save_btn.on_click(None)
                        saved_conf = yaml.safe_load(conf_path.read_text(encoding="utf-8"))
                        self.assertEqual(saved_conf["pdf_resolution"], 120)

                        # 3. Run full 5-page PDF OCR via GUI pipeline, counting PdfDocument opens and DEIM calls.
                        self.app.load_runtime_modules()
                        open_paths_in_pdfium = 0
                        orig_pdfium = pypdfium2.PdfDocument

                        class CountingPdfDoc:
                            def __init__(self, stream_or_path):
                                nonlocal open_paths_in_pdfium
                                if hasattr(stream_or_path, "read"):
                                    open_paths_in_pdfium += 1
                                self._inner = orig_pdfium(stream_or_path)

                            def __len__(self):
                                return len(self._inner)

                            def render(self, *a, **kw):
                                return self._inner.render(*a, **kw)

                            def close(self):
                                return self._inner.close()

                        file_picker = next(
                            s for s in fake_page.services if isinstance(s, ft.FilePicker)
                        )

                        async def fake_pick_files(**_kw):
                            return [SimpleNamespace(path=str(sample_pdf))]

                        # Spy on _run_ocr_on_image_array kwargs to assert preprocessed_det & include_markdown=False
                        seen_ocr_kwargs = []
                        orig_run_ocr = ocr._run_ocr_on_image_array

                        def spy_run_ocr(**kwargs):
                            seen_ocr_kwargs.append({
                                "has_pre": kwargs.get("preprocessed_det") is not None,
                                "include_markdown": kwargs.get("include_markdown"),
                            })
                            return orig_run_ocr(**kwargs)

                        with patch.object(file_picker, "pick_files", side_effect=fake_pick_files), patch.object(
                            self.app.pypdfium2, "PdfDocument", CountingPdfDoc
                        ), patch.object(ocr, "_run_ocr_on_image_array", side_effect=spy_run_ocr):
                            await pick_file_btn.on_click(None)
                            await ocr_btn.on_click(None)

                        # Preview render (1) + allsum count (1) + page_count (1) + prefetcher render (1) = 4 opens (< 5 pages!)
                        # Previously this would have been 1 + 1 + 1 + 5 = 8 opens across 5 pages.
                        self.assertEqual(open_paths_in_pdfium, 4)
                        self.assertEqual(len(seen_ocr_kwargs), 5)
                        self.assertTrue(all(item["has_pre"] for item in seen_ocr_kwargs))
                        self.assertTrue(all(item["include_markdown"] is False for item in seen_ocr_kwargs))
                        self.assertEqual(det.detect_from_preprocessed_calls, 5)
                        self.assertEqual(det.detect_calls, 0)

                        # Verify per-page PDF outputs + viz PNG + combined Markdown exist and reflect 120 DPI
                        for page_num in range(1, 6):
                            stem = f"book_{page_num:05}"
                            self.assertTrue((out_dir / f"{stem}.txt").is_file())
                            self.assertTrue((out_dir / f"{stem}.xml").is_file())
                            self.assertTrue((out_dir / f"{stem}_tei.xml").is_file())
                            j = json.loads((out_dir / f"{stem}.json").read_text(encoding="utf-8"))
                            self.assertEqual(j["imginfo"]["source_dpi"], 120.0)
                            self.assertEqual(j["contents"][0][0]["text"], "短行")
                            self.assertTrue((out_dir / f"viz_{stem}.png").is_file())
                        self.assertTrue((out_dir / "book.md").is_file())

                        # 4. Also test standalone directory images through GUI pipeline with TEI & PDF outputs.
                        img_out_dir = tmp_root / "img_out"
                        img_out_dir.mkdir()

                        # Configure output dir via pick_output_result
                        dir_pick_btn = by_key["select_input_directory"]
                        out_pick_btn = by_key["select_output_directory"]
                        with patch.object(
                            file_picker, "get_directory_path", side_effect=[str(img_dir), str(img_out_dir)]
                        ):
                            await dir_pick_btn.on_click(None)
                            await out_pick_btn.on_click(None)
                        await ocr_btn.on_click(None)

                        for idx in range(2):
                            self.assertTrue((img_out_dir / f"pic_{idx}.txt").is_file())
                            self.assertTrue((img_out_dir / f"pic_{idx}.xml").is_file())
                            self.assertTrue((img_out_dir / f"pic_{idx}.json").is_file())
                            self.assertTrue((img_out_dir / f"pic_{idx}.md").is_file())
                            self.assertTrue((img_out_dir / f"viz_pic_{idx}.png").is_file())
                        self.assertTrue((img_out_dir / "pic_0_tei.xml").is_file())
                finally:
                    # Windows refuses to unlink ndlocr-lite.log while RotatingFileHandler keeps it open.
                    for h in list(parent_logger.handlers):
                        if h not in orig_handlers:
                            parent_logger.removeHandler(h)
                            h.close()
                    self.app.GUI_LOG_PATH = orig_log_path

        asyncio.run(run_gui_test())


if __name__ == "__main__":
    unittest.main()
