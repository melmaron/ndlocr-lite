import logging
from logging.handlers import RotatingFileHandler
import asyncio
import inspect
import flet as ft
from typing import List, Dict, Tuple
import sys
import os
from pathlib import Path
import platform
import subprocess
import threading

APP_DIR = Path(__file__).resolve().parent
if __package__:
    SOURCE_DIR = APP_DIR.parent / 'src'
    if not SOURCE_DIR.is_dir():
        # Wheels install the OCR modules and config beside the GUI package.
        SOURCE_DIR = APP_DIR.parent
else:
    SOURCE_DIR = APP_DIR / 'src'
    if not SOURCE_DIR.is_dir():
        SOURCE_DIR = APP_DIR.parent / 'src'
if str(SOURCE_DIR) not in sys.path:
    sys.path.append(str(SOURCE_DIR))

import xml.etree.ElementTree as ET
import time
from concurrent.futures import ThreadPoolExecutor
import json
import shutil
import argparse
import yaml
import io
import glob
import ctypes
import math
from io import BytesIO
if __package__:
    from .gui_config import build_default_config
    from .pipeline_utils import (
        BoundedAsyncWriter,
        ImagePrefetcher,
        PdfPagePrefetcher,
        default_max_pending_writes,
        default_writer_workers,
    )
else:
    from gui_config import build_default_config
    from pipeline_utils import (
        BoundedAsyncWriter,
        ImagePrefetcher,
        PdfPagePrefetcher,
        default_max_pending_writes,
        default_writer_workers,
    )
from uicomponent.localelabel import TRANSLATIONS
from uicomponent.textdaisy import TextDaisyExporter
from collections import Counter
from optional_features import load_optional_features
from ort_provider import (
    DEVICE_CHOICES,
    PRECISION_CHOICES,
    detect_available_backends,
    format_device_diagnostics,
    normalize_device_name,
    normalize_precision_name,
    ort_supports_deform_conv,
    prepare_backend_for_parallel_init,
    should_parallelize_model_init,
)
from pdf_output_utils import (
    build_pdf_page_json_obj,
    should_output_pdf_pages_separately,
    write_pdf_page_outputs,
)
from character_inclusion import apply_character_inclusion
from html_output import build_html_document
from version import __version__
from copy import deepcopy
name = 'NDLOCR-Lite-GUI'

_feature_modules = load_optional_features()
DATA_POOL_FEATURE = (
    _feature_modules.data_pool.create_feature()
    if _feature_modules.data_pool is not None
    else None
)
LLM_FEATURE = (
    _feature_modules.llm.create_feature()
    if _feature_modules.llm is not None
    else None
)
TABLE_STRUCTURE_FEATURE = (
    _feature_modules.table_structure.create_feature()
    if _feature_modules.table_structure is not None
    else None
)

np = None
Image = None
ImageGrab = None
ocr = None
pypdfium2 = None
convert_tei = None
categories_org_name_index = None
_runtime_modules_lock = threading.Lock()
_runtime_modules_ready = False
_REPORTLAB_LOCK = threading.Lock()


def load_runtime_modules():
    """Load OCR/native dependencies after the first GUI frame is available."""
    global np, Image, ImageGrab, ocr, pypdfium2
    global convert_tei, categories_org_name_index
    global _runtime_modules_ready

    if _runtime_modules_ready:
        return
    with _runtime_modules_lock:
        if _runtime_modules_ready:
            return

        import numpy as numpy_module
        from PIL import Image as pil_image_module
        from PIL import ImageFile as pil_image_file_module
        from PIL import ImageGrab as pil_image_grab_module
        import ocr as ocr_module
        import pypdfium2 as pypdfium2_module
        from tools.ndlkoten2tei import convert_tei as convert_tei_function
        from ndl_parser import categories_org_name_index as category_mapping

        pil_image_file_module.MAXBLOCK = 1024 * 1024 * 128
        np = numpy_module
        Image = pil_image_module
        ImageGrab = pil_image_grab_module
        ocr = ocr_module
        pypdfium2 = pypdfium2_module
        convert_tei = convert_tei_function
        categories_org_name_index = category_mapping
        _runtime_modules_ready = True


def _build_result_markdown(page_result, table_fragments):
    rendered = ocr.build_markdown_text_from_xml(
        str(page_result.get('page_xml', '')),
        table_markdowns=table_fragments,
        markdown_styles=page_result.get('markdown_styles'),
    )
    if rendered or (page_result.get('page_xml') and table_fragments):
        return rendered
    return str(page_result.get('markdown_text') or page_result.get('text') or '')


def build_gui_result_markdown(page_result, page_tables=None):
    """Build preview content with links instead of structured table bodies."""
    fragments = (
        TABLE_STRUCTURE_FEATURE.preview_fragments(page_tables)
        if TABLE_STRUCTURE_FEATURE is not None
        else []
    )
    return _build_result_markdown(page_result, fragments)


def build_output_result_markdown(page_result, page_tables=None):
    fragments = (
        TABLE_STRUCTURE_FEATURE.markdown_fragments(page_tables)
        if TABLE_STRUCTURE_FEATURE is not None
        else []
    )
    return _build_result_markdown(page_result, fragments)


def build_output_result_html(document_name, pages):
    from uuid import uuid4

    markdown_pages = []
    table_snippets = {}
    for page_result, page_tables in pages:
        fragments = (
            TABLE_STRUCTURE_FEATURE.markdown_fragments(page_tables)
            if TABLE_STRUCTURE_FEATURE is not None
            else []
        )
        for fragment, item in zip(fragments, page_tables or []):
            if item.get('html') and not item.get('error'):
                token = 'NDLOCRTABLE' + uuid4().hex
                table_snippets[token] = str(item['html'])
                fragment['markdown'] = token
        markdown_pages.append(_build_result_markdown(page_result, fragments))
    return build_html_document(document_name, markdown_pages, table_snippets)


def open_local_path_sync(local_path, *, fallback_uri=None):
    import webbrowser

    target = str(local_path)
    if sys.platform == 'win32':
        os.startfile(target)
        return
    if sys.platform == 'darwin':
        subprocess.run(['open', target], check=True)
        return
    try:
        subprocess.run(['xdg-open', target], check=True)
        return
    except (FileNotFoundError, subprocess.SubprocessError):
        uri = fallback_uri
        if uri is None:
            try:
                uri = Path(target).resolve().as_uri()
            except ValueError:
                uri = target
        if not webbrowser.open(uri):
            raise OSError(f'Unable to open preview link: {target}')


async def open_local_path(local_path):
    await asyncio.to_thread(open_local_path_sync, local_path)


async def open_preview_link(event):
    import webbrowser
    from urllib.parse import urlparse
    from urllib.request import url2pathname

    url = str(event.data)
    parsed = urlparse(url)
    if parsed.scheme == 'file':
        path = ('//' + parsed.netloc if parsed.netloc else '') + parsed.path
        local_path = url2pathname(path)
        await asyncio.to_thread(open_local_path_sync, local_path, fallback_uri=url)
        return
    if not await asyncio.to_thread(webbrowser.open, url):
        raise OSError(f'Unable to open preview link: {url}')


def build_preview_markdown_style():
    """Return a compact style shared by every formatted OCR preview."""
    return ft.MarkdownStyleSheet(
        p_text_style=ft.TextStyle(size=13),
        code_text_style=ft.TextStyle(size=12),
        h1_text_style=ft.TextStyle(size=22, weight=ft.FontWeight.BOLD),
        h2_text_style=ft.TextStyle(size=20, weight=ft.FontWeight.BOLD),
        h3_text_style=ft.TextStyle(size=18, weight=ft.FontWeight.BOLD),
        h4_text_style=ft.TextStyle(size=16, weight=ft.FontWeight.W_600),
        h5_text_style=ft.TextStyle(size=14, weight=ft.FontWeight.W_600),
        h6_text_style=ft.TextStyle(size=13, weight=ft.FontWeight.W_600),
        blockquote_text_style=ft.TextStyle(size=13),
        list_bullet_text_style=ft.TextStyle(size=13),
        table_head_text_style=ft.TextStyle(size=12, weight=ft.FontWeight.W_600),
        table_body_text_style=ft.TextStyle(size=12),
        block_spacing=8,
    )


def resolve_main_shortcut(key, *, ctrl=False, meta=False, shift=False, alt=False):
    """Return the main-screen action assigned to a keyboard event."""
    normalized_key = str(key or '').strip().lower().replace(' ', '')
    primary = bool(ctrl or meta)

    if normalized_key in ('escape', 'esc') and not any((primary, shift, alt)):
        return 'stop_ocr'

    if primary:
        if normalized_key == 'o':
            if shift and not alt:
                return 'select_input_directory'
            if alt and not shift:
                return 'select_output_directory'
            if not shift and not alt:
                return 'select_input_file'
        if normalized_key in ('enter', 'return', 'numpadenter') and not alt:
            return 'crop_ocr' if shift else 'run_ocr'
        if normalized_key in (',', 'comma') and not shift and not alt:
            return 'customize_output'
        return None

    if alt and not shift:
        return {
            'c': 'capture_ocr',
            'p': 'preview_previous',
            'n': 'preview_next',
            'm': 'toggle_markdown',
        }.get(normalized_key)

    return None


def page_has_open_dialog(page):
    """Return whether either the test page or Flet Page has an active dialog."""
    dialog_stack = getattr(page, 'dialogs', None)
    if dialog_stack is not None:
        return bool(dialog_stack)
    dialog_registry = getattr(page, '_dialogs', None)
    return bool(getattr(dialog_registry, 'controls', None))


def get_character_inclusion_transform(config_obj):
    if config_obj.get('character_inclusion', False):
        return apply_character_inclusion
    return None


ASSETS_DIR = APP_DIR / 'assets'
DEFAULT_IMAGE_SRC = str(ASSETS_DIR / 'dummy.dat')
APP_DATA_DIR = Path(os.environ.get('FLET_APP_STORAGE_DATA', os.getcwd())).resolve()
APP_TEMP_DIR = Path(
    os.environ.get('FLET_APP_STORAGE_TEMP', str(APP_DATA_DIR / 'tmp'))
).resolve()
PDFTMPPATH = str(APP_TEMP_DIR / '4ab7ecc3-53fb-b3e7-64e8-a809b5a483d2')
USER_CONFIG_PATH = APP_DATA_DIR / 'userconf.yaml'
LOGGER = logging.getLogger('ndlocr.gui')
GUI_LOG_PATH: Path | None = None
_LOG_HANDLER_ATTR = '_ndlocr_gui_log_path'


def configure_gui_logging(log_dir=None, filename: str = 'ndlocr-lite.log') -> Path | None:
    """Configure persistent rotating file logging for GUI diagnostics and ORT warnings."""
    global GUI_LOG_PATH
    candidates = [Path(log_dir)] if log_dir is not None else [APP_DATA_DIR, APP_TEMP_DIR]
    parent_logger = logging.getLogger('ndlocr')
    parent_logger.setLevel(logging.INFO)
    parent_logger.propagate = False

    for candidate_dir in candidates:
        try:
            resolved_dir = Path(candidate_dir).resolve()
            resolved_dir.mkdir(parents=True, exist_ok=True)
            target_path = resolved_dir / filename

            existing = None
            for handler in list(parent_logger.handlers):
                recorded = getattr(handler, _LOG_HANDLER_ATTR, None)
                if recorded is not None:
                    if Path(recorded) == target_path:
                        existing = handler
                    else:
                        parent_logger.removeHandler(handler)
                        try:
                            handler.close()
                        except Exception:
                            pass

            if existing is None:
                handler = RotatingFileHandler(
                    target_path,
                    maxBytes=2 * 1024 * 1024,
                    backupCount=3,
                    encoding='utf-8',
                )
                handler.setLevel(logging.INFO)
                handler.setFormatter(
                    logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
                )
                setattr(handler, _LOG_HANDLER_ATTR, str(target_path))
                parent_logger.addHandler(handler)

            GUI_LOG_PATH = target_path
            return target_path
        except OSError:
            continue
    return None


def get_windows_scale_factor():
    try:
        ctypes.windll.user32.SetProcessDPIAware()
        hdc = ctypes.windll.user32.GetDC(0)
        dpi = ctypes.windll.gdi32.GetDeviceCaps(hdc, 88)
        ctypes.windll.user32.ReleaseDC(0, hdc)
        return dpi / 96.0
    except Exception:
        return 1.0


def get_display_scale_factor(page=None) -> float:
    """Return display scale factor on Windows, macOS, and Linux."""
    if sys.platform == 'win32':
        return get_windows_scale_factor()
    media = getattr(page, 'media', None) if page is not None else None
    dpr = getattr(media, 'device_pixel_ratio', None)
    try:
        if dpr is not None:
            dpr_val = float(dpr)
            if math.isfinite(dpr_val) and dpr_val > 0:
                return dpr_val
    except (TypeError, ValueError):
        pass
    for env_name in ('GDK_SCALE', 'QT_SCALE_FACTOR'):
        raw = os.environ.get(env_name)
        if not raw:
            continue
        try:
            val = float(raw)
            if math.isfinite(val) and val > 0:
                return val
        except (TypeError, ValueError):
            continue
    return 1.0


class RecogLine:
    def __init__(self, npimg: 'np.ndarray', idx: float, pred_char_cnt: int, pred_str: str = ''):
        self.npimg = npimg
        self.idx = idx
        self.pred_char_cnt = pred_char_cnt
        self.pred_str = pred_str

    def __lt__(self, other):
        return self.idx < other.idx


def process_cascade(alllineobj: RecogLine, recognizer30, recognizer50, recognizer100, is_cascade=True):
    load_runtime_modules()
    return ocr.process_cascade(
        alllineobj,
        recognizer30,
        recognizer50,
        recognizer100,
        is_cascade=is_cascade,
    )


def _warmup_gpu_model(model):
    profile = getattr(model, 'backend_profile', None)
    active_backend = getattr(profile, 'active_backend', 'cpu') if profile is not None else 'cpu'
    if active_backend != 'cpu':
        warmup_fn = getattr(model, 'warmup', None)
        if callable(warmup_fn):
            try:
                warmup_fn()
            except Exception as exc:
                print(
                    f'[WARN] GPU warm-up skipped for {type(model).__name__} on {active_backend}: {exc}',
                    file=sys.stderr,
                )
    return model


def _init_and_warmup_model(factory, *factory_args):
    return _warmup_gpu_model(factory(*factory_args))


def load_ocr_models(args, download_models=False, reuse_detector=None):
    """Create OCR ONNX sessions (optionally reusing detector when only PARSeq settings change)."""
    if download_models:
        from model_download import resolve_model_paths

        resolve_model_paths(args, SOURCE_DIR / 'model')
    load_runtime_modules()
    if hasattr(ocr, '_load_charlist'):
        ocr._load_charlist(os.path.abspath(args.rec_classes))

    effective_device = getattr(args, 'device', None) or 'auto'
    prepare_backend_for_parallel_init(effective_device)
    if should_parallelize_model_init(effective_device):
        workers = 3 if reuse_detector is not None else 4
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix='model-init') as executor:
            detector_future = (
                None
                if reuse_detector is not None
                else executor.submit(_init_and_warmup_model, ocr.get_detector, args)
            )
            recognizer100_future = executor.submit(_init_and_warmup_model, ocr.get_recognizer, args)
            recognizer30_future = executor.submit(_init_and_warmup_model, ocr.get_recognizer, args, args.rec_weights30)
            recognizer50_future = executor.submit(_init_and_warmup_model, ocr.get_recognizer, args, args.rec_weights50)
            detector = reuse_detector if reuse_detector is not None else detector_future.result()
            return (
                detector,
                recognizer100_future.result(),
                recognizer30_future.result(),
                recognizer50_future.result(),
            )

    detector = reuse_detector if reuse_detector is not None else _init_and_warmup_model(ocr.get_detector, args)
    recognizer100 = _init_and_warmup_model(ocr.get_recognizer, args)
    recognizer30 = _init_and_warmup_model(ocr.get_recognizer, args, args.rec_weights30)
    recognizer50 = _init_and_warmup_model(ocr.get_recognizer, args, args.rec_weights50)
    return detector, recognizer100, recognizer30, recognizer50


def load_table_recognizer(args, download_models=False):
    if TABLE_STRUCTURE_FEATURE is not None and not TABLE_STRUCTURE_FEATURE.is_supported():
        raise RuntimeError(
            TABLE_STRUCTURE_FEATURE.unsupported_reason('ja')
            or 'Table structure estimation is unsupported on this ONNX Runtime version.'
        )
    if download_models and args.table_weights is None:
        from model_download import get_model_path, TABLE_MODEL

        args.table_weights = str(get_model_path(*TABLE_MODEL, SOURCE_DIR / 'model'))
    recognizer = TABLE_STRUCTURE_FEATURE.create_recognizer(args)
    recognizer.load()
    return recognizer


def load_rgb_image_array(input_path: str) -> 'np.ndarray':
    load_runtime_modules()
    with Image.open(input_path) as pil_image:
        return np.asarray(pil_image.convert('RGB'))


def render_pdf_page_array(pdf_path: str, page_index: int, scale: float) -> 'np.ndarray':
    """Render one PDF page via binary stream without sharing PDFium objects across threads."""
    load_runtime_modules()
    with Path(pdf_path).open('rb') as pdf_stream:
        pdf_doc = pypdfium2.PdfDocument(pdf_stream)
        try:
            rendered_pages = pdf_doc.render(
                pypdfium2.PdfBitmap.to_pil,
                page_indices=[page_index],
                scale=scale,
            )
            pil_page = next(iter(rendered_pages))
            try:
                rgb_page = pil_page.convert('RGB')
                try:
                    return np.array(rgb_page, copy=True)
                finally:
                    if rgb_page is not pil_page and callable(getattr(rgb_page, 'close', None)):
                        rgb_page.close()
            finally:
                if callable(getattr(pil_page, 'close', None)):
                    pil_page.close()
        finally:
            pdf_doc.close()


def _write_pdf_page_task(
    *,
    output_dir: str,
    output_stem: str,
    page_index: int,
    page_name: str,
    img,
    page_result: dict,
    page_json_obj: dict,
    source_path: str,
    source_dpi: float,
    separate_page_outputs: bool,
    write_json: bool,
    write_xml: bool,
    write_txt: bool,
    write_tei: bool,
    write_html: bool,
    write_viz: bool,
    page_tables: list,
    detector,
    categories_mapping: dict,
    convert_tei_func,
    build_html_fn,
) -> str | None:
    """Write Stage 3 per-page PDF artifacts and optional visualization image."""
    if separate_page_outputs and (write_json or write_xml or write_txt or write_tei):
        write_pdf_page_outputs(
            output_dir,
            output_stem,
            page_index,
            page_result,
            source_path=source_path,
            write_json=write_json,
            write_xml=write_xml,
            write_txt=write_txt,
            write_tei=write_tei,
            convert_tei_func=convert_tei_func if write_tei else None,
            include_xml_in_json=True,
            source_dpi=source_dpi,
            page_json_obj=page_json_obj,
        )
    if write_html and separate_page_outputs:
        Path(output_dir, f'{output_stem}_{page_index + 1:05}.html').write_text(
            build_html_fn(page_name, [(page_result, page_tables)]),
            encoding='utf-8',
        )
    if write_viz:
        viz_path = os.path.join(output_dir, f'viz_{page_name}')
        detector.drawxml_detections(
            npimg=img,
            xmlstr='<OCRDATASET>\n' + page_result['page_xml'] + '\n</OCRDATASET>',
            categories=categories_mapping,
            outputimgpath=viz_path,
        )
        if os.path.exists(viz_path):
            return viz_path
    return None


def _write_image_page_task(
    *,
    inputpath: str,
    imgname: str,
    outputpath: str,
    output_stem: str,
    img,
    allxmlstr: str,
    alltextlist: list,
    resjsonarray: list,
    alljsonobj: dict,
    page_result: dict,
    page_tables: list,
    output_markdown: str,
    write_xml: bool,
    write_viz: bool,
    write_json: bool,
    write_txt: bool,
    write_markdown: bool,
    write_html: bool,
    write_pdf: bool,
    pdf_viztxt: bool,
    pdf_resolution: int,
    detector,
    categories_mapping: dict,
    build_html_fn,
    create_pdf_fn,
) -> tuple[str | None, str | None]:
    """Write Stage 3 artifacts for a single image file."""
    viz_result: str | None = None
    pdf_result: str | None = None

    if write_xml:
        with open(os.path.join(outputpath, output_stem + '.xml'), 'w', encoding='utf-8') as wf:
            wf.write(allxmlstr)

    if write_viz:
        output_vizpath = os.path.join(outputpath, 'viz_' + os.path.basename(inputpath))
        if output_vizpath.split('.')[-1] == 'jp2':
            output_vizpath = output_vizpath[:-4] + '.jpg'
        detector.drawxml_detections(
            npimg=img,
            xmlstr=allxmlstr,
            categories=categories_mapping,
            outputimgpath=output_vizpath,
        )
        if os.path.exists(output_vizpath):
            viz_result = output_vizpath

    if write_json:
        with open(os.path.join(outputpath, output_stem + '.json'), 'w', encoding='utf-8') as wf:
            wf.write(json.dumps(alljsonobj, ensure_ascii=False, indent=2))

    if write_txt:
        with open(os.path.join(outputpath, output_stem + '.txt'), 'w', encoding='utf-8') as wtf:
            wtf.write('\n'.join(alltextlist))

    if write_markdown:
        with open(os.path.join(outputpath, output_stem + '.md'), 'w', encoding='utf-8') as wf:
            wf.write(output_markdown)

    if write_html:
        Path(outputpath, output_stem + '.html').write_text(
            build_html_fn(imgname, [(page_result, page_tables)]),
            encoding='utf-8',
        )

    if write_pdf:
        pdf_outpath = os.path.join(outputpath, output_stem + '.pdf')
        create_pdf_fn(
            pdf_outpath,
            img,
            resjsonarray,
            pdf_viztxt,
            resolution=pdf_resolution,
        )
        pdf_result = pdf_outpath

    return viz_result, pdf_result


class ImageSelector:
    def __init__(self, page: ft.Page, config_obj: Dict, detector=None, recognizer30=None, recognizer50=None, recognizer100=None, table_feature=None, table_recognizer=None, outputdirpath=None, width: int = 600, height: int = 600, busy_tracker=None):
        self.cnt = 0
        self.page = page
        self.config_obj = config_obj
        self.langcode = config_obj['langcode']
        self.inputpathlist = []
        self.outputdirpath = outputdirpath

        self.image_src = DEFAULT_IMAGE_SRC
        self.dialog_width = width
        self.dialog_height = height
        self.page_index = 0
        self.detector = detector
        self.recognizer30 = recognizer30
        self.recognizer50 = recognizer50
        self.recognizer100 = recognizer100
        self.table_feature = table_feature
        self.table_recognizer = table_recognizer
        self.busy_tracker = busy_tracker

        self.start_x = 0
        self.start_y = 0

        self.selection_box = ft.Container(
            left=0,
            top=0,
            width=0,
            height=0,
            border=ft.Border.all(2, ft.Colors.BLUE),
            bgcolor=ft.Colors.TRANSPARENT,
        )

        self.overlay = ft.GestureDetector(
            content=ft.Container(
                width=self.dialog_width,
                height=self.dialog_height,
                bgcolor=ft.Colors.TRANSPARENT,
            ),
            on_pan_start=self.pan_start,
            on_pan_update=self.pan_update,
            on_pan_end=self.pan_end,
        )
        self.img = ft.Image(src=self.image_src, width=self.dialog_width, height=self.dialog_height, fit=ft.BoxFit.CONTAIN)
        self.imgzm = ft.Image(src=self.image_src, width=self.dialog_width, height=self.dialog_height, fit=ft.BoxFit.CONTAIN)
        self.image_stack = ft.Stack(
            width=self.dialog_width,
            height=self.dialog_height,
            controls=[
                self.img,
                self.selection_box,
                self.overlay,
            ],
        )
        self.cropocr_btn = ft.Button(TRANSLATIONS['imageselector_cropocr_btn'][self.langcode], on_click=self.crop_region)
        self.dialog = ft.AlertDialog(
            modal=True,
            content=self.image_stack,
            actions=[
                ft.Button(TRANSLATIONS['imageselector_zoom_btn'][self.langcode], icon=ft.Icons.ZOOM_IN, on_click=self.open_zoom_page),
                ft.Button(TRANSLATIONS['imageselector_prev_btn'][self.langcode], on_click=self.prev_page),
                ft.Button(TRANSLATIONS['imageselector_next_btn'][self.langcode], on_click=self.next_page),
                self.cropocr_btn,
                ft.Button(TRANSLATIONS['common_cancel'][self.langcode], on_click=self.close_dialog),
            ],
        )
        zoom_img = ft.InteractiveViewer(
            min_scale=1,
            max_scale=10,
            boundary_margin=ft.Margin.all(20),
            content=self.imgzm,
        )

        self.zoom_dialog = ft.AlertDialog(
            modal=True,
            content=zoom_img,
            actions=[
                ft.Button(TRANSLATIONS['common_cancel'][self.langcode], on_click=self.close_zoom_page),
            ],
        )
        self.resulttext = ft.Markdown(
            value='',
            selectable=True,
            extension_set=ft.MarkdownExtensionSet.GITHUB_WEB,
            on_tap_link=open_preview_link,
            md_style_sheet=build_preview_markdown_style(),
            soft_line_break=True,
            visible=bool(self.config_obj.get('result_markdown_display', False)),
        )
        self.result_plain_text = ft.Text(
            value='',
            selectable=True,
            visible=not bool(self.config_obj.get('result_markdown_display', False)),
        )
        self.result_view_switch = ft.Switch(
            label='整形表示（Markdown）',
            value=bool(self.config_obj.get('result_markdown_display', False)),
            on_change=self.change_result_view_mode,
        )

        self.crop_image = ft.Image(src=self.image_src, width=300, height=300, fit=ft.BoxFit.CONTAIN)
        crop_image_col = ft.Column(
            controls=[self.crop_image],
            width=300,
            height=300,
            expand=False,
        )
        self.crop_image_int = ft.InteractiveViewer(
            min_scale=1,
            max_scale=5,
            boundary_margin=ft.Margin.all(20),
            content=crop_image_col,
        )
        self.result_text_col = ft.Column(
            controls=[self.result_view_switch, self.result_plain_text, self.resulttext],
            scroll=ft.ScrollMode.ALWAYS,
            width=800,
            height=300,
            expand=False,
        )

        self.result_dialog = ft.AlertDialog(
            title=ft.Text(TRANSLATIONS['imageselector_result_title'][self.langcode]),
            modal=True,
            content=ft.Row([self.crop_image_int, self.result_text_col]),
            actions=[
                ft.Button('OK', on_click=self.close_result_page),
            ],
        )

    def change_result_view_mode(self, e=None):
        use_markdown = bool(self.result_view_switch.value)
        self.config_obj['result_markdown_display'] = use_markdown
        self.resulttext.visible = use_markdown
        self.result_plain_text.visible = not use_markdown
        self.page.update()

    def open_result_page(self):
        if self.dialog.open:
            self.page.pop_dialog()
        self.page.show_dialog(self.result_dialog)

    def close_result_page(self, e):
        if self.result_dialog.open:
            self.page.pop_dialog()
        self.page.show_dialog(self.dialog)

    def set_image(self, inputpathlist):
        self.cnt = 0
        self.inputpathlist = inputpathlist
        self.page_index = 0
        if not inputpathlist:
            return
        self.image_src = inputpathlist[self.page_index]
        self.img.src = inputpathlist[self.page_index]
        self.imgzm.src = inputpathlist[self.page_index]
        self.page.update()

    def set_outputdir(self, outputdirpath):
        self.outputdirpath = outputdirpath

    def open_zoom_page(self, e):
        if self.dialog.open:
            self.page.pop_dialog()
        self.page.show_dialog(self.zoom_dialog)

    def close_zoom_page(self, e):
        if self.zoom_dialog.open:
            self.page.pop_dialog()
        self.page.show_dialog(self.dialog)

    def pan_start(self, e: ft.DragStartEvent):
        self.start_x = e.local_position.x
        self.start_y = e.local_position.y
        self.selection_box.left = self.start_x
        self.selection_box.top = self.start_y
        self.selection_box.width = 0
        self.selection_box.height = 0
        self.page.update()

    def pan_update(self, e: ft.DragUpdateEvent):
        cur_x, cur_y = e.local_position.x, e.local_position.y
        left = min(self.start_x, cur_x)
        top = min(self.start_y, cur_y)
        width = abs(cur_x - self.start_x)
        height = abs(cur_y - self.start_y)
        self.selection_box.left = left
        self.selection_box.top = top
        self.selection_box.width = width
        self.selection_box.height = height
        self.page.update()

    def pan_end(self, e: ft.DragEndEvent):
        self.page.update()

    def open_dialog(self, e):
        if not self.dialog.open:
            self.page.show_dialog(self.dialog)

    def prev_page(self, e):
        if not self.inputpathlist:
            return
        if self.page_index > 0:
            self.page_index -= 1
        else:
            self.page_index = len(self.inputpathlist) - 1
        self.img.src = self.inputpathlist[self.page_index]
        self.imgzm.src = self.inputpathlist[self.page_index]
        self.page.update()

    def next_page(self, e):
        if not self.inputpathlist:
            return
        if self.page_index < len(self.inputpathlist) - 1:
            self.page_index += 1
        else:
            self.page_index = 0
        self.img.src = self.inputpathlist[self.page_index]
        self.imgzm.src = self.inputpathlist[self.page_index]
        self.page.update()

    def crop_region(self, e):
        pilimg = Image.open(self.img.src)
        pilimg = pilimg.convert('RGB')
        rwidth, rheight = pilimg.size
        if rheight < rwidth:
            window_h = self.dialog_height * rheight / rwidth
            window_w = self.dialog_width
            offset_h = (window_w - window_h) / 2
            offset_w = 0
        else:
            window_h = self.dialog_height
            window_w = self.dialog_width * rwidth / rheight
            offset_w = (window_h - window_w) / 2
            offset_h = 0
        hratio = rheight / window_h
        wratio = rwidth / window_w
        cropx = int((self.selection_box.left - offset_w) * wratio)
        cropy = int((self.selection_box.top - offset_h) * hratio)
        cropw = int(self.selection_box.width * wratio)
        croph = int(self.selection_box.height * hratio)
        if cropx > 0 and cropy > 0 and cropw > 10 and croph > 0:
            im_crop = pilimg.crop((cropx, cropy, cropx + cropw, cropy + croph))
        else:
            return
        buff = BytesIO()
        im_crop.save(buff, 'png')
        self.crop_image.src = buff.getvalue()
        self.outputcroppedpath = os.path.join(PDFTMPPATH, os.path.splitext(os.path.basename(self.image_src))[0] + '_cropped_{}.jpg'.format(self.cnt))
        self.page.run_task(self.mini_ocr, im_crop, self.outputcroppedpath)
        self.cnt += 1
        self.page.update()

    async def mini_ocr(self, im_crop, outputcroppedpath=None):
        self.cropocr_btn.disabled = True
        self.page.update()
        inputname = os.path.basename(outputcroppedpath or self.outputcroppedpath)

        npimg = np.array(im_crop)
        if self.busy_tracker is not None:
            self.busy_tracker.enter()
        try:
            page_result = await asyncio.to_thread(
                ocr._run_ocr_on_image_array,
                detector=self.detector,
                recognizer30=self.recognizer30,
                recognizer50=self.recognizer50,
                recognizer100=self.recognizer100,
                inputname=inputname,
                img=npimg,
                outputpath=self.outputdirpath or os.getcwd(),
                save_viz=False,
                text_transform=get_character_inclusion_transform(self.config_obj),
                include_markdown=False,
            )
            page_tables = []
            if (
                self.table_feature is not None
                and self.config_obj.get('table_structure', True)
                and self.table_recognizer is not None
            ):
                page_result, page_tables = await asyncio.to_thread(
                    self.table_feature.infer,
                    npimg,
                    page_result,
                    self.table_recognizer,
                )
            alltextlist = [page_result['text']]
            output_dir = self.outputdirpath or os.getcwd()
            output_stem = os.path.splitext(os.path.basename(inputname))[0]
            if page_tables:
                self.table_feature.write_html_files(
                    inputname, page_tables, output_dir, output_stem,
                )
            display_markdown = build_gui_result_markdown(page_result, page_tables)
            if page_result['line_count'] == 0 or (
                page_result['line_count'] > 0 and page_result['vertical_line_count'] / page_result['line_count'] > 0.5
            ):
                alltextlist = alltextlist[::-1]
            os.makedirs(self.outputdirpath or os.getcwd(), exist_ok=True)
            if self.config_obj.get('html', False):
                Path(output_dir, output_stem + '.html').write_text(
                    build_output_result_html(inputname, [(page_result, page_tables)]),
                    encoding='utf-8',
                )
            with open(os.path.join(self.outputdirpath or os.getcwd(), os.path.splitext(os.path.basename(inputname))[0] + '.txt'), 'w', encoding='utf-8') as wtf:
                wtf.write('\n'.join(alltextlist))
            self.result_plain_text.value = page_result['text']
            self.resulttext.value = display_markdown
        except Exception as ex:
            self.result_plain_text.value = f'エラーが発生しました: {ex}'
            self.resulttext.value = f'エラーが発生しました: {ex}'
        finally:
            self.cropocr_btn.disabled = False
            self.open_result_page()
            self.page.update()
            if self.busy_tracker is not None:
                self.busy_tracker.leave()

    def close_dialog(self, e):
        if self.dialog.open:
            self.page.pop_dialog()


class CaptureTool:
    def __init__(self, page: ft.Page, config_obj: Dict, detector=None, recognizer30=None, recognizer50=None, recognizer100=None, table_feature=None, table_recognizer=None, width: int = 400, height: int = 300, busy_tracker=None):
        self.page = page
        self.config_obj = config_obj
        self.langcode = config_obj['langcode']
        self.detector = detector
        self.recognizer30 = recognizer30
        self.recognizer50 = recognizer50
        self.recognizer100 = recognizer100
        self.table_feature = table_feature
        self.table_recognizer = table_recognizer
        self.busy_tracker = busy_tracker
        self.dialog_width = width
        self.dialog_height = height
        self.im_crop = None
        self.img_str = ''
        self.result_jsonstr = ''
        self.outputdirpath = os.getcwd()
        self.scale_factor = get_display_scale_factor(self.page)
        self.start_x = 0
        self.start_y = 0
        self.current_x = 0
        self.current_y = 0

        self.original_width = 0
        self.original_height = 0
        self.original_left = 0
        self.original_top = 0
        self.original_bgcolor = None

        self.selection_box = ft.Container(
            border=ft.Border.all(2, ft.Colors.RED),
            bgcolor=ft.Colors.with_opacity(0.2, ft.Colors.RED),
            visible=False,
        )
        self.img_control = ft.Image(
            src=b'',
            width=self.dialog_width,
            height=self.dialog_height,
            fit=ft.BoxFit.CONTAIN,
            gapless_playback=True,
        )
        self.retry_btn = ft.Button(TRANSLATIONS['capturetool_retry_btn'][self.langcode], on_click=self.start_capture)
        self.cboc_fixed = ft.Checkbox(label=TRANSLATIONS['capturetool_fixregion'][self.langcode], value=False, disabled=True)
        self.cb_table_structure = (
            self.table_feature.create_capture_control(self.config_obj)
            if self.table_feature is not None
            else None
        )
        self.ocr_btn = ft.Button(TRANSLATIONS['capturetool_ocr_button'][self.langcode], on_click=self.mini_ocr)

        self.errorlog = ft.Text('')
        self.dialog_content = ft.Column(
            controls=[
                self.errorlog,
                ft.Container(
                    content=self.img_control,
                    border=ft.Border.all(1, ft.Colors.GREY),
                    alignment=ft.Alignment.CENTER,
                ),
            ],
            alignment=ft.MainAxisAlignment.CENTER,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            tight=True,
        )

        self.dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text(TRANSLATIONS['capturetool_result_title'][self.langcode]),
            content=self.dialog_content,
            actions=[
                ft.Row([
                    self.retry_btn,
                    ft.Column(
                        controls=[
                            self.cboc_fixed,
                            *(
                                [self.cb_table_structure]
                                if self.cb_table_structure is not None
                                else []
                            ),
                        ],
                        spacing=0,
                        tight=True,
                    ),
                    self.ocr_btn,
                    ft.Button(TRANSLATIONS['common_close'][self.langcode], on_click=self.close_dialog),
                ]),
            ],
            actions_alignment=ft.MainAxisAlignment.END,
        )
        self.resulttext = ft.Markdown(
            value='',
            selectable=True,
            extension_set=ft.MarkdownExtensionSet.GITHUB_WEB,
            on_tap_link=open_preview_link,
            md_style_sheet=build_preview_markdown_style(),
            soft_line_break=True,
            visible=bool(self.config_obj.get('result_markdown_display', False)),
        )
        self.result_plain_text = ft.Text(
            value='',
            selectable=True,
            visible=not bool(self.config_obj.get('result_markdown_display', False)),
        )
        self.result_view_switch = ft.Switch(
            label='整形表示（Markdown）',
            value=bool(self.config_obj.get('result_markdown_display', False)),
            on_change=self.change_result_view_mode,
        )
        self.resultsmessage = ft.Text(value='', selectable=True)
        self.result_crop_image = ft.Image(src=b'', width=300, height=300, fit=ft.BoxFit.CONTAIN)

        self.crop_image_int = ft.InteractiveViewer(
            min_scale=1,
            max_scale=5,
            boundary_margin=ft.Margin.all(20),
            content=ft.Column([self.result_crop_image], width=300, height=300),
        )

        self.result_text_col = ft.Column(
            controls=[self.result_view_switch, self.result_plain_text, self.resulttext],
            scroll=ft.ScrollMode.ALWAYS,
            width=600,
            height=300,
        )
        self.result_dialog = ft.AlertDialog(
            title=ft.Text(TRANSLATIONS['capturetool_resultocr_title'][self.langcode]),
            modal=True,
            content=ft.Row(
                controls=[self.crop_image_int, self.result_text_col],
                alignment=ft.MainAxisAlignment.START,
                vertical_alignment=ft.CrossAxisAlignment.START,
            ),
            actions=[
                self.resultsmessage,
                ft.Button(TRANSLATIONS['common_close'][self.langcode], on_click=self.close_result_page),
            ],
        )
        self.overlay_stack = ft.Stack(
            controls=[
                ft.GestureDetector(
                    on_pan_start=self._on_pan_start,
                    on_pan_update=self._on_pan_update,
                    on_pan_end=self._on_pan_end,
                    drag_interval=10,
                ),
                self.selection_box,
            ],
            expand=True,
            visible=False,
        )

        self.page.overlay.append(self.overlay_stack)

    def start_capture(self, e=None):
        if self.dialog.open:
            self.close_dialog(e)
        if self.cboc_fixed.value:
            self._capture_and_restore(self.x1_phys, self.y1_phys, self.x2_phys, self.y2_phys)
            return
        self.scale_factor = get_display_scale_factor(self.page)
        self.original_width = self.page.window.width
        self.original_height = self.page.window.height
        self.original_left = self.page.window.left
        self.original_top = self.page.window.top
        self.original_bgcolor = self.page.bgcolor

        self.page.window.maximized = True
        self.page.window.title_bar_hidden = True
        self.page.window.title_bar_buttons_hidden = True
        self.page.window.always_on_top = True
        self.page.window.opacity = 0.3
        self.page.window.bgcolor = ft.Colors.TRANSPARENT
        self.page.bgcolor = ft.Colors.with_opacity(0.3, ft.Colors.BLACK)

        self.overlay_stack.visible = True
        self.page.update()

    def _on_pan_start(self, e: ft.DragStartEvent):
        self.start_x = e.local_position.x
        self.start_y = e.local_position.y
        self.current_x = e.local_position.x
        self.current_y = e.local_position.y
        self.selection_box.visible = True
        self.selection_box.left = self.start_x
        self.selection_box.top = self.start_y
        self.selection_box.width = 0
        self.selection_box.height = 0
        self.page.update()

    def _on_pan_update(self, e: ft.DragUpdateEvent):
        self.current_x = e.local_position.x
        self.current_y = e.local_position.y

        left = min(self.start_x, self.current_x)
        top = min(self.start_y, self.current_y)
        width = abs(self.current_x - self.start_x)
        height = abs(self.current_y - self.start_y)

        self.selection_box.left = left
        self.selection_box.top = top
        self.selection_box.width = width
        self.selection_box.height = height
        self.page.update()

    def _on_pan_end(self, e: ft.DragEndEvent):
        x1_local = min(self.start_x, self.current_x)
        y1_local = min(self.start_y, self.current_y)
        x2_local = max(self.start_x, self.current_x)
        y2_local = max(self.start_y, self.current_y)

        offset_x = self.page.window.left or 0
        offset_y = self.page.window.top or 0

        x1_global = x1_local + offset_x
        y1_global = y1_local + offset_y
        x2_global = x2_local + offset_x
        y2_global = y2_local + offset_y

        self.x1_phys = int(x1_global * self.scale_factor)
        self.y1_phys = int(y1_global * self.scale_factor)
        self.x2_phys = int(x2_global * self.scale_factor)
        self.y2_phys = int(y2_global * self.scale_factor)

        self._capture_and_restore(self.x1_phys, self.y1_phys, self.x2_phys, self.y2_phys)

    def _capture_and_restore(self, x1, y1, x2, y2):
        self.page.window.opacity = 0
        self.page.update()
        time.sleep(0.2)
        if (x2 - x1) > 5 and (y2 - y1) > 5:
            try:
                self.im_crop = ImageGrab.grab(bbox=(x1, y1, x2, y2)).convert('RGB')
                self.cboc_fixed.disabled = False
                buffered = io.BytesIO()
                self.im_crop.save(buffered, format='png')
                image_bytes = buffered.getvalue()
                self.img_control.src = image_bytes
                self.result_crop_image.src = image_bytes
                self.page.show_dialog(self.dialog)
            except Exception as ex:
                print(f'Capture failed: {ex}')

        self.overlay_stack.visible = False
        self.selection_box.visible = False

        self.page.window.opacity = 1
        self.page.window.maximized = False
        self.page.window.title_bar_hidden = False
        self.page.window.title_bar_buttons_hidden = False
        self.page.window.always_on_top = False
        self.page.window.bgcolor = ft.Colors.WHITE
        self.page.bgcolor = self.original_bgcolor
        self.page.update()
        time.sleep(0.2)

        self.page.window.width = self.original_width
        self.page.window.height = self.original_height
        self.page.window.left = self.original_left
        self.page.window.top = self.original_top
        self.page.update()

    def change_result_view_mode(self, e=None):
        use_markdown = bool(self.result_view_switch.value)
        self.config_obj['result_markdown_display'] = use_markdown
        self.resulttext.visible = use_markdown
        self.result_plain_text.visible = not use_markdown
        self.page.update()

    async def mini_ocr(self, e):
        if self.im_crop is None:
            return
        if self.busy_tracker is not None:
            self.busy_tracker.enter()
        self.ocr_btn.disabled = True
        self.resultsmessage.value = ''
        self.page.update()
        try:
            allstart = time.time()
            filename_base = f'captureimg_{time.time_ns()}'
            npimg = np.array(self.im_crop)

            page_result = await asyncio.to_thread(
                ocr._run_ocr_on_image_array,
                detector=self.detector,
                recognizer30=self.recognizer30,
                recognizer50=self.recognizer50,
                recognizer100=self.recognizer100,
                inputname=filename_base,
                img=npimg,
                outputpath=self.outputdirpath,
                save_viz=False,
                text_transform=get_character_inclusion_transform(self.config_obj),
                include_markdown=False,
            )
            page_tables = []
            if (
                self.table_feature is not None
                and self.cb_table_structure is not None
                and self.cb_table_structure.value
                and self.table_recognizer is not None
            ):
                page_result, page_tables = await asyncio.to_thread(
                    self.table_feature.infer,
                    npimg,
                    page_result,
                    self.table_recognizer,
                )

            if page_tables:
                self.table_feature.write_html_files(
                    filename_base, page_tables, self.outputdirpath, filename_base,
                )
            if self.config_obj.get('html', False):
                Path(self.outputdirpath).mkdir(parents=True, exist_ok=True)
                Path(self.outputdirpath, filename_base + '.html').write_text(
                    build_output_result_html(filename_base, [(page_result, page_tables)]),
                    encoding='utf-8',
                )
            final_text = build_gui_result_markdown(page_result, page_tables)
            self.resultsmessage.value = '{:.2f} sec'.format(time.time() - allstart)
            self.result_plain_text.value = page_result['text']
            self.resulttext.value = final_text
            self.result_jsonstr = json.dumps(page_result['json_lines'], ensure_ascii=False)
            self.open_result_page()

        except Exception as e:
            print(f'OCR Error: {e}')
            self.result_plain_text.value = f'エラーが発生しました: {e}'
            self.resulttext.value = f'エラーが発生しました: {e}'
            self.open_result_page()
        finally:
            self.ocr_btn.disabled = False
            self.page.update()
            if self.busy_tracker is not None:
                self.busy_tracker.leave()

    def open_dialog(self, e=None):
        self.start_capture()

    def close_dialog(self, e):
        if self.dialog.open:
            self.page.pop_dialog()

    def open_result_page(self):
        if self.dialog.open:
            self.page.pop_dialog()
        self.page.show_dialog(self.result_dialog)

    def close_result_page(self, e):
        if self.result_dialog.open:
            self.page.pop_dialog()
        self.page.show_dialog(self.dialog)

def parse_gui_args(argv=None, download_models=False):
    parser = argparse.ArgumentParser(description='Argument for Inference using ONNXRuntime')
    parser.add_argument('--det-weights', type=str, required=False, help='Path to rtmdet onnx file', default=str(SOURCE_DIR / 'model' / 'deim-s-1024x1024-261124.onnx'))
    parser.add_argument('--det-classes', type=str, required=False, help='Path to list of class in yaml file', default=str(SOURCE_DIR / 'config' / 'ndl.yaml'))
    parser.add_argument('--det-score-threshold', type=float, required=False, default=0.2)
    parser.add_argument('--det-conf-threshold', type=float, required=False, default=0.2)
    parser.add_argument('--det-iou-threshold', type=float, required=False, default=0.2)
    parser.add_argument('--det-overlap-threshold', type=float, default=0.7,
                        help='Remove smaller same-label regions covered by this fraction (0 to disable, otherwise >0 and <=1)')
    parser.add_argument('--rec-weights30', type=str, required=False, help='Path to parseq-tiny onnx file', default=str(SOURCE_DIR / 'model' / 'parseq-ndl-24x256-30-tiny-499epoch-261124.onnx'))
    parser.add_argument('--rec-weights50', type=str, required=False, help='Path to parseq-tiny onnx file', default=str(SOURCE_DIR / 'model' / 'parseq-ndl-24x384-50-tiny-278epoch-261124.onnx'))
    parser.add_argument('--rec-weights', type=str, required=False, help='Path to parseq-tiny onnx file', default=str(SOURCE_DIR / 'model' / 'parseq-ndl-24x768-100-tiny-153epoch-261124.onnx'))
    parser.add_argument('--rec-classes', type=str, required=False, help='Path to list of class in yaml file', default=str(SOURCE_DIR / 'config' / 'NDLmoji.yaml'))
    parser.add_argument(
        '--device',
        type=str,
        required=False,
        help='Execution device (auto, cpu, cuda, migraphx, rocm, openvino, webgpu, dml)',
        choices=list(DEVICE_CHOICES),
        default=None,
    )
    parser.add_argument('--device-id', type=int, required=False, default=None, help='GPU device/adapter index')
    parser.add_argument('--batch-size', type=int, required=False, default=None, help='PARSeq batch size (0 for automatic)')
    parser.add_argument(
        '--precision',
        type=str,
        required=False,
        choices=list(PRECISION_CHOICES),
        default=None,
        help='PARSeq inference precision (fp32 or fp16)',
    )
    parser.add_argument('--strict-device', action='store_true', default=False, help='Fail instead of falling back to CPU')
    if TABLE_STRUCTURE_FEATURE is not None:
        TABLE_STRUCTURE_FEATURE.add_arguments(parser, SOURCE_DIR)
    if download_models:
        from model_download import MODELS

        parser.set_defaults(**{option: None for option in MODELS})
        if TABLE_STRUCTURE_FEATURE is not None:
            parser.set_defaults(table_weights=None)
        return parser.parse_args(argv)
    # Standalone Flet builds may append their own host arguments.
    args, _ = parser.parse_known_args(argv)
    return args


def resolve_gui_engine_args(args, config_obj, cli_overrides=None):
    """Apply saved config defaults to engine args unless explicitly overridden via CLI."""
    overrides = cli_overrides if cli_overrides is not None else {}
    device = overrides.get('device') if 'device' in overrides else config_obj.get('device', 'auto')
    device_id = overrides.get('device_id') if 'device_id' in overrides else config_obj.get('device_id', 0)
    batch_size = overrides.get('batch_size') if 'batch_size' in overrides else config_obj.get('batch_size', 0)
    precision = overrides.get('precision') if 'precision' in overrides else config_obj.get('precision', 'fp32')

    if not isinstance(device, str):
        device = 'auto'
    normalized_device = normalize_device_name(device)
    if normalized_device not in DEVICE_CHOICES:
        normalized_device = 'auto'
    args.device = normalized_device
    try:
        args.device_id = max(0, int(device_id))
    except (TypeError, ValueError):
        args.device_id = 0
    try:
        args.batch_size = max(0, int(batch_size))
    except (TypeError, ValueError):
        args.batch_size = 0
    if not isinstance(precision, str):
        precision = 'fp32'
    normalized_precision = normalize_precision_name(precision)
    if normalized_precision not in PRECISION_CHOICES:
        normalized_precision = 'fp32'
    args.precision = normalized_precision
    return args


def extract_engine_state(args_obj) -> dict:
    """Return normalized execution engine configuration fields from an args namespace."""
    raw_prec = getattr(args_obj, 'precision', 'fp32')
    norm_prec = normalize_precision_name(raw_prec) if isinstance(raw_prec, str) else 'fp32'
    if norm_prec not in PRECISION_CHOICES:
        norm_prec = 'fp32'
    raw_dev = getattr(args_obj, 'device', 'auto')
    norm_dev = normalize_device_name(raw_dev) if isinstance(raw_dev, str) else 'auto'
    if norm_dev not in DEVICE_CHOICES:
        norm_dev = 'auto'
    try:
        device_id = max(0, int(getattr(args_obj, 'device_id', 0) or 0))
    except (TypeError, ValueError):
        device_id = 0
    try:
        batch_size = max(0, int(getattr(args_obj, 'batch_size', 0) or 0))
    except (TypeError, ValueError):
        batch_size = 0
    return {
        'device': norm_dev,
        'device_id': device_id,
        'batch_size': batch_size,
        'precision': norm_prec,
    }


def compute_engine_reload_scope(loaded_engine_state, target_engine_state) -> str:
    """Decide whether a reload needs all models, only PARSeq recognizers, or nothing."""
    if not loaded_engine_state or not target_engine_state:
        return 'all'
    if (
        loaded_engine_state.get('device') != target_engine_state.get('device')
        or int(loaded_engine_state.get('device_id', 0)) != int(target_engine_state.get('device_id', 0))
    ):
        return 'all'
    if (
        int(loaded_engine_state.get('batch_size', 0)) != int(target_engine_state.get('batch_size', 0))
        or loaded_engine_state.get('precision', 'fp32') != target_engine_state.get('precision', 'fp32')
    ):
        return 'recognizers'
    return 'none'


def _normalize_warning_signature(message: str) -> str:
    """Normalize per-model labels out of legacy warning strings for deduplication."""
    import re

    text = message.strip()
    text = re.sub(
        r"(CPU で実行します)\s*\((?:[^()]+|\([^()]*\))*\)(:)",
        r"\1\2",
        text,
    )
    text = re.sub(
        r"(指定されたデバイス '[^']+' を )\S+( で有効化できなかったため)",
        r"\1<model>\2",
        text,
    )
    text = re.sub(
        r"(Requested device '[^']+' failed for )[^:]+(:)",
        r"\1<model>\2",
        text,
    )
    return text


def _strip_model_label_from_warning(message: str) -> str:
    """Remove per-model parenthetical tags from grouped warning text for display."""
    import re

    cleaned = message.strip()
    cleaned = re.sub(
        r"(CPU で実行します)\s*\((?:[^()]+|\([^()]*\))*\)(:)",
        r"\1\2",
        cleaned,
    )
    return cleaned


def collect_engine_warnings(
    detector,
    recognizer,
    recognizer30=None,
    recognizer50=None,
    table_recognizer=None,
    table_error=None,
) -> dict:
    """Collect deduplicated fallback warnings and table errors grouped across models."""
    model_entries = (
        ('DEIMv2', detector),
        ('PARSeq-100', recognizer),
        ('PARSeq-30', recognizer30),
        ('PARSeq-50', recognizer50),
        ('TableCenterNet', table_recognizer),
    )
    groups = []
    group_index_by_key = {}
    fallback_occurred = False

    for model_name, model in model_entries:
        profile = getattr(model, 'backend_profile', None)
        if profile is None:
            continue
        if getattr(profile, 'fallback_occurred', False):
            fallback_occurred = True
        msg = getattr(profile, 'warning_message', None)
        reason = getattr(profile, 'fallback_reason', None)
        hint = getattr(profile, 'fallback_hint', None)
        category = getattr(profile, 'fallback_category', None)

        cleaned = msg.strip() if isinstance(msg, str) else ''
        if not cleaned and not (isinstance(reason, str) and reason.strip()):
            continue

        if category or (isinstance(reason, str) and reason.strip()):
            key = ('structured', category or '', (reason or '').strip())
        else:
            key = ('raw', _normalize_warning_signature(cleaned), '')

        idx = group_index_by_key.get(key)
        if idx is None:
            group_index_by_key[key] = len(groups)
            groups.append({
                'models': [model_name],
                'category': category,
                'reason': reason.strip() if isinstance(reason, str) and reason.strip() else None,
                'hint': hint.strip() if isinstance(hint, str) and hint.strip() else None,
                'raw_warning': cleaned,
            })
        elif model_name not in groups[idx]['models']:
            groups[idx]['models'].append(model_name)

    warnings = []
    for group in groups:
        if group['raw_warning']:
            display_msg = (
                _strip_model_label_from_warning(group['raw_warning'])
                if len(group['models']) > 1
                else group['raw_warning']
            )
            if display_msg not in warnings:
                warnings.append(display_msg)
        elif group['reason']:
            if group['reason'] not in warnings:
                warnings.append(group['reason'])

    table_warning = str(table_error).strip() if table_error is not None else None
    if table_warning == '':
        table_warning = None
    return {
        'fallback_occurred': fallback_occurred,
        'warnings': warnings,
        'grouped_warnings': groups,
        'table_warning': table_warning,
    }


def _concise_reason_for_display(group: dict | None, raw_warning: str, langcode: str) -> str:
    """Return a concise single-line root-cause summary without noisy prefixes."""
    import re

    category = (group or {}).get('category') if isinstance(group, dict) else None
    reason = (group or {}).get('reason') if isinstance(group, dict) else None
    en_by_category = {
        'cuda_driver_missing': 'NVIDIA GPU driver (nvcuda.dll / libcuda.so.1) not found',
        'cuda_driver_old': 'Installed NVIDIA driver is too old for CUDA 13 (R580+ required)',
        'cuda_no_device': 'No CUDA-capable NVIDIA GPU detected',
        'cuda_runtime_missing': 'Bundled CUDA 13 / cuDNN 9 libraries could not be loaded',
        'webgpu_vulkan_missing': 'Vulkan runtime (libvulkan.so.1) not found',
        'webgpu_d3d12_missing': 'DirectX 12 runtime (d3d12.dll) not found',
        'webgpu_dxc_error': 'WebGPU shader compilation failed (DXC)',
        'webgpu_plugin_error': 'WebGPU plugin failed to load',
        'openvino_plugin_error': 'OpenVINO plugin failed to load',
        'dml_deim_unsupported': 'Layout detection (DEIMv2) runs on CPU while PARSeq uses DirectML GPU',
    }
    if langcode == 'en' and category in en_by_category:
        return en_by_category[category]
    if reason:
        return re.sub(r'^(cuda|webgpu|openvino|migraphx|rocm|dml)\s*:\s*', '', reason, flags=re.IGNORECASE)

    first_line = (raw_warning or '').splitlines()[0].strip() if raw_warning else ''
    for pat in (
        r'^\[WARN\]\s*',
        r'^自動選択されたGPUバックエンドの初期化に失敗したため\s*CPU\s*で実行します\s*(?:\((?:[^()]+|\([^()]*\))*\))?\s*:\s*',
        r'^指定されたデバイス\s*\'[^\']+\'\s*を.*?フォールバックしました。\s*',
        r'^(cuda|webgpu|openvino|migraphx|rocm|dml)\s*:\s*',
    ):
        first_line = re.sub(pat, '', first_line, flags=re.IGNORECASE).strip()
    return first_line


def format_fallback_banner_summary(
    warning_info: dict,
    langcode: str = 'ja',
    init_error=None,
) -> str:
    """Format a concise 1-line warning banner without duplicated fallback prefixes."""
    if init_error is not None:
        return (
            f'OCRモデル初期化に失敗しました: {init_error}'
            if langcode == 'ja'
            else f'OCR model initialization failed: {init_error}'
        )
    if warning_info.get('fallback_occurred'):
        groups = warning_info.get('grouped_warnings') or []
        first_group = groups[0] if groups else None
        raw_warnings = warning_info.get('warnings') or []
        raw_first = raw_warnings[0] if raw_warnings else ''
        reason_text = _concise_reason_for_display(first_group, raw_first, langcode)
        base = TRANSLATIONS['engine_fallback_banner_short'][langcode]
        if reason_text:
            return f'{base} — {reason_text}'
        return base
    if warning_info.get('table_warning'):
        prefix = TRANSLATIONS['table_structure_unavailable_prefix'][langcode]
        return f"{prefix}: {warning_info['table_warning'].splitlines()[0]}"
    return ''


def build_friendly_diagnostics_guidance(
    warning_info: dict,
    langcode: str = 'ja',
    init_error=None,
    active_backend: str | None = None,
) -> dict:
    """Build plain-language guidance steps and helpful links for the Diagnostics dialog."""
    nvidia_url = 'https://www.nvidia.com/drivers'
    releases_url = 'https://github.com/melmaron/ndlocr-lite-gpu/releases'
    is_ja = langcode != 'en'

    if init_error is not None:
        return {
            'severity': 'error',
            'category': 'init_error',
            'headline': (
                f'OCRモデルの読み込みに失敗しました: {init_error}'
                if is_ja
                else f'Failed to initialize OCR models: {init_error}'
            ),
            'steps': (
                [
                    '配布フォルダを日本語（全角文字）を含まない場所（例: C:\\ndlocr-lite\\）へ展開し直してください。',
                    '下の「ログフォルダを開く」から ndlocr-lite.log を確認するか、「診断結果をコピー」して Issue へお送りください。',
                ]
                if is_ja
                else [
                    'Re-extract the archive to a path without non-ASCII characters (e.g. C:\\ndlocr-lite\\).',
                    'Open the log folder below to inspect ndlocr-lite.log, or copy diagnostics for an issue report.',
                ]
            ),
            'links': [
                (TRANSLATIONS['engine_diagnostics_open_releases'][langcode], releases_url),
            ],
        }

    if not warning_info.get('fallback_occurred'):
        if warning_info.get('table_warning'):
            return {
                'severity': 'warning',
                'category': 'table_warning',
                'headline': (
                    f"{TRANSLATIONS['table_structure_unavailable_prefix'][langcode]}: "
                    f"{warning_info['table_warning'].splitlines()[0]}"
                ),
                'steps': (
                    ['通常の文字認識（OCR）はそのまま利用できます。表構造認識を有効にするには ONNX Runtime 1.25 以上が必要です。']
                    if is_ja
                    else ['Standard text OCR continues to work normally. Table structure recognition requires ONNX Runtime >= 1.25.']
                ),
                'links': [],
            }
        backend_label = active_backend or 'cpu'
        is_gpu = backend_label not in ('cpu', 'none', '')
        return {
            'severity': 'ok',
            'category': None,
            'headline': (
                TRANSLATIONS['engine_diagnostics_summary_ok'][langcode]
                if is_gpu
                else TRANSLATIONS['engine_diagnostics_summary_cpu'][langcode]
            ),
            'steps': (
                ['問題なく動作しています。GPU を利用したい場合は「出力設定」から実行デバイスを確認できます。']
                if is_ja and not is_gpu
                else (
                    ['GPU バックエンドは正常に初期化されています。このまま高速 OCR を利用できます。']
                    if is_ja
                    else ['No action needed.']
                )
            ),
            'links': [],
        }

    groups = warning_info.get('grouped_warnings') or []
    first_group = groups[0] if groups else None
    category = (first_group or {}).get('category')
    raw_first = (first_group or {}).get('raw_warning') or (
        (warning_info.get('warnings') or [''])[0]
    )
    reason_text = _concise_reason_for_display(first_group, raw_first, langcode)

    if category == 'cuda_driver_missing':
        headline = reason_text or (
            'NVIDIA GPU ドライバが見つからないため CPU で動作しています'
            if is_ja
            else 'NVIDIA GPU driver not found — running on CPU'
        )
        steps = (
            [
                'このパッケージには CUDA 13 + cuDNN 9 ランタイムが同梱されているため、CUDA Toolkit の別途インストールは不要です。',
                'NVIDIA GPU 搭載 PC の場合は、NVIDIA 公式ドライバ（R580 以降）をインストールして PC を再起動してください。',
                'AMD Radeon / Intel Arc 搭載 PC の場合は、Releases ページから gpu_generic 版をダウンロードしてご利用ください。',
                'ドライバ更新前でも、CPU モードのまま OCR は正常に利用できます。',
            ]
            if is_ja
            else [
                'CUDA 13 and cuDNN 9 runtimes are already bundled in this package (no separate CUDA Toolkit needed).',
                'If you have an NVIDIA GPU, install the latest NVIDIA driver (R580 or newer) and restart.',
                'If you have an AMD Radeon or Intel Arc GPU, download the gpu_generic package from Releases.',
                'OCR continues to work normally on CPU in the meantime.',
            ]
        )
        links = [
            (TRANSLATIONS['engine_diagnostics_open_nvidia_driver'][langcode], nvidia_url),
            (TRANSLATIONS['engine_diagnostics_open_releases'][langcode], releases_url),
        ]
    elif category == 'cuda_driver_old':
        headline = reason_text
        steps = (
            [
                'CUDA 13 + cuDNN 9 ランタイム自体は同梱されていますが、お使いの NVIDIA GPU ドライバが旧バージョンのため GPU を起動できませんでした。',
                'NVIDIA 公式ページから最新ドライバ（R580 以降）をインストールして再起動してください（CUDA Toolkit の導入は不要です）。',
            ]
            if is_ja
            else [
                'CUDA 13 + cuDNN 9 runtimes are bundled, but your NVIDIA display driver is older than R580.',
                'Install the latest official NVIDIA driver (R580+) and reboot (no CUDA Toolkit install needed).',
            ]
        )
        links = [
            (TRANSLATIONS['engine_diagnostics_open_nvidia_driver'][langcode], nvidia_url),
            (TRANSLATIONS['engine_diagnostics_open_releases'][langcode], releases_url),
        ]
    elif category == 'cuda_no_device':
        headline = reason_text
        steps = (
            [
                '認識可能な NVIDIA GPU が見つかりませんでした。GPU 搭載環境ではデバイス認識と NVIDIA ドライバを確認してください。',
                'AMD Radeon / Intel Arc 搭載環境の場合は、Releases ページの gpu_generic 版をご利用ください。',
            ]
            if is_ja
            else [
                'No NVIDIA CUDA-capable GPU was detected.',
                'If you use AMD Radeon or Intel Arc graphics, please download the gpu_generic package from Releases.',
            ]
        )
        links = [
            (TRANSLATIONS['engine_diagnostics_open_releases'][langcode], releases_url),
        ]
    elif category == 'cuda_runtime_missing':
        headline = reason_text
        steps = (
            [
                '同梱の CUDA 13 / cuDNN 9 ライブラリを読み込めませんでした。圧縮ファイルを日本語（全角文字）を含まないフォルダへ一式すべて再展開してください。',
                'セキュリティソフト等により site-packages\\nvidia 配下の DLL が隔離されていないか確認してください。',
                'NVIDIA 公式ドライバが最新（R580 以降）かもあわせてご確認ください。',
            ]
            if is_ja
            else [
                'Bundled CUDA 13 / cuDNN 9 libraries failed to load. Re-extract the full archive into a path with ASCII characters only.',
                'Check that antivirus software did not quarantine files under site-packages/nvidia.',
                'Also ensure your NVIDIA GPU driver is R580 or newer.',
            ]
        )
        links = [
            (TRANSLATIONS['engine_diagnostics_open_releases'][langcode], releases_url),
            (TRANSLATIONS['engine_diagnostics_open_nvidia_driver'][langcode], nvidia_url),
        ]
    elif category == 'webgpu_vulkan_missing':
        headline = reason_text
        steps = (
            [
                'Linux 環境で WebGPU を使うには Vulkan ローダ (libvulkan.so.1) と GPU 用 Vulkan ドライバが必要です。',
                'Ubuntu / Debian 系の例: sudo apt install libvulkan1 mesa-vulkan-drivers を実行してアプリを再起動してください。',
            ]
            if is_ja
            else [
                'WebGPU on Linux requires the Vulkan loader (libvulkan.so.1) and a GPU Vulkan driver.',
                'On Ubuntu/Debian, install `libvulkan1` and `mesa-vulkan-drivers` (or your vendor GPU driver), then restart.',
            ]
        )
        links = [
            (TRANSLATIONS['engine_diagnostics_open_releases'][langcode], releases_url),
        ]
    elif category == 'webgpu_d3d12_missing':
        headline = reason_text
        steps = (
            [
                'DirectX 12 ランタイム (d3d12.dll) が見つかりませんでした。Windows 10 (1909 以降) / 11 と最新の GPU ドライバをご確認ください。',
            ]
            if is_ja
            else [
                'DirectX 12 runtime (d3d12.dll) was not found. Ensure Windows 10 (1909+)/11 and updated GPU drivers are installed.',
            ]
        )
        links = [
            (TRANSLATIONS['engine_diagnostics_open_releases'][langcode], releases_url),
        ]
    elif category == 'webgpu_dxc_error':
        headline = reason_text
        steps = (
            [
                'NVIDIA GPU 環境をお使いの場合は、Releases ページから cuda 版（windows_cuda.zip）をご利用ください。',
                'AMD Radeon / Intel Arc をお使いの場合は、GPU ドライバを最新版へ更新して再起動してください。',
            ]
            if is_ja
            else [
                'If you have an NVIDIA GPU, please use the cuda package (windows_cuda.zip) from Releases.',
                'If you have an AMD Radeon or Intel Arc GPU, update your graphics driver and restart.',
            ]
        )
        links = [
            (TRANSLATIONS['engine_diagnostics_open_releases'][langcode], releases_url),
        ]
    elif category in ('webgpu_plugin_error', 'openvino_plugin_error'):
        headline = reason_text
        steps = (
            [
                'GPU プラグインの読み込みに失敗しました。配布アーカイブを日本語を含まないパスへ欠けなく再展開してください。',
                'GPU ドライバを最新版へ更新してから再起動してください。',
            ]
            if is_ja
            else [
                'GPU plugin failed to load. Re-extract the full archive to an ASCII-only folder.',
                'Update your GPU drivers and restart.',
            ]
        )
        links = [
            (TRANSLATIONS['engine_diagnostics_open_releases'][langcode], releases_url),
        ]
    elif category == 'dml_deim_unsupported':
        headline = reason_text
        steps = (
            ['レイアウト検出のみ精度保護のため CPU で実行し、文字認識 (PARSeq) は DirectML GPU で実行しています。そのままご利用いただけます。']
            if is_ja
            else ['Layout detection runs on CPU for accuracy protection while PARSeq runs on DirectML GPU.']
        )
        links = []
    else:
        headline = reason_text or TRANSLATIONS['engine_fallback_banner_short'][langcode]
        steps = (
            [
                'GPU バックエンドの初期化に失敗したため CPU モードで動作しています（OCR 機能自体はそのまま利用できます）。',
                'GPU ドライバの更新や、配布フォルダを日本語（全角文字）を含まないパスへ展開しているかご確認ください。',
            ]
            if is_ja
            else [
                'Running in CPU fallback mode because GPU initialization failed (OCR continues to work normally).',
                'Ensure GPU drivers are up to date and the package is extracted to an ASCII-only folder.',
            ]
        )
        links = [
            (TRANSLATIONS['engine_diagnostics_open_releases'][langcode], releases_url),
        ]

    return {
        'severity': 'warning',
        'category': category,
        'headline': headline,
        'steps': steps,
        'links': links,
    }


def build_device_dropdown_options(availability: dict, is_ja: bool):
    """Build Device dropdown options showing verified vs missing-runtime vs unavailable."""
    device_labels = {
        'auto': '自動（GPU優先）' if is_ja else 'Auto (Prefer GPU)',
        'cuda': 'CUDA (NVIDIA)',
        'openvino': 'OpenVINO (Intel Arc / iGPU)',
        'webgpu': 'WebGPU (Radeon / Arc / NVIDIA)',
        'migraphx': 'MIGraphX (Linux Radeon)',
        'rocm': 'ROCm (Linux Radeon)',
        'dml': 'DirectML (Windows)',
        'cpu': 'CPU',
    }
    options = []
    for dev_key in DEVICE_CHOICES:
        label = device_labels.get(dev_key, dev_key)
        if dev_key not in ('auto', 'cpu'):
            info = availability.get(dev_key, {}) if isinstance(availability, dict) else {}
            ready = bool(info.get('available'))
            status = info.get('status')
            if ready:
                suffix = '利用可能' if is_ja else 'ready'
            elif status == 'missing_runtime':
                suffix = 'ドライバ/環境不足' if is_ja else 'missing driver/runtime'
            else:
                suffix = '未検出' if is_ja else 'unavailable'
            label = f'{label} [{suffix}]'
        options.append(ft.dropdown.Option(key=dev_key, text=label))
    return options


def build_gui_diagnostics_report(
    detector=None,
    recognizer=None,
    recognizer30=None,
    recognizer50=None,
    table_recognizer=None,
    table_error=None,
    log_path=None,
    init_error=None,
    *,
    refresh: bool = False,
) -> str:
    """Build full diagnostics text for the GUI Diagnostics dialog and log."""
    sections = []
    if init_error is not None:
        sections.append(f'Initialization error: {init_error}')

    model_entries = (
        ('DEIMv2 (detector)', detector),
        ('PARSeq-100', recognizer),
        ('PARSeq-30', recognizer30),
        ('PARSeq-50', recognizer50),
        ('TableCenterNet', table_recognizer),
    )
    active_lines = []
    for label, model in model_entries:
        if model is None:
            continue
        profile = getattr(model, 'backend_profile', None)
        if profile is None:
            active_lines.append(f'  - {label}: loaded (no profile)')
        else:
            fb = ' [fallback]' if getattr(profile, 'fallback_occurred', False) else ''
            active_lines.append(
                f'  - {label}: {profile.active_backend} ({profile.active_providers}){fb}'
            )
    if active_lines:
        sections.append('Active sessions:\n' + '\n'.join(active_lines))

    warning_info = collect_engine_warnings(
        detector,
        recognizer,
        recognizer30,
        recognizer50,
        table_recognizer,
        table_error=table_error,
    )
    grouped = warning_info.get('grouped_warnings') or []
    if grouped:
        blocks = []
        for grp in grouped:
            models_line = f"Affected models: {', '.join(grp['models'])}"
            body = grp.get('raw_warning') or grp.get('reason') or ''
            if len(grp['models']) > 1 and grp.get('raw_warning'):
                body = _strip_model_label_from_warning(grp['raw_warning'])
            elif not grp.get('raw_warning') and grp.get('hint'):
                body = f"{body}\n  Hint: {grp['hint']}"
            blocks.append(f"{models_line}\n{body}".strip())
        sections.append('Fallback warnings:\n' + '\n\n'.join(blocks))
    elif warning_info['warnings']:
        sections.append('Fallback warnings:\n' + '\n\n'.join(warning_info['warnings']))
    if warning_info['table_warning']:
        sections.append(f"TableCenterNet status:\n  {warning_info['table_warning']}")

    effective_log = log_path if log_path is not None else GUI_LOG_PATH
    if effective_log is not None:
        sections.append(f'Log file: {effective_log}')

    sections.append(format_device_diagnostics(refresh=refresh))
    return '\n\n'.join(sections)


def format_gui_engine_status(detector, recognizer, langcode='ja'):
    """Format active backend & batch size badge for the GUI status header."""
    prefix = TRANSLATIONS['engine_status_prefix'][langcode]
    profile = getattr(recognizer, 'backend_profile', None) or getattr(detector, 'backend_profile', None)
    if profile is None:
        return TRANSLATIONS['engine_status_loading'][langcode]
    bs = getattr(recognizer, 'batch_size', 1)
    bs_text = f', batch={bs}' if isinstance(bs, int) and bs > 1 else ''
    prec = getattr(recognizer, 'effective_precision', None)
    req_prec = getattr(recognizer, 'requested_precision', None)
    prec_fallback = bool(getattr(recognizer, 'precision_fallback_occurred', False))
    if isinstance(prec, str) and prec == 'fp16':
        prec_text = ', fp16 (近似)' if langcode == 'ja' else ', fp16 (approx)'
    elif prec_fallback or (isinstance(req_prec, str) and req_prec == 'fp16' and prec == 'fp32'):
        prec_text = ', fp16→fp32'
    else:
        prec_text = ''
    fb_occurred = any(
        bool(getattr(getattr(m, 'backend_profile', None), 'fallback_occurred', False))
        for m in (recognizer, detector)
    )
    fb_text = ' [fallback]' if fb_occurred else ''
    return f'{prefix}: {profile.display_name}{bs_text}{prec_text}{fb_text}'



class BusyTracker:
    """Track long-running OCR work so UI/engine changes can be deferred."""

    def __init__(self, on_idle=None):
        self._active = 0
        self._on_idle = on_idle

    def set_on_idle(self, callback):
        self._on_idle = callback

    def enter(self):
        self._active += 1

    def leave(self):
        was_active = self._active > 0
        self._active = max(0, self._active - 1)
        if was_active and self._active == 0 and self._on_idle is not None:
            self._on_idle()

    @property
    def active(self):
        return self._active > 0


def load_user_config(path, defaults):
    """Read userconf.yaml defensively, returning (config, warning_or_None)."""
    merged = dict(defaults)
    warning = None
    path = Path(path)
    try:
        if path.exists():
            with path.open(encoding='utf-8') as stream:
                loaded = yaml.safe_load(stream)
            if loaded is None:
                loaded = {}
            if not isinstance(loaded, dict):
                warning = f'user config is not a mapping: {type(loaded).__name__}'
            else:
                merged.update(loaded)
    except Exception as exc:
        warning = f'failed to load user config: {exc}'
    supported_locales = set(next(iter(TRANSLATIONS.values()), {'ja': '', 'en': ''}).keys())
    langcode = merged.get('langcode')
    if not isinstance(langcode, str) or langcode not in supported_locales:
        merged['langcode'] = 'ja'
    return merged, warning


def save_user_config(path, config_obj):
    """Write userconf.yaml atomically so a crash cannot truncate it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + '.tmp')
    with tmp_path.open('w', encoding='utf-8') as stream:
        yaml.dump(config_obj, stream, default_flow_style=False, allow_unicode=True)
    os.replace(tmp_path, path)


def parse_pdf_dpi(value, default=300.0, minimum=1.0):
    """Parse a user-supplied PDF DPI, clamping invalid values to a safe default."""
    try:
        dpi = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(dpi) or dpi <= 0:
        return float(default)
    return max(float(minimum), dpi)


def drop_changed_cli_overrides(cli_overrides, old_values, new_values):
    """Forget CLI overrides only for engine keys the dialog actually changed."""
    for key, new_value in new_values.items():
        if old_values.get(key) != new_value:
            cli_overrides.pop(key, None)


def main(page: ft.Page, args=None, download_models=False):
    if args is None:
        args = parse_gui_args(download_models=download_models)
    cli_overrides = {}
    if getattr(args, 'device', None) is not None:
        cli_overrides['device'] = args.device
    if getattr(args, 'device_id', None) is not None:
        cli_overrides['device_id'] = args.device_id
    if getattr(args, 'batch_size', None) is not None:
        cli_overrides['batch_size'] = args.batch_size
    if getattr(args, 'precision', None) is not None:
        cli_overrides['precision'] = args.precision

    page.title = 'NDLOCR-Lite-GUI v{}'.format(__version__)
    page.theme_mode = ft.ThemeMode.SYSTEM
    page.window.icon = str(ASSETS_DIR / 'icon.png')
    page.window.width = 1024
    page.window.height = 900
    page.window.min_width = 1024
    page.window.min_height = 900
    default_config = build_default_config()
    for feature in (DATA_POOL_FEATURE, LLM_FEATURE, TABLE_STRUCTURE_FEATURE):
        if feature is not None:
            default_config.update(feature.default_config())
    config_obj, config_warning = load_user_config(USER_CONFIG_PATH, default_config)
    resolve_gui_engine_args(args, config_obj, cli_overrides)
    log_path = configure_gui_logging()
    LOGGER.info(
        'Starting %s v%s | python=%s | platform=%s | app_data_dir=%s | log=%s',
        name,
        __version__,
        platform.python_version(),
        platform.platform(),
        APP_DATA_DIR,
        log_path,
    )
    if config_warning:
        print(f'[WARN] {config_warning}')
        LOGGER.warning('%s', config_warning)
    try:
        LOGGER.info('Startup device diagnostics:\n%s', format_device_diagnostics())
    except Exception as diag_exc:
        LOGGER.warning('Startup device diagnostics failed: %s', diag_exc)

    page.locale_configuration = ft.LocaleConfiguration(
        supported_locales=[
            ft.Locale('ja', 'JP'),
            ft.Locale('en', 'US'),
        ],
        current_locale=ft.Locale('ja', 'JP') if config_obj['langcode'] == 'ja' else ft.Locale('en', 'US'),
    )

    def save_config():
        save_user_config(USER_CONFIG_PATH, config_obj)

    def handle_locale_change(e):
        if busy_tracker.active or model_state['loading']:
            # Rebuilding the view while OCR tasks hold controls would orphan
            # dialogs and detached controls; ignore the switch while busy.
            e.control.selected_index = 0 if config_obj.get('langcode', 'ja') == 'ja' else 1
            page.update()
            return
        index = e.control.selected_index
        if index == 0:
            page.locale_configuration.current_locale = ft.Locale('ja', 'JP')
        elif index == 1:
            page.locale_configuration.current_locale = ft.Locale('en', 'US')
        config_obj['langcode'] = page.locale_configuration.current_locale.language_code
        save_config()
        page.update()
        renderui()

    origin_detector = None
    origin_recognizer = None
    origin_recognizer30 = None
    origin_recognizer50 = None
    origin_table_recognizer = None
    loaded_engine_state = None
    model_state = {
        'loading': False,
        'error': None,
        'table_error': None,
        'elapsed': None,
        'reload_scope': 'all',
        'banner_dismissed': False,
    }
    busy_tracker = BusyTracker()
    pending_engine_reload = {'pending': False}

    def renderui():
        page.on_keyboard_event = None
        page.clean()
        page.overlay.clear()
        page.services.clear()
        page.update()
        inputpathlist = []
        visualizepathlist = []
        outputtxtlist = []
        outputplaintextlist = []
        ocr_cancel_event=threading.Event()
        pdf_job_list = []
        datapool_batch_extension = None
        capture_tool = None
        main_interaction_sections = []
        model_loading_handlers = []

        def build_main_semantics_section(label, content, key):
            return ft.Semantics(
                content=content,
                label=label,
                container=True,
                header=True,
                heading_level=2,
                key=key,
            )

        def set_model_loading_lock(locked: bool):
            for section in main_interaction_sections:
                section.disabled = locked
            for control, event_name, handler in model_loading_handlers:
                setattr(control, event_name, None if locked else handler)

        def register_model_loading_handlers():
            seen = set()
            stack = list(main_interaction_sections)
            while stack:
                control = stack.pop()
                if id(control) in seen:
                    continue
                seen.add(id(control))
                for event_name in ('on_click', 'on_change', 'on_tap'):
                    handler = getattr(control, event_name, None)
                    if handler is not None:
                        model_loading_handlers.append((control, event_name, handler))
                children = getattr(control, 'controls', None)
                if children:
                    stack.extend(children)
                content = getattr(control, 'content', None)
                if isinstance(content, ft.Control):
                    stack.append(content)

        def create_pdf_func(outputpath: str, img: object, bboxlistobj: list, viztxtflag: bool, resolution: int = 300):
            from reportlab.pdfgen import canvas
            from reportlab.pdfbase import pdfmetrics
            from reportlab.pdfbase.cidfonts import UnicodeCIDFont
            from reportlab.lib.utils import ImageReader
            from reportlab.lib.colors import blue

            img_h, img_w = img.shape[:2]
            dpi = max(float(resolution), 1.0)
            page_w = img_w * 72.0 / dpi
            page_h = img_h * 72.0 / dpi

            pilimg_data = io.BytesIO()
            pilimg = Image.fromarray(img)
            if pilimg.mode in ("RGBA", "LA", "P"):
                pilimg = pilimg.convert("RGB")
            try:
                pilimg.save(pilimg_data, format='JPEG')
            except Exception:
                pilimg.save(pilimg_data, format='PNG')
            pilimg_data.seek(0)

            with _REPORTLAB_LOCK:
                pdfmetrics.registerFont(UnicodeCIDFont('HeiseiMin-W3', isVertical=True))
                pdfmetrics.registerFont(UnicodeCIDFont('HeiseiKakuGo-W5', isVertical=False))

                c = canvas.Canvas(outputpath, pagesize=(page_w, page_h))
                c.drawImage(
                    ImageReader(pilimg_data),
                    0,
                    0,
                    width=page_w,
                    height=page_h,
                    preserveAspectRatio=False,
                    mask=None,
                )

                if viztxtflag:
                    c.setFillAlpha(1)
                    c.setFillColor(blue)
                else:
                    c.setFillAlpha(0)

                for bboxobj in bboxlistobj:
                    text = bboxobj.get('text', '')
                    if not text:
                        continue
                    bbox = bboxobj['boundingBox']
                    xmin = bbox[0][0]
                    ymin = bbox[0][1]
                    line_w = bbox[2][0] - bbox[0][0]
                    line_h = bbox[1][1] - bbox[0][1]
                    line = {
                        'x': xmin,
                        'y': ymin,
                        'width': line_w,
                        'height': line_h,
                        'text': text,
                        'is_vertical': line_h > line_w,
                    }
                    ocr._draw_text_layer_line(
                        canvas_obj=c,
                        line=line,
                        img_width=img_w,
                        img_height=img_h,
                        page_width=page_w,
                        page_height=page_h,
                        visible=viztxtflag,
                    )
                c.save()

        def is_pdf_tmp_path(path: str) -> bool:
            try:
                tmp_dir = os.path.abspath(PDFTMPPATH)
                return os.path.abspath(path).startswith(tmp_dir + os.sep)
            except Exception:
                return False

        def render_pdf_preview(pdf_path: str, filestem: str):
            load_runtime_modules()
            os.makedirs(PDFTMPPATH, exist_ok=True)
            with Path(pdf_path).open('rb') as pdf_stream:
                doc = pypdfium2.PdfDocument(pdf_stream)
                try:
                    for ix in range(len(doc)):
                        rendered = doc.render(
                            pypdfium2.PdfBitmap.to_pil,
                            page_indices=[ix],
                            scale=100 / 72,
                        )
                        image = next(iter(rendered)).convert('RGB')
                        outputtmppath = os.path.join(PDFTMPPATH, '{}_{:05}.jpg'.format(filestem, ix))
                        inputpathlist.append(outputtmppath)
                        image.save(outputtmppath)
                        image.close()
                finally:
                    doc.close()

        def models_are_ready():
            return all((
                origin_detector,
                origin_recognizer,
                origin_recognizer30,
                origin_recognizer50,
            ))

        def parts_control(flag:bool,allow_ocr_cancel:bool=False):
            file_upload_btn.disabled = flag
            directory_upload_btn.disabled = flag
            directory_output_btn.disabled = flag
            chkbx_visualize.disabled = flag
            customize_btn.disabled = flag
            preview_prev_btn.disabled = flag or not outputtxtlist
            preview_next_btn.disabled = flag or not outputtxtlist
            has_input = bool(inputpathlist or pdf_job_list)
            has_output = bool(selected_output_path.value)
            ocr_btn.disabled = flag or not (models_are_ready() and has_input and has_output)
            crop_btn.disabled = flag or not (models_are_ready() and inputpathlist and has_output)
            cap_btn.disabled = flag or not models_are_ready()
            localebutton.disabled = flag or model_state['loading']
            stop_ocr_btn.disabled=not (flag and allow_ocr_cancel)
            if chkbx_table_structure is not None:
                chkbx_table_structure.disabled = flag or origin_table_recognizer is None
            if capture_tool is not None and capture_tool.cb_table_structure is not None:
                capture_tool.cb_table_structure.disabled = flag or origin_table_recognizer is None
            if datapool_batch_extension is not None:
                datapool_batch_extension.set_external_busy(flag)

        def start_engine_reload():
            if model_state['loading'] or busy_tracker.active:
                # Never swap sessions under a running OCR task; reload once it ends.
                pending_engine_reload['pending'] = True
                return
            resolve_gui_engine_args(args, config_obj, cli_overrides)
            target_state = extract_engine_state(args)
            active_baseline = loaded_engine_state if models_are_ready() else None
            reload_scope = compute_engine_reload_scope(active_baseline, target_state)
            if models_are_ready() and reload_scope == 'none':
                model_state['error'] = None
                refresh_engine_status_ui()
                page.update()
                return
            model_state['reload_scope'] = reload_scope
            model_state['error'] = None
            model_state['banner_dismissed'] = False
            model_state['loading'] = True
            set_model_loading_lock(True)
            parts_control(True)
            progressbar.value = None
            engine_status_text.value = TRANSLATIONS['engine_status_loading'][config_obj['langcode']]
            engine_warning_banner.visible = False
            if config_obj['langcode'] == 'ja':
                set_progress_message('OCRモデルを再読み込み中…')
            else:
                set_progress_message('Reloading OCR models…')
            page.update()
            page.run_task(initialize_models)

        def _flush_pending_engine_reload():
            if (
                pending_engine_reload['pending']
                and not busy_tracker.active
                and not model_state['loading']
            ):
                pending_engine_reload['pending'] = False
                start_engine_reload()

        busy_tracker.set_on_idle(_flush_pending_engine_reload)

        async def process_pdf_with_original_layer(pdf_path: str, outputpath: str, alljsonobjlist: list, pdf_outpath_list: list, allsum: int):
            pdf_path_obj = Path(pdf_path)
            output_stem = pdf_path_obj.stem
            render_dpi = parse_pdf_dpi(pdf_resolution_textfield.value)
            render_scale = render_dpi / 72.0
            pdf_started_at = time.perf_counter()

            with pdf_path_obj.open('rb') as pdf_stream:
                pdf_doc_meta = pypdfium2.PdfDocument(pdf_stream)
                try:
                    page_count = len(pdf_doc_meta)
                finally:
                    pdf_doc_meta.close()

            page_results = []
            all_json_contents = []
            page_infos = []
            all_text_pages = []
            all_markdown_pages = []
            html_pages = []
            all_page_xml = []
            tei_jsonobjlist = []

            separate_page_outputs = should_output_pdf_pages_separately(
                page_count,
                config_obj.get('pdf_page_output', True),
            )
            write_json = bool(chkbx_json.value)
            write_xml = bool(chkbx_xml.value)
            write_txt = bool(chkbx_txt.value)
            write_tei = bool(chkbx_tei.value)
            write_html = bool(chkbx_html.value)
            write_markdown = bool(chkbx_markdown.value)
            write_viz = bool(chkbx_visualize.value)
            write_pdf = bool(chkbx_pdf.value)
            pdf_viztxt = bool(chkbx_pdf_viztxt.value)
            table_enabled = bool(
                TABLE_STRUCTURE_FEATURE is not None
                and chkbx_table_structure is not None
                and chkbx_table_structure.value
                and origin_table_recognizer is not None
            )
            text_transform = get_character_inclusion_transform(config_obj)
            writer_workers = default_writer_workers()
            max_pending_writes = default_max_pending_writes(writer_workers)

            prefetcher = PdfPagePrefetcher(
                pdf_path=str(pdf_path_obj),
                page_count=page_count,
                render_scale=render_scale,
                detector=origin_detector,
                prefetch_size=3,
                cancel_event=ocr_cancel_event,
                pdfium_module=pypdfium2,
                np_module=np,
            )
            writer = BoundedAsyncWriter(
                max_workers=writer_workers,
                max_pending=max_pending_writes,
                thread_name_prefix='gui-pdf-writer',
            )

            processed_pages = 0
            completed = True
            suppress_writer_errors = False
            last_ui_update = time.monotonic()

            def _on_pdf_page_written(viz_result):
                if viz_result:
                    visualizepathlist.append(viz_result)

            try:
                while True:
                    if ocr_cancel_event.is_set():
                        completed = False
                        suppress_writer_errors = True
                        break

                    item = await prefetcher.next_page()
                    if item is None:
                        if ocr_cancel_event.is_set():
                            completed = False
                            suppress_writer_errors = True
                        break

                    if ocr_cancel_event.is_set():
                        completed = False
                        suppress_writer_errors = True
                        break

                    page_index, img, det_pre = item
                    page_name = f'{output_stem}_{page_index + 1:05}.png'
                    set_progress_message(f'{pdf_path_obj.name} page {page_index + 1}/{page_count}')
                    now_ui = time.monotonic()
                    if page_index == 0 or (now_ui - last_ui_update) >= 0.1:
                        progressmessage.update()
                        last_ui_update = now_ui

                    page_result = await asyncio.to_thread(
                        ocr._run_ocr_on_image_array,
                        detector=origin_detector,
                        recognizer30=origin_recognizer30,
                        recognizer50=origin_recognizer50,
                        recognizer100=origin_recognizer,
                        inputname=page_name,
                        img=img,
                        outputpath=outputpath,
                        save_viz=False,
                        text_transform=text_transform,
                        preprocessed_det=det_pre,
                        include_markdown=False,
                    )

                    page_tables = []
                    if table_enabled:
                        page_result, page_tables = await asyncio.to_thread(
                            TABLE_STRUCTURE_FEATURE.infer,
                            img,
                            page_result,
                            origin_table_recognizer,
                            args.table_crop_padding,
                        )
                        if page_tables:
                            for table_item in page_tables:
                                table_item['page_index'] = page_index
                            await asyncio.to_thread(
                                TABLE_STRUCTURE_FEATURE.write_html_files,
                                pdf_path_obj.name,
                                page_tables,
                                outputpath,
                                output_stem,
                            )

                    page_results.append(page_result)
                    if not separate_page_outputs:
                        all_json_contents.append(page_result['json_lines'])
                        page_infos.append({
                            'page_index': page_index,
                            'img_width': page_result['img_width'],
                            'img_height': page_result['img_height'],
                            'img_name': page_result['img_name'],
                        })
                        all_text_pages.append(page_result['text'])
                        all_page_xml.append(page_result['page_xml'])

                    page_markdown = build_gui_result_markdown(page_result, page_tables)
                    if write_markdown:
                        output_markdown = (
                            build_output_result_markdown(page_result, page_tables)
                            if page_tables
                            else page_markdown
                        )
                        all_markdown_pages.append(output_markdown)
                    if write_html and not separate_page_outputs:
                        html_pages.append((page_result, page_tables))

                    outputplaintextlist.append(page_result['text'])
                    outputtxtlist.append(page_markdown)

                    page_json_obj = build_pdf_page_json_obj(
                        page_result,
                        source_path=str(pdf_path_obj),
                        include_xml=True,
                        source_page_index=page_index,
                        source_dpi=render_dpi,
                    )
                    alljsonobjlist.append(page_json_obj)
                    if not separate_page_outputs:
                        tei_jsonobjlist.append(page_json_obj)

                    await writer.submit(
                        _write_pdf_page_task,
                        output_dir=outputpath,
                        output_stem=output_stem,
                        page_index=page_index,
                        page_name=page_name,
                        img=img,
                        page_result=page_result,
                        page_json_obj=page_json_obj,
                        source_path=str(pdf_path_obj),
                        source_dpi=render_dpi,
                        separate_page_outputs=separate_page_outputs,
                        write_json=write_json,
                        write_xml=write_xml,
                        write_txt=write_txt,
                        write_tei=write_tei,
                        write_html=write_html,
                        write_viz=write_viz,
                        page_tables=page_tables,
                        detector=origin_detector,
                        categories_mapping=categories_org_name_index,
                        convert_tei_func=convert_tei,
                        build_html_fn=build_output_result_html,
                        on_complete=_on_pdf_page_written,
                    )

                    processed_pages += 1
                    progressbar.value += 1 / max(allsum, 1)
                    now_ui = time.monotonic()
                    if (
                        page_index == 0
                        or page_index == page_count - 1
                        or (now_ui - last_ui_update) >= 0.1
                    ):
                        page.update()
                        last_ui_update = now_ui

                if completed:
                    writer.flush_completed()
            except BaseException:
                suppress_writer_errors = True
                raise
            finally:
                prefetcher.close()
                await writer.close(suppress_errors=suppress_writer_errors)

            if not completed:
                return processed_pages, False, tei_jsonobjlist

            if write_xml and not separate_page_outputs:
                with open(os.path.join(outputpath, output_stem + '.xml'), 'w', encoding='utf-8') as wf:
                    wf.write('<OCRDATASET>\n')
                    wf.write('\n'.join(all_page_xml))
                    wf.write('\n</OCRDATASET>')

            if write_txt and not separate_page_outputs:
                with open(os.path.join(outputpath, output_stem + '.txt'), 'w', encoding='utf-8') as wf:
                    wf.write('\n\n'.join(all_text_pages))

            if write_markdown:
                with open(os.path.join(outputpath, output_stem + '.md'), 'w', encoding='utf-8') as wf:
                    wf.write('\n\n---\n\n'.join(all_markdown_pages))
            if write_html and not separate_page_outputs:
                Path(outputpath, output_stem + '.html').write_text(
                    build_output_result_html(pdf_path_obj.name, html_pages), encoding='utf-8',
                )

            pdf_json_obj = {
                'contents': all_json_contents,
                'pdfinfo': {
                    'pdf_path': str(pdf_path_obj),
                    'pdf_name': pdf_path_obj.name,
                    'page_count': len(page_results),
                    'render_dpi': render_dpi,
                },
                'pages': page_infos,
            }

            if write_json and not separate_page_outputs:
                with open(os.path.join(outputpath, output_stem + '.json'), 'w', encoding='utf-8') as wf:
                    wf.write(json.dumps(pdf_json_obj, ensure_ascii=False, indent=2))

            if write_pdf and page_results:
                output_pdf = os.path.join(outputpath, output_stem + '_text.pdf')
                await asyncio.to_thread(
                    ocr.embed_text_layer_pdf,
                    input_pdf=str(pdf_path_obj),
                    output_pdf=output_pdf,
                    page_results=page_results,
                    visible_text=pdf_viztxt,
                )
                pdf_outpath_list.append(output_pdf)

            pdf_elapsed = max(time.perf_counter() - pdf_started_at, 1e-6)
            LOGGER.info(
                'Completed PDF OCR: file=%s | pages=%d | elapsed=%.2fs (%.2f p/s) | dpi=%.0f | viz=%s | writers=%d',
                pdf_path_obj.name,
                processed_pages,
                pdf_elapsed,
                processed_pages / pdf_elapsed,
                render_dpi,
                write_viz,
                writer_workers,
            )

            return page_count, True, tei_jsonobjlist

        def request_ocr_cancel(e):
            if ocr_cancel_event.is_set():
                return
            ocr_cancel_event.set()
            stop_ocr_btn.disabled=True
            set_progress_message(TRANSLATIONS["main_ocr_cancel_requested"][config_obj["langcode"]])
            page.update()

        async def ocr_button_result(e):
            ocr_cancel_event.clear()
            progressbar.value = 0
            outputpath = selected_output_path.value
            nonlocal inputpathlist, outputtxtlist, visualizepathlist, preview_index, args
            nonlocal origin_recognizer, origin_recognizer30, origin_recognizer50
            nonlocal origin_detector

            if not outputpath:
                set_progress_message('Output directory is not selected.')
                progressmessage.update()
                return

            preview_index = 0
            parts_control(True,allow_ocr_cancel=True)
            busy_tracker.enter()
            page.update()
            set_progress_message('Start')
            progressmessage.update()

            try:
                allstart = time.time()
                progressbar.value = 0
                progressbar.update()
                outputtxtlist.clear()
                outputplaintextlist.clear()
                visualizepathlist.clear()
                alljsonobjlist = []
                tei_jsonobjlist = []
                pdf_outpath_list = []
                completed_count=0
                cancelled=False
                standalone_inputpathlist = [p for p in inputpathlist if not is_pdf_tmp_path(p)]

                allsum = len(standalone_inputpathlist)
                for pdf_path in pdf_job_list:
                    try:
                        with Path(pdf_path).open('rb') as pdf_stream:
                            pdf_doc_tmp = pypdfium2.PdfDocument(pdf_stream)
                            try:
                                allsum += len(pdf_doc_tmp)
                            finally:
                                pdf_doc_tmp.close()
                    except Exception:
                        allsum += 1

                if allsum == 0:
                    set_progress_message('Images are not found.')
                    progressmessage.update()
                    return

                for pdf_path in pdf_job_list:
                    processed_pages, pdf_completed, pdf_tei_jsonobjlist = await process_pdf_with_original_layer(
                        pdf_path,
                        outputpath,
                        alljsonobjlist,
                        pdf_outpath_list,
                        allsum,
                    )
                    tei_jsonobjlist.extend(pdf_tei_jsonobjlist)
                    completed_count += processed_pages
                    if not pdf_completed:
                        cancelled = True
                        break

                if standalone_inputpathlist and not cancelled:
                    write_xml = bool(chkbx_xml.value)
                    write_viz = bool(chkbx_visualize.value)
                    write_json = bool(chkbx_json.value)
                    write_txt = bool(chkbx_txt.value)
                    write_markdown = bool(chkbx_markdown.value)
                    write_html = bool(chkbx_html.value)
                    write_pdf = bool(chkbx_pdf.value)
                    pdf_viztxt = bool(chkbx_pdf_viztxt.value)
                    pdf_resolution = int(parse_pdf_dpi(pdf_resolution_textfield.value))
                    table_enabled = bool(
                        TABLE_STRUCTURE_FEATURE is not None
                        and chkbx_table_structure is not None
                        and chkbx_table_structure.value
                        and origin_table_recognizer is not None
                    )
                    text_transform = get_character_inclusion_transform(config_obj)
                    img_writer_workers = default_writer_workers()
                    img_max_pending = default_max_pending_writes(img_writer_workers)
                    img_prefetcher = ImagePrefetcher(
                        paths=standalone_inputpathlist,
                        load_image_fn=load_rgb_image_array,
                        detector=origin_detector,
                        prefetch_size=3,
                        cancel_event=ocr_cancel_event,
                    )
                    img_writer = BoundedAsyncWriter(
                        max_workers=img_writer_workers,
                        max_pending=img_max_pending,
                        thread_name_prefix='gui-img-writer',
                    )
                    img_loop_failed = False
                    last_img_ui_update = time.monotonic()

                    def _on_image_page_written(result):
                        viz_res, pdf_res = result
                        if viz_res:
                            visualizepathlist.append(viz_res)
                        if pdf_res:
                            pdf_outpath_list.append(pdf_res)

                    try:
                        while True:
                            if ocr_cancel_event.is_set():
                                cancelled = True
                                break
                            prefetched = await img_prefetcher.next_item()
                            if prefetched is None:
                                if ocr_cancel_event.is_set():
                                    cancelled = True
                                break
                            if ocr_cancel_event.is_set():
                                cancelled = True
                                break

                            idx, inputpath, img, det_pre = prefetched
                            set_progress_message(inputpath)
                            now_ui = time.monotonic()
                            if idx == 0 or (now_ui - last_img_ui_update) >= 0.1:
                                progressmessage.update()
                                last_img_ui_update = now_ui

                            start = time.time()
                            img_h, img_w = img.shape[:2]
                            imgname = os.path.basename(inputpath)

                            page_result = await asyncio.to_thread(
                                ocr._run_ocr_on_image_array,
                                detector=origin_detector,
                                recognizer30=origin_recognizer30,
                                recognizer50=origin_recognizer50,
                                recognizer100=origin_recognizer,
                                inputname=imgname,
                                img=img,
                                outputpath=outputpath,
                                save_viz=False,
                                text_transform=text_transform,
                                preprocessed_det=det_pre,
                                include_markdown=False,
                            )

                            page_tables = []
                            if table_enabled:
                                page_result, page_tables = await asyncio.to_thread(
                                    TABLE_STRUCTURE_FEATURE.infer,
                                    img,
                                    page_result,
                                    origin_table_recognizer,
                                    args.table_crop_padding,
                                )

                            allxmlstr = '<OCRDATASET>\n' + page_result['page_xml'] + '\n</OCRDATASET>'
                            alltextlist = [page_result['text']]
                            resjsonarray = page_result['json_lines']
                            output_stem = os.path.splitext(os.path.basename(inputpath))[0]
                            if page_tables:
                                await asyncio.to_thread(
                                    TABLE_STRUCTURE_FEATURE.write_html_files,
                                    imgname,
                                    page_tables,
                                    outputpath,
                                    output_stem,
                                )
                            page_markdown = build_gui_result_markdown(page_result, page_tables)
                            output_markdown = (
                                build_output_result_markdown(page_result, page_tables)
                                if (write_markdown and page_tables)
                                else (page_markdown if write_markdown else '')
                            )
                            outputplaintextlist.append(page_result['text'])
                            outputtxtlist.append(page_markdown)

                            alljsonobj = {
                                'contents': [resjsonarray],
                                'imginfo': {
                                    'img_width': img_w,
                                    'img_height': img_h,
                                    'img_path': inputpath,
                                    'img_name': os.path.basename(inputpath),
                                },
                                'xml': allxmlstr,
                            }
                            alljsonobjlist.append(alljsonobj)
                            tei_jsonobjlist.append(alljsonobj)

                            await img_writer.submit(
                                _write_image_page_task,
                                inputpath=inputpath,
                                imgname=imgname,
                                outputpath=outputpath,
                                output_stem=output_stem,
                                img=img,
                                allxmlstr=allxmlstr,
                                alltextlist=alltextlist,
                                resjsonarray=resjsonarray,
                                alljsonobj=alljsonobj,
                                page_result=page_result,
                                page_tables=page_tables,
                                output_markdown=output_markdown,
                                write_xml=write_xml,
                                write_viz=write_viz,
                                write_json=write_json,
                                write_txt=write_txt,
                                write_markdown=write_markdown,
                                write_html=write_html,
                                write_pdf=write_pdf,
                                pdf_viztxt=pdf_viztxt,
                                pdf_resolution=pdf_resolution,
                                detector=origin_detector,
                                categories_mapping=categories_org_name_index,
                                build_html_fn=build_output_result_html,
                                create_pdf_fn=create_pdf_func,
                                on_complete=_on_image_page_written,
                            )

                            print('Total calculation time (Detection + Recognition):', time.time() - start)
                            completed_count += 1
                            progressbar.value += 1 / max(allsum, 1)
                            preview_prev_btn.disabled = False
                            preview_next_btn.disabled = False
                            if outputtxtlist:
                                preview_plain_text.value = outputplaintextlist[preview_index] if preview_index < len(outputplaintextlist) else ''
                                preview_text.value = outputtxtlist[preview_index]
                            if write_viz:
                                img_writer.flush_completed()
                            if len(visualizepathlist) > preview_index:
                                preview_image.src = visualizepathlist[preview_index]
                            elif inputpathlist:
                                preview_image.src = inputpathlist[min(preview_index, len(inputpathlist) - 1)]
                            if inputpathlist:
                                current_visualizeimgname.value = os.path.basename(inputpathlist[min(preview_index, len(inputpathlist) - 1)])
                                preview_name_semantics.value = current_visualizeimgname.value
                            now_ui = time.monotonic()
                            if (
                                idx == 0
                                or idx == len(standalone_inputpathlist) - 1
                                or (now_ui - last_img_ui_update) >= 0.1
                            ):
                                preview_image.update()
                                page.update()
                                last_img_ui_update = now_ui
                    except BaseException:
                        img_loop_failed = True
                        raise
                    finally:
                        img_prefetcher.close()
                        await img_writer.close(suppress_errors=(cancelled or img_loop_failed))

                batch_elapsed = max(time.time() - allstart, 1e-6)
                LOGGER.info(
                    'OCR batch finished | completed=%d/%d | cancelled=%s | elapsed=%.2fs (%.2f items/s)',
                    completed_count,
                    allsum,
                    cancelled,
                    batch_elapsed,
                    completed_count / batch_elapsed,
                )
                if cancelled:
                    set_progress_message(
                        TRANSLATIONS["main_ocr_cancelled"][config_obj["langcode"]].format(
                            completed=completed_count,total=allsum,elapsed=time.time()-allstart
                        )
                    )
                elif config_obj['langcode'] == 'ja':
                    set_progress_message('{} 件OCR完了 / 所要時間 {:.2f} 秒'.format(allsum, time.time() - allstart))
                else:
                    set_progress_message('{} items completed / Total time {:.2f} sec'.format(allsum, time.time() - allstart))
                progressmessage.update()

                if chkbx_tei.value and len(tei_jsonobjlist)>0:
                    tei_base_path = inputpathlist[0] if inputpathlist else tei_jsonobjlist[0]['imginfo']['img_name']
                    with open(os.path.join(outputpath, os.path.splitext(os.path.basename(tei_base_path))[0] + '_tei.xml'), 'wb') as wf:
                        allxmlstrtei = convert_tei(tei_jsonobjlist)
                        wf.write(allxmlstrtei)

                if chkbx_text_daisy.value and alljsonobjlist:
                    first_name = alljsonobjlist[0].get('imginfo', {}).get('img_name', 'ocr')
                    book_stem = os.path.splitext(os.path.basename(first_name))[0] or 'ocr'
                    text_daisy_result = TextDaisyExporter(
                        title=book_stem,
                        language=config_obj['langcode'],
                        producer='NDLOCR-Lite',
                    ).export(
                        alljsonobjlist,
                        outputpath,
                        folder_name=book_stem + '_textdaisy',
                        make_zip=True,
                    )
                    daisy_path = text_daisy_result.zip_path or text_daisy_result.directory
                    set_progress_message(
                        f'{progressmessage.value} / Text DAISY: {os.path.basename(daisy_path)}'
                    )

                if chkbx_pdf.value and chkbx_pdf_merge.value and len(pdf_outpath_list) > 1:
                    from pypdf import PdfReader, PdfWriter

                    writer = PdfWriter()
                    for p in pdf_outpath_list:
                        reader = PdfReader(p)
                        for page_obj in reader.pages:
                            writer.add_page(page_obj)

                    merged_pdf_path = os.path.join(
                        outputpath,
                        os.path.splitext(os.path.basename(inputpathlist[0]))[0] + '_merged.pdf',
                    )
                    with open(merged_pdf_path, 'wb') as wf:
                        writer.write(wf)

                    for p in pdf_outpath_list:
                        try:
                            os.remove(p)
                        except OSError:
                            pass

                if outputtxtlist:
                    preview_plain_text.value = outputplaintextlist[0] if outputplaintextlist else ''
                    preview_text.value = outputtxtlist[0]
                if len(visualizepathlist) > 0:
                    preview_image.src = visualizepathlist[0]
                    current_visualizeimgname.value = os.path.basename(visualizepathlist[0])
                    preview_name_semantics.value = current_visualizeimgname.value
                elif inputpathlist:
                    preview_image.src = inputpathlist[0]
                    current_visualizeimgname.value = os.path.basename(inputpathlist[0])
                    preview_name_semantics.value = current_visualizeimgname.value
                preview_image.update()
                preview_plain_text.update()
                preview_text.update()

            except Exception as e:
                print(e)
                set_progress_message(str(e))
                progressmessage.update()
            finally:
                ocr_cancel_event.clear()
                parts_control(False)
                page.update()
                busy_tracker.leave()

        async def pick_files_result(e):
            files = await file_picker.pick_files(allow_multiple=False)
            if files and files[0].path:
                selected_file_path = files[0].path
                selected_input_path.value = selected_file_path
                input_value_semantics.value = selected_file_path
                nonlocal inputpathlist, outputtxtlist, pdf_job_list
                inputpathlist.clear()
                outputtxtlist.clear()
                outputplaintextlist.clear()
                pdf_job_list.clear()
                ext = selected_file_path.split('.')[-1].lower()
                if ext == 'pdf':
                    pdf_job_list.append(selected_file_path)
                    filestem = os.path.basename(selected_file_path)[:-4]
                    if config_obj['langcode'] == 'ja':
                        set_progress_message('pdfファイルの前処理中…… {} '.format(selected_file_path))
                    else:
                        set_progress_message('preprocessing pdf…… {} '.format(selected_file_path))
                    parts_control(True)
                    page.update()
                    try:
                        for p in glob.glob(os.path.join(PDFTMPPATH, '*.jpg')):
                            if os.path.isfile(p):
                                os.remove(p)
                        os.makedirs(PDFTMPPATH, exist_ok=True)
                        await asyncio.to_thread(render_pdf_preview, selected_file_path, filestem)
                        if config_obj['langcode'] == 'ja':
                            set_progress_message('pdfファイルの前処理完了')
                        else:
                            set_progress_message('Preprocessing of pdf complete')
                    except Exception as err:
                        print(f'PDF preprocessing failed: {err}')
                        pdf_job_list.remove(selected_file_path) if selected_file_path in pdf_job_list else None
                        set_progress_message(f'PDF前処理エラー: {err}')
                    finally:
                        parts_control(False)
                        page.update()
                else:
                    inputpathlist.append(selected_file_path)
                selector.set_image(inputpathlist)
                if selected_output_path.value is not None:
                    parts_control(False)
            selected_input_path.update()
            page.update()

        async def pick_directory_result(e):
            selected_directory = await file_picker.get_directory_path()
            if selected_directory:
                selected_input_path.value = selected_directory
                input_value_semantics.value = selected_directory
                nonlocal inputpathlist, outputtxtlist, pdf_job_list
                inputpathlist.clear()
                outputtxtlist.clear()
                outputplaintextlist.clear()
                pdf_job_list.clear()

                for p in glob.glob(os.path.join(PDFTMPPATH, '*.jpg')):
                    if os.path.isfile(p):
                        try:
                            os.remove(p)
                        except Exception:
                            pass
                os.makedirs(PDFTMPPATH, exist_ok=True)

                parts_control(True)
                crop_btn.disabled = True
                ocr_btn.disabled = True
                page.update()

                all_files_to_process = []
                pdf_filename_counter = Counter()

                for root, dirs, files in os.walk(selected_directory):
                    dirs.sort()
                    files.sort()
                    for filename in files:
                        full_path = os.path.join(root, filename)
                        ext = filename.split('.')[-1].lower()

                        if ext in ['jpg', 'png', 'tiff', 'jp2', 'tif', 'jpeg', 'bmp', 'webp']:
                            all_files_to_process.append((full_path, 'image'))
                        elif ext == 'pdf':
                            all_files_to_process.append((full_path, 'pdf'))
                            pdf_filename_counter[filename] += 1

                all_files_to_process.sort(key=lambda x: x[0])

                for inputpath, filetype in all_files_to_process:
                    if filetype == 'image':
                        inputpathlist.append(inputpath)
                    elif filetype == 'pdf':
                        pdf_job_list.append(inputpath)
                        filename = os.path.basename(inputpath)
                        if pdf_filename_counter[filename] > 1:
                            rel_path = os.path.relpath(inputpath, start=selected_directory)
                            filestem = os.path.splitext(rel_path)[0].replace(os.sep, '-')
                        else:
                            filestem = os.path.splitext(filename)[0]

                        if config_obj['langcode'] == 'ja':
                            set_progress_message('pdfファイルの前処理中…… {} '.format(inputpath))
                        else:
                            set_progress_message('preprocessing pdf…… {} '.format(inputpath))
                        page.update()

                        try:
                            await asyncio.to_thread(render_pdf_preview, inputpath, filestem)
                        except Exception as err:
                            print(f'Error processing {inputpath}: {err}')

                if config_obj['langcode'] == 'ja':
                    set_progress_message('処理完了')
                else:
                    set_progress_message('Processing complete')

                selector.set_image(inputpathlist)
                if not inputpathlist and not pdf_job_list:
                    set_progress_message('処理可能なファイルが見つかりませんでした / No processable files found')
                # Restore the UI even for empty directories or failed PDF previews.
                parts_control(False)

            selected_input_path.update()
            page.update()

        async def pick_output_result(e):
            nonlocal inputpathlist
            selected_directory = await file_picker.get_directory_path()
            if selected_directory:
                selected_output_path.value = selected_directory
                output_value_semantics.value = selected_directory
                selected_output_path.update()
                config_obj['selected_output_path'] = selected_directory
                save_config()
                selector.set_outputdir(selected_directory)
                capture_tool.outputdirpath = selected_directory
                if len(inputpathlist) > 0:
                    parts_control(False)
            page.update()

        preview_index = 0

        def next_image(e):
            nonlocal inputpathlist, outputtxtlist, preview_index
            if not outputtxtlist:
                return
            if preview_index < min(len(inputpathlist) - 1, len(outputtxtlist) - 1):
                preview_index += 1
            else:
                preview_index = 0

            if len(visualizepathlist) > 0:
                preview_image.src = visualizepathlist[min(preview_index, len(visualizepathlist) - 1)]
                current_visualizeimgname.value = os.path.basename(preview_image.src)
                preview_name_semantics.value = current_visualizeimgname.value
            elif 0 <= preview_index < len(inputpathlist):
                preview_image.src = inputpathlist[preview_index]
                current_visualizeimgname.value = os.path.basename(inputpathlist[preview_index])
                preview_name_semantics.value = current_visualizeimgname.value
            if 0 <= preview_index < len(outputtxtlist):
                preview_plain_text.value = outputplaintextlist[preview_index] if preview_index < len(outputplaintextlist) else ''
                preview_text.value = outputtxtlist[preview_index]
            preview_image.update()
            preview_plain_text.update()
            preview_text.update()
            page.update()

        def prev_image(e):
            nonlocal inputpathlist, outputtxtlist, preview_index
            if not outputtxtlist:
                return
            if preview_index > 0:
                preview_index -= 1
            else:
                preview_index = min(len(inputpathlist) - 1, len(outputtxtlist) - 1)

            if len(visualizepathlist) > 0:
                preview_image.src = visualizepathlist[min(preview_index, len(visualizepathlist) - 1)]
                current_visualizeimgname.value = os.path.basename(preview_image.src)
                preview_name_semantics.value = current_visualizeimgname.value
            elif 0 <= preview_index < len(inputpathlist):
                preview_image.src = inputpathlist[preview_index]
                current_visualizeimgname.value = os.path.basename(inputpathlist[preview_index])
                preview_name_semantics.value = current_visualizeimgname.value
            if 0 <= preview_index < len(outputtxtlist):
                preview_plain_text.value = outputplaintextlist[preview_index] if preview_index < len(outputplaintextlist) else ''
                preview_text.value = outputtxtlist[preview_index]
            preview_image.update()
            preview_plain_text.update()
            preview_text.update()
            page.update()

        def change_preview_view_mode(e=None):
            use_markdown = bool(preview_view_switch.value)
            config_obj['result_markdown_display'] = use_markdown
            save_config()
            preview_text.visible = use_markdown
            preview_plain_text.visible = not use_markdown
            # Keep the crop/capture result viewers on the same mode when available.
            for viewer in (selector, capture_tool):
                viewer.result_view_switch.value = use_markdown
                viewer.resulttext.visible = use_markdown
                viewer.result_plain_text.visible = not use_markdown
            page.update()

        def handle_customize_dlg_modal_close(e):
            nonlocal origin_detector, origin_recognizer
            nonlocal origin_recognizer30, origin_recognizer50
            nonlocal origin_table_recognizer

            new_device = normalize_device_name(dropdown_device.value or 'auto')
            try:
                new_batch_size = max(0, int(dropdown_batch_size.value or 0))
            except (TypeError, ValueError):
                new_batch_size = 0
            try:
                new_device_id = max(0, int(device_id_textfield.value or 0))
            except (TypeError, ValueError):
                new_device_id = 0
            raw_precision = dropdown_precision.value if isinstance(dropdown_precision.value, str) else 'fp32'
            new_precision = normalize_precision_name(raw_precision)
            if new_precision not in PRECISION_CHOICES:
                new_precision = 'fp32'

            target_state = {
                'device': new_device,
                'batch_size': new_batch_size,
                'device_id': new_device_id,
                'precision': new_precision,
            }
            baseline_state = (
                loaded_engine_state
                if loaded_engine_state is not None
                else extract_engine_state(args)
            )
            engine_changed = (
                not models_are_ready()
                or loaded_engine_state is None
                or target_state != loaded_engine_state
            )

            updated_config = {
                'json': chkbx_json.value,
                'txt': chkbx_txt.value,
                'markdown': chkbx_markdown.value,
                'html': chkbx_html.value,
                'xml': chkbx_xml.value,
                'tei': chkbx_tei.value,
                'pdf': chkbx_pdf.value,
                'pdf_viztxt': chkbx_pdf_viztxt.value,
                'pdf_merge': chkbx_pdf_merge.value,
                'pdf_page_output': chkbx_pdf_page_output.value,
                'pdf_resolution': int(parse_pdf_dpi(pdf_resolution_textfield.value)),
                'text_daisy': chkbx_text_daisy.value,
                'character_inclusion': chkbx_character_inclusion.value,
                'device': new_device,
                'batch_size': new_batch_size,
                'device_id': new_device_id,
                'precision': new_precision,
            }
            if chkbx_table_structure is not None and (
                TABLE_STRUCTURE_FEATURE is None or TABLE_STRUCTURE_FEATURE.is_supported()
            ):
                updated_config['table_structure'] = chkbx_table_structure.value
            config_obj.update(updated_config)
            save_config()
            if customize_dlg_modal.open:
                page.pop_dialog()

            if engine_changed:
                drop_changed_cli_overrides(
                    cli_overrides,
                    baseline_state,
                    target_state,
                )
                start_engine_reload()
            else:
                resolve_gui_engine_args(args, config_obj, cli_overrides)

        def change_pdfstatus(e):
            chkbx_pdf_viztxt.disabled = not chkbx_pdf.value
            chkbx_pdf_viztxt.update()
            chkbx_pdf_merge.disabled = not chkbx_pdf.value
            chkbx_pdf_merge.update()
            pdf_options_panel.opacity = 1.0 if chkbx_pdf.value else 0.55
            pdf_options_panel.update()

        def build_settings_section(title, description, icon, controls, key):
            return ft.Container(
                key=key,
                padding=12,
                bgcolor=ft.Colors.SURFACE_CONTAINER_LOW,
                border_radius=10,
                content=ft.Column(
                    controls=[
                        ft.Row(
                            controls=[
                                ft.Icon(icon, color=ft.Colors.PRIMARY, size=22),
                                ft.Column(
                                    controls=[
                                        ft.Text(
                                            title,
                                            theme_style=ft.TextThemeStyle.TITLE_MEDIUM,
                                            weight=ft.FontWeight.W_600,
                                        ),
                                        ft.Text(
                                            description,
                                            theme_style=ft.TextThemeStyle.BODY_SMALL,
                                            color=ft.Colors.ON_SURFACE_VARIANT,
                                        ),
                                    ],
                                    spacing=1,
                                    tight=True,
                                    expand=True,
                                ),
                            ],
                            spacing=8,
                            vertical_alignment=ft.CrossAxisAlignment.CENTER,
                        ),
                        ft.Divider(height=1),
                        *controls,
                    ],
                    spacing=6,
                    tight=True,
                ),
            )

        preview_image = ft.Image(
            src=DEFAULT_IMAGE_SRC,
            width=400,
            height=300,
            gapless_playback=True,
            semantics_label=TRANSLATIONS['a11y_main_preview_image'][config_obj['langcode']],
        )
        preview_text = ft.Markdown(
            value='',
            selectable=True,
            extension_set=ft.MarkdownExtensionSet.GITHUB_WEB,
            on_tap_link=open_preview_link,
            md_style_sheet=build_preview_markdown_style(),
            soft_line_break=True,
            visible=bool(config_obj.get('result_markdown_display', False)),
        )
        preview_plain_text = ft.Text(
            value='',
            selectable=True,
            visible=not bool(config_obj.get('result_markdown_display', False))
        )
        preview_view_switch = ft.Switch(
            label='整形表示（Markdown）',
            value=bool(config_obj.get('result_markdown_display', False)),
            on_change=change_preview_view_mode,
        )

        file_picker = ft.FilePicker()
        clipboard = ft.Clipboard()
        page.services.extend([file_picker, clipboard])
        progressbar = ft.ProgressBar(
            width=200,
            value=0,
            semantics_label=TRANSLATIONS['a11y_main_progress'][config_obj['langcode']],
        )
        selected_input_path = ft.Text(selectable=True)
        selected_output_path = ft.Text(config_obj['selected_output_path'], selectable=True)
        current_visualizeimgname = ft.Text(selectable=True)
        progressmessage = ft.Text(width=400, selectable=True)
        progress_message_semantics = None

        def set_progress_message(message):
            progressmessage.value = message
            if progress_message_semantics is not None:
                progress_message_semantics.label = message

        def change_visualizestatus(e):
            config_obj['visualize'] = bool(chkbx_visualize.value)
            save_config()

        chkbx_visualize = ft.Checkbox(
            label=TRANSLATIONS['main_visualize_label'][config_obj['langcode']],
            value=bool(config_obj.get('visualize', True)),
            on_change=change_visualizestatus,
            key='visualize',
        )
        chkbx_json = ft.Checkbox(label='JSON', value=config_obj['json'], key='output_json')
        chkbx_txt = ft.Checkbox(label='TXT', value=config_obj['txt'], key='output_txt')
        chkbx_markdown = ft.Checkbox(label='Markdown（.md）', value=config_obj.get('markdown', False), key='output_markdown')
        chkbx_html = ft.Checkbox(label='HTML（.html）', value=config_obj.get('html', False), key='output_html')
        chkbx_xml = ft.Checkbox(label='XML', value=config_obj['xml'], key='output_xml')
        chkbx_tei = ft.Checkbox(label='TEI', value=config_obj['tei'], key='output_tei')
        chkbx_pdf = ft.Checkbox(label='透明テキスト付PDF', value=config_obj['pdf'], on_change=change_pdfstatus, key='output_pdf')
        chkbx_pdf_viztxt = ft.Checkbox(label='PDFに青色で文字を重ねる', value=config_obj['pdf_viztxt'], disabled=not chkbx_pdf.value, key='output_pdf_visible_text')
        chkbx_pdf_merge = ft.Checkbox(label='出力ファイルを1つのpdfにまとめる', value=config_obj['pdf_merge'], disabled=not chkbx_pdf.value, key='output_pdf_merge')
        chkbx_pdf_page_output = ft.Checkbox(label='複数ページPDFを入力とする時はTXT/JSON/XML/TEI/HTMLをページ毎に出力する', value=config_obj.get('pdf_page_output', True), key='output_pdf_page_output')
        pdf_resolution_textfield = ft.TextField(
            label=TRANSLATIONS['customize_pdf_resolution_label'][config_obj['langcode']],
            value=str(config_obj['pdf_resolution']),
            suffix='dpi',
            keyboard_type=ft.KeyboardType.NUMBER,
            width=210,
            dense=True,
            key='output_pdf_resolution',
        )
        chkbx_text_daisy = ft.Checkbox(
            label='テキストDAISY（DAISY 3）',
            value=config_obj['text_daisy'],
            key='output_text_daisy',
        )
        chkbx_character_inclusion = ft.Checkbox(
            label=TRANSLATIONS['main_character_inclusion_label'][config_obj['langcode']],
            value=config_obj.get('character_inclusion', False),
            key='output_character_inclusion',
        )
        chkbx_table_structure = (
            TABLE_STRUCTURE_FEATURE.create_output_control(config_obj)
            if TABLE_STRUCTURE_FEATURE is not None
            else None
        )
        table_structure_notice_text = ft.Text(
            '',
            size=11,
            color=ft.Colors.AMBER_800,
            visible=False,
            selectable=True,
            key='table_structure_notice_text',
        )
        backend_availability = detect_available_backends()
        is_ja = config_obj['langcode'] == 'ja'
        device_options = build_device_dropdown_options(backend_availability, is_ja)

        current_device_val = normalize_device_name(args.device)
        if current_device_val not in DEVICE_CHOICES:
            current_device_val = 'auto'
        dropdown_device = ft.Dropdown(
            label=TRANSLATIONS['customize_device_label'][config_obj['langcode']],
            value=current_device_val,
            options=device_options,
            width=360,
            dense=True,
            key='settings_device',
        )
        batch_options = [
            ft.dropdown.Option(key='0', text='自動（推奨）' if is_ja else 'Auto (Recommended)'),
            ft.dropdown.Option(key='1', text='1'),
            ft.dropdown.Option(key='8', text='8'),
            ft.dropdown.Option(key='16', text='16'),
            ft.dropdown.Option(key='32', text='32'),
            ft.dropdown.Option(key='64', text='64'),
        ]
        current_batch_val = str(int(getattr(args, 'batch_size', 0)))
        if current_batch_val not in {'0', '1', '8', '16', '32', '64'}:
            batch_options.append(ft.dropdown.Option(key=current_batch_val, text=current_batch_val))
        dropdown_batch_size = ft.Dropdown(
            label=TRANSLATIONS['customize_batch_size_label'][config_obj['langcode']],
            value=current_batch_val,
            options=batch_options,
            width=210,
            dense=True,
            key='settings_batch_size',
        )
        device_id_textfield = ft.TextField(
            label=TRANSLATIONS['customize_device_id_label'][config_obj['langcode']],
            value=str(int(getattr(args, 'device_id', 0))),
            keyboard_type=ft.KeyboardType.NUMBER,
            width=110,
            dense=True,
            key='settings_device_id',
        )
        precision_options = [
            ft.dropdown.Option(
                key='fp32',
                text='FP32（標準・高精度）' if is_ja else 'FP32 (Default, Exact)',
            ),
            ft.dropdown.Option(
                key='fp16',
                text='FP16（高速・省VRAM / 近似精度）' if is_ja else 'FP16 (Fast, Low VRAM / Approx.)',
            ),
        ]
        current_precision_val = normalize_precision_name(
            getattr(args, 'precision', 'fp32')
            if isinstance(getattr(args, 'precision', 'fp32'), str)
            else 'fp32'
        )
        if current_precision_val not in PRECISION_CHOICES:
            current_precision_val = 'fp32'
        dropdown_precision = ft.Dropdown(
            label=TRANSLATIONS['customize_precision_label'][config_obj['langcode']],
            value=current_precision_val,
            options=precision_options,
            width=290,
            dense=True,
            key='settings_precision',
        )
        engine_status_text = ft.Text(
            format_gui_engine_status(origin_detector, origin_recognizer, config_obj['langcode']),
            size=12,
            color=ft.Colors.ON_SURFACE_VARIANT,
            selectable=True,
            key='engine_status_text',
        )
        engine_warning_text = ft.Text(
            '',
            size=12,
            color=ft.Colors.GREY_900,
            expand=True,
            selectable=True,
            key='engine_warning_text',
        )

        def build_current_diagnostics_text(refresh: bool = False) -> str:
            return build_gui_diagnostics_report(
                detector=origin_detector,
                recognizer=origin_recognizer,
                recognizer30=origin_recognizer30,
                recognizer50=origin_recognizer50,
                table_recognizer=origin_table_recognizer,
                table_error=model_state.get('table_error'),
                log_path=GUI_LOG_PATH,
                init_error=model_state.get('error'),
                refresh=refresh,
            )

        def show_diagnostics_dialog(e=None):
            import webbrowser

            lang = config_obj['langcode']
            diag_body = ft.Text(
                build_current_diagnostics_text(refresh=True),
                size=12,
                font_family='monospace',
                selectable=True,
                key='diagnostics_report_text',
            )
            diag_copy_status = ft.Text('', size=12, color=ft.Colors.GREEN_700)
            guidance_container = ft.Container(
                border_radius=8,
                padding=12,
                key='diagnostics_guidance_card',
            )

            async def open_external_url(url: str):
                await asyncio.to_thread(webbrowser.open, url)

            async def open_log_folder(_=None):
                target_dir = GUI_LOG_PATH.parent if GUI_LOG_PATH is not None else APP_DATA_DIR
                try:
                    await open_local_path(target_dir)
                except Exception as exc:
                    diag_copy_status.value = f'{target_dir} ({exc})'
                    diag_copy_status.color = ft.Colors.RED_700
                    page.update()

            def populate_guidance_card():
                active_lang = config_obj['langcode']
                warning_info = collect_engine_warnings(
                    origin_detector,
                    origin_recognizer,
                    origin_recognizer30,
                    origin_recognizer50,
                    origin_table_recognizer,
                    table_error=model_state.get('table_error'),
                )
                active_prof = (
                    getattr(origin_recognizer, 'backend_profile', None)
                    or getattr(origin_detector, 'backend_profile', None)
                )
                active_backend = getattr(active_prof, 'active_backend', None) if active_prof else None
                guidance = build_friendly_diagnostics_guidance(
                    warning_info,
                    langcode=active_lang,
                    init_error=model_state.get('error'),
                    active_backend=active_backend,
                )
                severity = guidance.get('severity', 'ok')
                if severity == 'error':
                    guidance_container.bgcolor = ft.Colors.RED_50
                    guidance_container.border = ft.Border.all(1, ft.Colors.RED_300)
                    icon_name = ft.Icons.ERROR_OUTLINE
                    icon_color = ft.Colors.RED_700
                elif severity == 'warning':
                    guidance_container.bgcolor = ft.Colors.AMBER_100
                    guidance_container.border = ft.Border.all(1, ft.Colors.AMBER_400)
                    icon_name = ft.Icons.WARNING_AMBER
                    icon_color = ft.Colors.AMBER_900
                else:
                    guidance_container.bgcolor = ft.Colors.GREEN_50
                    guidance_container.border = ft.Border.all(1, ft.Colors.GREEN_300)
                    icon_name = ft.Icons.CHECK_CIRCLE_OUTLINE
                    icon_color = ft.Colors.GREEN_700

                card_controls = [
                    ft.Row(
                        controls=[
                            ft.Icon(icon_name, color=icon_color, size=20),
                            ft.Text(
                                guidance['headline'],
                                size=13,
                                weight=ft.FontWeight.W_600,
                                color=ft.Colors.GREY_900,
                                expand=True,
                                selectable=True,
                            ),
                        ],
                        spacing=8,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                ]
                steps = guidance.get('steps') or []
                if steps:
                    card_controls.append(
                        ft.Text(
                            TRANSLATIONS['engine_diagnostics_steps_title'][active_lang],
                            size=12,
                            weight=ft.FontWeight.W_600,
                            color=ft.Colors.GREY_800,
                        )
                    )
                    for idx, step in enumerate(steps, start=1):
                        card_controls.append(
                            ft.Text(
                                f'{idx}. {step}',
                                size=12,
                                color=ft.Colors.GREY_900,
                                selectable=True,
                            )
                        )
                links = guidance.get('links') or []
                if links:
                    link_buttons = [
                        ft.OutlinedButton(
                            link_label,
                            icon=ft.Icons.OPEN_IN_NEW,
                            on_click=lambda _ev, u=link_url: page.run_task(open_external_url, u),
                        )
                        for link_label, link_url in links
                    ]
                    card_controls.append(ft.Row(controls=link_buttons, wrap=True, spacing=8))
                guidance_container.content = ft.Column(
                    controls=card_controls,
                    spacing=4,
                    tight=True,
                )

            populate_guidance_card()

            def refresh_diag(_):
                diag_body.value = build_current_diagnostics_text(refresh=True)
                diag_copy_status.value = ''
                diag_copy_status.color = ft.Colors.GREEN_700
                populate_guidance_card()
                refresh_engine_status_ui()
                page.update()

            async def copy_diag(_):
                try:
                    await clipboard.set(diag_body.value or '')
                    diag_copy_status.color = ft.Colors.GREEN_700
                    diag_copy_status.value = TRANSLATIONS['engine_diagnostics_copied'][config_obj['langcode']]
                except Exception as exc:
                    diag_copy_status.color = ft.Colors.RED_700
                    diag_copy_status.value = f'Copy failed: {exc}'
                page.update()

            def close_diag(_):
                if diag_modal.open:
                    page.pop_dialog()

            diag_modal = ft.AlertDialog(
                modal=True,
                title=ft.Row(
                    controls=[
                        ft.Icon(ft.Icons.MONITOR_HEART, color=ft.Colors.PRIMARY),
                        ft.Text(TRANSLATIONS['engine_diagnostics_title'][lang]),
                    ],
                    spacing=8,
                ),
                content=ft.Column(
                    controls=[
                        guidance_container,
                        diag_copy_status,
                        ft.Text(
                            TRANSLATIONS['engine_diagnostics_tech_details'][lang],
                            size=12,
                            weight=ft.FontWeight.W_600,
                            color=ft.Colors.ON_SURFACE_VARIANT,
                        ),
                        ft.Container(
                            content=ft.Column(
                                controls=[diag_body],
                                scroll=ft.ScrollMode.AUTO,
                            ),
                            width=760,
                            height=250,
                            padding=10,
                            bgcolor=ft.Colors.SURFACE_CONTAINER_LOW,
                            border_radius=8,
                        ),
                    ],
                    tight=True,
                    spacing=6,
                ),
                actions=[
                    ft.TextButton(
                        TRANSLATIONS['engine_diagnostics_open_log_folder'][lang],
                        icon=ft.Icons.FOLDER_OPEN,
                        on_click=open_log_folder,
                        key='diagnostics_open_log_btn',
                    ),
                    ft.TextButton(
                        TRANSLATIONS['engine_diagnostics_refresh'][lang],
                        on_click=refresh_diag,
                        key='diagnostics_refresh_btn',
                    ),
                    ft.TextButton(
                        TRANSLATIONS['engine_diagnostics_copy'][lang],
                        icon=ft.Icons.CONTENT_COPY,
                        on_click=copy_diag,
                        key='diagnostics_copy_btn',
                    ),
                    ft.Button(
                        TRANSLATIONS['common_close'][lang],
                        on_click=close_diag,
                        key='diagnostics_close_btn',
                    ),
                ],
                actions_alignment=ft.MainAxisAlignment.END,
                key='engine_diagnostics_dialog',
            )
            page.show_dialog(diag_modal)

        def dismiss_engine_warning_banner(_=None):
            model_state['banner_dismissed'] = True
            engine_warning_banner.visible = False
            page.update()

        engine_diag_btn = ft.OutlinedButton(
            TRANSLATIONS['engine_diagnostics_btn'][config_obj['langcode']],
            icon=ft.Icons.MONITOR_HEART,
            on_click=show_diagnostics_dialog,
            key='engine_diagnostics_btn',
        )
        engine_warning_banner = ft.Container(
            visible=False,
            padding=ft.Padding.symmetric(horizontal=12, vertical=8),
            bgcolor=ft.Colors.AMBER_100,
            border=ft.Border.all(1, ft.Colors.AMBER_400),
            border_radius=8,
            key='engine_warning_banner',
            content=ft.Row(
                controls=[
                    ft.Icon(ft.Icons.WARNING_AMBER, color=ft.Colors.AMBER_900, size=20),
                    engine_warning_text,
                    ft.TextButton(
                        TRANSLATIONS['engine_fallback_details_btn'][config_obj['langcode']],
                        on_click=show_diagnostics_dialog,
                        key='engine_warning_details_btn',
                    ),
                    ft.IconButton(
                        icon=ft.Icons.CLOSE,
                        icon_color=ft.Colors.AMBER_900,
                        icon_size=16,
                        on_click=dismiss_engine_warning_banner,
                        key='engine_warning_dismiss_btn',
                    ),
                ],
                spacing=8,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
        )

        def refresh_engine_status_ui():
            lang = config_obj['langcode']
            if models_are_ready():
                engine_status_text.value = format_gui_engine_status(
                    origin_detector, origin_recognizer, lang
                )
            elif model_state.get('error') is not None:
                engine_status_text.value = (
                    '実行エンジン: 初期化失敗' if lang == 'ja' else 'Engine: initialization failed'
                )
            else:
                engine_status_text.value = TRANSLATIONS['engine_status_loading'][lang]

            warning_info = collect_engine_warnings(
                origin_detector,
                origin_recognizer,
                origin_recognizer30,
                origin_recognizer50,
                origin_table_recognizer,
                table_error=model_state.get('table_error'),
            )
            has_issue = bool(
                model_state.get('error') is not None
                or warning_info['fallback_occurred']
                or warning_info['table_warning']
            )
            engine_diag_btn.icon = ft.Icons.WARNING_AMBER if has_issue else ft.Icons.MONITOR_HEART

            if TABLE_STRUCTURE_FEATURE is not None:
                if not TABLE_STRUCTURE_FEATURE.is_supported():
                    table_structure_notice_text.value = (
                        TABLE_STRUCTURE_FEATURE.unsupported_reason(lang) or ''
                    )
                    table_structure_notice_text.visible = bool(table_structure_notice_text.value)
                elif model_state.get('table_error') is not None:
                    prefix = TRANSLATIONS['table_structure_unavailable_prefix'][lang]
                    table_structure_notice_text.value = f"{prefix}: {model_state['table_error']}"
                    table_structure_notice_text.visible = True
                else:
                    table_structure_notice_text.value = ''
                    table_structure_notice_text.visible = False

            if has_issue and not model_state.get('banner_dismissed'):
                engine_warning_text.value = format_fallback_banner_summary(
                    warning_info,
                    langcode=lang,
                    init_error=model_state.get('error'),
                )
                engine_warning_banner.visible = bool(engine_warning_text.value)
            else:
                engine_warning_banner.visible = False
        for settings_checkbox in (
            chkbx_json,
            chkbx_txt,
            chkbx_markdown,
            chkbx_html,
            chkbx_xml,
            chkbx_tei,
            chkbx_pdf,
            chkbx_pdf_viztxt,
            chkbx_pdf_merge,
            chkbx_pdf_page_output,
            chkbx_text_daisy,
            chkbx_character_inclusion,
            chkbx_table_structure,
        ):
            if settings_checkbox is not None:
                settings_checkbox.visual_density = ft.VisualDensity.COMPACT
                settings_checkbox.label_style = ft.TextStyle(size=13)
        file_upload_btn = ft.Button(
            TRANSLATIONS['main_file_upload_btn'][config_obj['langcode']],
            icon=ft.Icons.UPLOAD_FILE,
            on_click=pick_files_result,
            key='select_input_file',
        )
        directory_upload_btn = ft.Button(
            TRANSLATIONS['main_directory_upload_btn'][config_obj['langcode']],
            icon=ft.Icons.FOLDER_OPEN,
            on_click=pick_directory_result,
            key='select_input_directory',
        )
        directory_output_btn = ft.Button(
            TRANSLATIONS['main_directory_output_btn'][config_obj['langcode']],
            on_click=pick_output_result,
            key='select_output_directory',
        )
        ocr_btn = ft.Button(
            content='OCR',
            on_click=ocr_button_result,
            style=ft.ButtonStyle(
                padding=30,
                shape=ft.RoundedRectangleBorder(radius=10),
            ),
            disabled=True,
            key='run_ocr',
        )
        stop_ocr_btn=ft.Button(
            content=TRANSLATIONS["main_stop_ocr_btn"][config_obj["langcode"]],
            on_click=request_ocr_cancel,
            disabled=True,
            key='stop_ocr')
        preview_image_col = ft.Column(
            controls=[preview_image],
            width=400,
            height=300,
            expand=False,
        )

        preview_image_int = ft.InteractiveViewer(
            min_scale=1,
            max_scale=10,
            boundary_margin=ft.Margin.all(20),
            content=preview_image_col,
        )
        preview_text_col = ft.Column(
            controls=[preview_view_switch, preview_plain_text, preview_text],
            #controls=[preview_plain_text, preview_text],
            scroll=ft.ScrollMode.ALWAYS,
            height=300,
            expand=True,
        )
        preview_prev_btn = ft.Button(content=TRANSLATIONS['main_prev_btn'][config_obj['langcode']], on_click=prev_image, disabled=True)
        preview_next_btn = ft.Button(content=TRANSLATIONS['main_next_btn'][config_obj['langcode']], on_click=next_image, disabled=True)
        def open_customize_dialog(e):
            dropdown_device.options = build_device_dropdown_options(
                detect_available_backends(refresh=True), config_obj['langcode'] == 'ja'
            )
            refresh_engine_status_ui()
            page.show_dialog(customize_dlg_modal)

        customize_btn = ft.Button(
            TRANSLATIONS['main_customize_btn'][config_obj['langcode']],
            on_click=open_customize_dialog,
            key='customize_output',
        )
        format_grid = ft.Row(
            controls=[
                ft.Container(chkbx_txt, width=100),
                ft.Container(chkbx_json, width=100),
                ft.Container(chkbx_markdown, width=150),
                ft.Container(chkbx_html, width=150),
                ft.Container(chkbx_xml, width=100),
                ft.Container(chkbx_tei, width=100),
            ],
            spacing=4,
            wrap=True,
            run_spacing=0,
            key='settings_format_row',
        )
        pdf_options_panel = ft.Container(
            key='output_pdf_options',
            margin=ft.Margin.only(left=8),
            padding=ft.Padding.only(left=10, top=4, right=8, bottom=4),
            border=ft.Border.only(left=ft.BorderSide(2, ft.Colors.OUTLINE_VARIANT)),
            opacity=1.0 if chkbx_pdf.value else 0.55,
            content=ft.Row(
                controls=[
                    ft.Container(chkbx_pdf_viztxt, expand=5),
                    ft.Container(chkbx_pdf_merge, expand=7),
                ],
                spacing=4,
                tight=True,
            ),
        )
        output_formats_section = build_settings_section(
            TRANSLATIONS['customize_formats_title'][config_obj['langcode']],
            TRANSLATIONS['customize_formats_explain'][config_obj['langcode']],
            ft.Icons.DESCRIPTION_OUTLINED,
            [format_grid],
            'settings_output_formats_section',
        )
        pdf_section = build_settings_section(
            TRANSLATIONS['customize_pdf_title'][config_obj['langcode']],
            TRANSLATIONS['customize_pdf_explain'][config_obj['langcode']],
            ft.Icons.PICTURE_AS_PDF_OUTLINED,
            [
                ft.Row(
                    controls=[
                        ft.Container(chkbx_pdf, expand=True),
                        pdf_resolution_textfield,
                    ],
                    spacing=8,
                ),
                pdf_options_panel,
                ft.Divider(height=1),
                chkbx_pdf_page_output,
            ],
            'settings_pdf_section',
        )
        additional_section = build_settings_section(
            TRANSLATIONS['customize_additional_title'][config_obj['langcode']],
            TRANSLATIONS['customize_additional_explain'][config_obj['langcode']],
            ft.Icons.AUTO_FIX_HIGH_OUTLINED,
            [
                chkbx_text_daisy,
                chkbx_character_inclusion,
                *(
                    [chkbx_table_structure, table_structure_notice_text]
                    if chkbx_table_structure is not None
                    else []
                ),
            ],
            'settings_additional_section',
        )
        execution_engine_section = build_settings_section(
            TRANSLATIONS['customize_execution_title'][config_obj['langcode']],
            TRANSLATIONS['customize_execution_explain'][config_obj['langcode']],
            ft.Icons.MEMORY_OUTLINED,
            [
                ft.Row(
                    controls=[
                        dropdown_device,
                        dropdown_batch_size,
                        dropdown_precision,
                        device_id_textfield,
                        ft.OutlinedButton(
                            TRANSLATIONS['engine_diagnostics_btn'][config_obj['langcode']],
                            icon=ft.Icons.MONITOR_HEART,
                            on_click=show_diagnostics_dialog,
                            key='settings_diagnostics_btn',
                        ),
                    ],
                    spacing=8,
                    wrap=True,
                ),
            ],
            'settings_execution_engine_section',
        )
        customize_dlg_modal = ft.AlertDialog(
            modal=True,
            title=ft.Row(
                controls=[
                    ft.Icon(ft.Icons.TUNE, color=ft.Colors.PRIMARY),
                    ft.Text(TRANSLATIONS['customize_dlg_modal_title'][config_obj['langcode']]),
                ],
                spacing=10,
            ),
            content=ft.Column(
                controls=[
                    ft.Text(
                        TRANSLATIONS['customize_dlg_modal_explain'][config_obj['langcode']],
                        color=ft.Colors.ON_SURFACE_VARIANT,
                    ),
                    execution_engine_section,
                    output_formats_section,
                    ft.ResponsiveRow(
                        controls=[
                            ft.Container(pdf_section, col={'sm': 12, 'md': 8}),
                            ft.Container(additional_section, col={'sm': 12, 'md': 4}),
                        ],
                        spacing=8,
                        run_spacing=8,
                        key='settings_detail_row',
                    ),
                ],
                width=860,
                height=430,
                spacing=8,
                scroll=ft.ScrollMode.AUTO,
            ),
            actions=[
                ft.Button(
                    TRANSLATIONS['customize_save'][config_obj['langcode']],
                    icon=ft.Icons.CHECK,
                    on_click=handle_customize_dlg_modal_close,
                    key='customize_output_ok',
                ),
            ],
            actions_alignment=ft.MainAxisAlignment.END,
            title_padding=ft.Padding.only(left=20, top=16, right=20, bottom=8),
            content_padding=ft.Padding.only(left=20, top=0, right=20, bottom=4),
            actions_padding=ft.Padding.only(left=20, top=4, right=20, bottom=12),
            inset_padding=ft.Padding.symmetric(horizontal=28, vertical=16),
            key='customize_output_dialog',
        )
        selector = ImageSelector(
            page,
            config_obj,
            detector=origin_detector,
            recognizer30=origin_recognizer30,
            recognizer50=origin_recognizer50,
            recognizer100=origin_recognizer,
            table_feature=TABLE_STRUCTURE_FEATURE,
            table_recognizer=origin_table_recognizer,
            outputdirpath=selected_output_path.value,
            busy_tracker=busy_tracker,
        )

        capture_tool = CaptureTool(
            page,
            config_obj,
            detector=origin_detector,
            recognizer30=origin_recognizer30,
            recognizer50=origin_recognizer50,
            recognizer100=origin_recognizer,
            table_feature=TABLE_STRUCTURE_FEATURE,
            table_recognizer=origin_table_recognizer,
            busy_tracker=busy_tracker,
        )
        if selected_output_path.value:
            capture_tool.outputdirpath = selected_output_path.value

        llm_capture_extension = (
            LLM_FEATURE.attach_capture_tool(capture_tool, config_obj, save_config)
            if LLM_FEATURE is not None
            else None
        )
        datapool_capture_extension = (
            DATA_POOL_FEATURE.attach_capture_tool(
                capture_tool,
                llm_capture_extension,
                config_obj,
                Path(PDFTMPPATH),
            )
            if DATA_POOL_FEATURE is not None
            else None
        )

        async def batch_ocr_image(image_path: str):
            if not models_are_ready():
                raise RuntimeError('OCRモデルの読み込みが完了していません')
            img = await asyncio.to_thread(load_rgb_image_array, image_path)
            page_result = await asyncio.to_thread(
                ocr._run_ocr_on_image_array,
                detector=origin_detector,
                recognizer30=origin_recognizer30,
                recognizer50=origin_recognizer50,
                recognizer100=origin_recognizer,
                inputname=os.path.basename(image_path),
                img=img,
                outputpath=PDFTMPPATH,
                save_viz=False,
                text_transform=get_character_inclusion_transform(config_obj),
                include_markdown=False,
            )
            return {
                'text': page_result['text'],
                'json_lines': page_result['json_lines'],
            }

        datapool_batch_extension = (
            DATA_POOL_FEATURE.attach_case_batch(
                page,
                config_obj,
                save_config,
                LLM_FEATURE,
                batch_ocr_image,
                file_picker,
                models_are_ready(),
                parts_control,
            )
            if DATA_POOL_FEATURE is not None and LLM_FEATURE is not None
            else None
        )
        datapool_header_controls = (
            DATA_POOL_FEATURE.build_auth_controls(
                page,
                config_obj,
                save_config,
                datapool_capture_extension,
                datapool_batch_extension,
            )
            if DATA_POOL_FEATURE is not None
            else []
        )

        crop_btn = ft.Button(
            content='Crop&OCR',
            on_click=selector.open_dialog,
            style=ft.ButtonStyle(
                padding=10,
                shape=ft.RoundedRectangleBorder(radius=10),
            ),
            disabled=True,
            key='crop_ocr',
        )
        cap_btn = ft.Button(
            content=TRANSLATIONS['main_cap_btn'][config_obj['langcode']],
            on_click=capture_tool.start_capture,
            style=ft.ButtonStyle(
                padding=10,
                shape=ft.RoundedRectangleBorder(radius=10),
            ),
            disabled=True,
            key='capture_ocr',
        )
        explain_label = ft.Text(TRANSLATIONS['main_explain'][config_obj['langcode']])
        localebutton = ft.CupertinoSlidingSegmentedButton(
            selected_index=0 if config_obj['langcode'] == 'ja' else 1,
            thumb_color=ft.Colors.BLUE_400,
            on_change=handle_locale_change,
            controls=[ft.Text('日本語'), ft.Text('English')],
            key='locale',
        )
        input_value_semantics = ft.Semantics(
            # Keep a non-empty layout anchor while no path is selected so
            # Flutter does not prune this status from the semantics tree.
            content=ft.Row(
                [selected_input_path, ft.Text('\u00a0')],
                spacing=0,
                tight=True,
            ),
            label=TRANSLATIONS['a11y_main_input_value'][config_obj['langcode']],
            value=(
                selected_input_path.value
                or TRANSLATIONS['a11y_value_not_selected'][config_obj['langcode']]
            ),
            container=True,
            live_region=True,
            read_only=True,
            focusable=True,
            exclude_semantics=True,
            key='a11y_input_value',
        )
        output_value_semantics = ft.Semantics(
            content=selected_output_path,
            label=TRANSLATIONS['a11y_main_output_value'][config_obj['langcode']],
            value=(
                selected_output_path.value
                or TRANSLATIONS['a11y_value_not_selected'][config_obj['langcode']]
            ),
            container=True,
            live_region=True,
            read_only=True,
            focusable=True,
            key='a11y_output_value',
        )
        preview_name_semantics = ft.Semantics(
            content=current_visualizeimgname,
            label=TRANSLATIONS['a11y_main_preview_name'][config_obj['langcode']],
            value=TRANSLATIONS['a11y_value_not_selected'][config_obj['langcode']],
            container=True,
            live_region=True,
            read_only=True,
            focusable=True,
            key='a11y_preview_name',
        )
        progress_message_semantics = ft.Semantics(
            content=progressmessage,
            label=(
                progressmessage.value
                or TRANSLATIONS['a11y_main_status_group'][config_obj['langcode']]
            ),
            container=True,
            live_region=True,
            read_only=True,
            exclude_semantics=True,
            key='a11y_progress_message',
        )
        progress_status_semantics = ft.Semantics(
            content=ft.Column([
                progress_message_semantics,
                progressbar,
            ], width=400),
            label=TRANSLATIONS['a11y_main_status_group'][config_obj['langcode']],
            container=True,
            live_region=True,
            read_only=True,
            key='a11y_progress_status',
        )
        app_settings_section = build_main_semantics_section(
            TRANSLATIONS['a11y_main_settings_group'][config_obj['langcode']],
            ft.Row([
                localebutton,
                engine_status_text,
                engine_diag_btn,
                *datapool_header_controls,
            ], spacing=12, vertical_alignment=ft.CrossAxisAlignment.CENTER, wrap=True),
            'a11y_app_settings_section',
        )
        quick_actions_section = build_main_semantics_section(
            TRANSLATIONS['a11y_main_quick_group'][config_obj['langcode']],
            ft.Row([
                explain_label,
                cap_btn,
            ]),
            'a11y_quick_actions_section',
        )
        input_section = build_main_semantics_section(
            TRANSLATIONS['a11y_main_input_group'][config_obj['langcode']],
            ft.Row([
                file_upload_btn,
                directory_upload_btn,
                ft.Text(TRANSLATIONS['main_target_label'][config_obj['langcode']]),
                input_value_semantics,
            ]),
            'a11y_input_section',
        )
        output_section = build_main_semantics_section(
            TRANSLATIONS['a11y_main_output_group'][config_obj['langcode']],
            ft.Row([
                directory_output_btn,
                ft.Text(TRANSLATIONS['main_output_label'][config_obj['langcode']]),
                output_value_semantics,
            ]),
            'a11y_output_section',
        )
        run_controls_semantics = ft.Semantics(
            content=ft.Row([ocr_btn, stop_ocr_btn, crop_btn]),
            label=TRANSLATIONS['a11y_main_run_controls_group'][config_obj['langcode']],
            container=True,
            key='a11y_run_controls',
        )
        options_status_semantics = ft.Semantics(
            content=ft.Column([
                chkbx_visualize,
                ft.Row([
                    customize_btn,
                    progress_status_semantics,
                ]),
            ]),
            label=TRANSLATIONS['a11y_main_options_status_group'][config_obj['langcode']],
            container=True,
            key='a11y_options_status',
        )
        execution_section = build_main_semantics_section(
            TRANSLATIONS['a11y_main_execution_group'][config_obj['langcode']],
            ft.Row([
                run_controls_semantics,
                options_status_semantics,
            ]),
            'a11y_execution_section',
        )
        preview_controls_semantics = ft.Semantics(
            content=ft.Row([
                ft.Text(TRANSLATIONS['main_preview_label'][config_obj['langcode']]),
                preview_prev_btn,
                preview_next_btn,
                preview_name_semantics,
            ]),
            label=TRANSLATIONS['a11y_main_preview_controls_group'][config_obj['langcode']],
            container=True,
            key='a11y_preview_controls',
        )
        preview_content_semantics = ft.Semantics(
            content=ft.Row([preview_image_int, preview_text_col]),
            label=TRANSLATIONS['a11y_main_preview_content_group'][config_obj['langcode']],
            container=True,
            key='a11y_preview_content',
        )
        preview_section = build_main_semantics_section(
            TRANSLATIONS['a11y_main_preview_group'][config_obj['langcode']],
            ft.Column([
                preview_controls_semantics,
                preview_content_semantics,
            ]),
            'a11y_preview_section',
        )
        main_interaction_sections.extend([
            app_settings_section,
            quick_actions_section,
            input_section,
            output_section,
            execution_section,
            preview_section,
        ])

        async def invoke_control_event(control, event_name='on_click'):
            if control.disabled:
                return False
            handler = getattr(control, event_name, None)
            if handler is None:
                return False
            result = handler(None)
            if inspect.isawaitable(result):
                await result
            return True

        async def handle_main_keyboard_shortcut(e: ft.KeyboardEvent):
            if model_state['loading'] or any(
                section.disabled for section in main_interaction_sections
            ):
                return
            if page_has_open_dialog(page):
                return

            action = resolve_main_shortcut(
                e.key,
                ctrl=e.ctrl,
                meta=e.meta,
                shift=e.shift,
                alt=e.alt,
            )
            shortcut_targets = {
                'select_input_file': file_upload_btn,
                'select_input_directory': directory_upload_btn,
                'select_output_directory': directory_output_btn,
                'run_ocr': ocr_btn,
                'crop_ocr': crop_btn,
                'stop_ocr': stop_ocr_btn,
                'customize_output': customize_btn,
                'capture_ocr': cap_btn,
                'preview_previous': preview_prev_btn,
                'preview_next': preview_next_btn,
            }
            if action == 'toggle_markdown':
                if preview_view_switch.disabled or preview_view_switch.on_change is None:
                    return
                preview_view_switch.value = not bool(preview_view_switch.value)
                await invoke_control_event(preview_view_switch, 'on_change')
                return
            target = shortcut_targets.get(action)
            if target is not None:
                await invoke_control_event(target)

        page.on_keyboard_event = handle_main_keyboard_shortcut
        register_model_loading_handlers()
        initial_model_lock = not models_are_ready()
        set_model_loading_lock(initial_model_lock)
        parts_control(initial_model_lock)
        page.add(
            app_settings_section,
            engine_warning_banner,
            quick_actions_section,
            ft.Divider(),
            input_section,
            ft.Divider(),
            output_section,
            ft.Divider(),
            execution_section,
            ft.Divider(),
            preview_section,
        )
        page.update()

        async def initialize_models():
            nonlocal origin_detector, origin_recognizer
            nonlocal origin_recognizer30, origin_recognizer50
            nonlocal origin_table_recognizer, loaded_engine_state

            started_at = time.perf_counter()
            had_previous = models_are_ready()
            prev_models = (
                origin_detector,
                origin_recognizer,
                origin_recognizer30,
                origin_recognizer50,
                origin_table_recognizer,
            )
            prev_engine_state = dict(loaded_engine_state) if loaded_engine_state is not None else None
            reload_scope = model_state.get('reload_scope', 'all')
            attempted_state = extract_engine_state(args)
            LOGGER.info(
                'Initializing OCR models | scope=%s | target=%s',
                reload_scope,
                attempted_state,
            )
            reuse_det = origin_detector if (reload_scope == 'recognizers' and had_previous) else None
            reuse_table = origin_table_recognizer if (reload_scope == 'recognizers' and had_previous) else None
            try:
                (
                    new_detector,
                    new_recognizer,
                    new_recognizer30,
                    new_recognizer50,
                ) = await asyncio.to_thread(
                    load_ocr_models,
                    args,
                    download_models=download_models,
                    reuse_detector=reuse_det,
                )
                if reuse_det is not None:
                    new_table_recognizer = reuse_table
                    new_table_error = model_state.get('table_error')
                else:
                    new_table_recognizer = None
                    new_table_error = None
                    if TABLE_STRUCTURE_FEATURE is not None:
                        if not TABLE_STRUCTURE_FEATURE.is_supported():
                            new_table_error = TABLE_STRUCTURE_FEATURE.unsupported_reason(
                                config_obj['langcode']
                            )
                            LOGGER.info('Table structure gated off: %s', new_table_error)
                        else:
                            try:
                                new_table_recognizer = await asyncio.to_thread(
                                    load_table_recognizer, args, download_models=download_models
                                )
                            except Exception as table_ex:
                                new_table_error = table_ex
                                LOGGER.warning('Table structure initialization failed: %s', table_ex)
                                print(f'[WARN] Table structure initialization failed: {table_ex}')

                origin_detector = new_detector
                origin_recognizer = new_recognizer
                origin_recognizer30 = new_recognizer30
                origin_recognizer50 = new_recognizer50
                origin_table_recognizer = new_table_recognizer
                loaded_engine_state = attempted_state
                model_state['table_error'] = new_table_error
                model_state['elapsed'] = time.perf_counter() - started_at
                model_state['error'] = None
                prev_models = None

                for model_label, model_obj in (
                    ('DEIMv2', origin_detector),
                    ('PARSeq-100', origin_recognizer),
                    ('PARSeq-30', origin_recognizer30),
                    ('PARSeq-50', origin_recognizer50),
                    ('TableCenterNet', origin_table_recognizer),
                ):
                    prof = getattr(model_obj, 'backend_profile', None)
                    if prof is not None:
                        LOGGER.info(
                            'Model ready: %s | backend=%s | providers=%s | fallback=%s | warning=%s',
                            model_label,
                            prof.active_backend,
                            prof.active_providers,
                            prof.fallback_occurred,
                            prof.warning_message,
                        )
                LOGGER.info('Model initialization completed in %.2fs', model_state['elapsed'])

                for tool in (selector, capture_tool):
                    tool.detector = origin_detector
                    tool.recognizer30 = origin_recognizer30
                    tool.recognizer50 = origin_recognizer50
                    tool.recognizer100 = origin_recognizer
                    tool.table_recognizer = origin_table_recognizer

                progressbar.value = 0
                refresh_engine_status_ui()
                if config_obj['langcode'] == 'ja':
                    if TABLE_STRUCTURE_FEATURE is None:
                        set_progress_message(f'OCRモデル準備完了 ({model_state["elapsed"]:.1f}秒)')
                    elif model_state.get('table_error') is None:
                        set_progress_message(f'OCR / 表推定モデル準備完了 ({model_state["elapsed"]:.1f}秒)')
                    else:
                        set_progress_message(f'OCRモデル準備完了 / 表推定初期化失敗 ({model_state["elapsed"]:.1f}秒)')
                else:
                    if TABLE_STRUCTURE_FEATURE is None:
                        set_progress_message(f'OCR models ready ({model_state["elapsed"]:.1f} sec)')
                    elif model_state.get('table_error') is None:
                        set_progress_message(f'OCR / TableInfer models ready ({model_state["elapsed"]:.1f} sec)')
                    else:
                        set_progress_message(f'OCR models ready / TableInfer models unavailable ({model_state["elapsed"]:.1f} sec)')
            except Exception as ex:
                LOGGER.exception('OCR model initialization failed: %s', ex)
                model_state['error'] = ex
                progressbar.value = 0
                if had_previous and prev_models is not None:
                    (
                        origin_detector,
                        origin_recognizer,
                        origin_recognizer30,
                        origin_recognizer50,
                        origin_table_recognizer,
                    ) = prev_models
                    loaded_engine_state = prev_engine_state
                    model_state['error'] = None
                    refresh_engine_status_ui()
                    if config_obj['langcode'] == 'ja':
                        set_progress_message(f'モデル再読み込み失敗（以前の設定を維持）: {ex}')
                    else:
                        set_progress_message(f'Model reload failed (kept previous engine): {ex}')
                else:
                    refresh_engine_status_ui()
                    set_progress_message(f'OCR model initialization failed: {ex}')
            finally:
                model_state['loading'] = False
                model_state['reload_scope'] = 'all'
                if datapool_batch_extension is not None:
                    datapool_batch_extension.set_models_ready(models_are_ready())
                set_model_loading_lock(False)
                parts_control(False)
                page.update()
                if pending_engine_reload['pending'] and not busy_tracker.active:
                    pending_engine_reload['pending'] = False
                    resolve_gui_engine_args(args, config_obj, cli_overrides)
                    if (
                        model_state.get('error') is None
                        or extract_engine_state(args) != attempted_state
                    ):
                        start_engine_reload()

        if models_are_ready():
            set_model_loading_lock(False)
            parts_control(False)
            refresh_engine_status_ui()
            if model_state['elapsed'] is not None:
                if config_obj['langcode'] == 'ja':
                    if TABLE_STRUCTURE_FEATURE is None:
                        set_progress_message(f'OCRモデル準備完了 ({model_state["elapsed"]:.1f}秒)')
                    elif model_state.get('table_error') is None:
                        set_progress_message(f'OCR / 表推定モデル準備完了 ({model_state["elapsed"]:.1f}秒)')
                    else:
                        set_progress_message(f'OCRモデル準備完了 / 表推定初期化失敗 ({model_state["elapsed"]:.1f}秒)')
                else:
                    if TABLE_STRUCTURE_FEATURE is None:
                        set_progress_message(f'OCR models ready ({model_state["elapsed"]:.1f} sec)')
                    elif model_state.get('table_error') is None:
                        set_progress_message(f'OCR / TableInfer models ready ({model_state["elapsed"]:.1f} sec)')
                    else:
                        set_progress_message(f'OCR models ready / TableInfer models unavailable ({model_state["elapsed"]:.1f} sec)')
            page.update()
        elif not model_state['loading']:
            model_state['loading'] = True
            set_model_loading_lock(True)
            parts_control(True)
            progressbar.value = None
            if config_obj['langcode'] == 'ja':
                set_progress_message('OCRモデルを読み込み中…')
            else:
                set_progress_message('Loading OCR models…')
            page.update()
            page.run_task(initialize_models)

    renderui()


if __name__ == '__main__':
    ft.run(main, assets_dir=str(ASSETS_DIR))
