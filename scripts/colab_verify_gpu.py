"""End-to-end GPU verification and benchmark script for NDLOCR-Lite (designed for Colab T4 / CUDA)."""

from __future__ import annotations

import argparse
from argparse import Namespace
import contextlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Dict, List
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import ocr
import onnx_batch_patcher as patcher
import ort_provider
from parseq import PARSEQ
from tablecenternet import TableCenterNetRecognizer


RESOURCE_IMAGES = [
    REPO_ROOT / "resource" / "digidepo_2531162_0024.jpg",
    REPO_ROOT / "resource" / "digidepo_11048278_po_geppo1803_00021.jpg",
    REPO_ROOT / "resource" / "digidepo_3048008_0025.jpg",
    REPO_ROOT / "resource" / "tategaki2026-04-24-094138.png",
    REPO_ROOT / "resource" / "yokogaki2026-04-24-103617.png",
]


def _run_cmd_output(cmd: List[str]) -> str:
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=15, check=False)
        out = (res.stdout or res.stderr).strip()
        return out or f"(exit={res.returncode}, no output)"
    except Exception as exc:
        return f"(failed: {exc})"


def _resolve_models(model_dir: Path) -> Dict[str, Path]:
    models = {
        "det": model_dir / "deim-s-1024x1024-261124.onnx",
        "rec30": model_dir / "parseq-ndl-24x256-30-tiny-499epoch-261124.onnx",
        "rec50": model_dir / "parseq-ndl-24x384-50-tiny-278epoch-261124.onnx",
        "rec100": model_dir / "parseq-ndl-24x768-100-tiny-153epoch-261124.onnx",
        "table": model_dir / "tablecenternet_s_800_v3.onnx",
    }
    if all(path.is_file() for path in models.values()):
        return models

    from model_download import MODELS, TABLE_MODEL, get_model_path

    key_map = {
        "det": MODELS["det_weights"],
        "rec30": MODELS["rec_weights30"],
        "rec50": MODELS["rec_weights50"],
        "rec100": MODELS["rec_weights"],
        "table": TABLE_MODEL,
    }
    return {
        name: get_model_path(*spec, model_dir)
        for name, spec in key_map.items()
    }


def _make_ocr_args(
    *,
    sourcedir: Path,
    output: Path,
    models: Dict[str, Path],
    device: str,
    batch_size: int,
    strict_device: bool,
    enable_tcy: bool = False,
) -> Namespace:
    return Namespace(
        sourcedir=str(sourcedir),
        sourceimg=None,
        sourcepdf=None,
        output=str(output),
        viz=False,
        det_weights=str(models["det"]),
        det_classes=str(SRC_DIR / "config" / "ndl.yaml"),
        det_score_threshold=0.2,
        det_conf_threshold=0.25,
        det_iou_threshold=0.2,
        det_overlap_threshold=0.9,
        rec_weights30=str(models["rec30"]),
        rec_weights50=str(models["rec50"]),
        rec_weights=str(models["rec100"]),
        rec_classes=str(SRC_DIR / "config" / "NDLmoji.yaml"),
        device=device,
        device_id=0,
        batch_size=batch_size,
        strict_device=strict_device,
        check_device=False,
        json_only=False,
        enable_tcy=enable_tcy,
    )


def _extract_xml_texts(xml_text: str) -> List[str]:
    root = ET.fromstring(xml_text)
    return [line.get("STRING", "") for line in root.findall(".//LINE")]


def check_parseq_tiers_cuda(models: Dict[str, Path]) -> Dict[str, object]:
    with (SRC_DIR / "config" / "NDLmoji.yaml").open("r", encoding="utf-8") as stream:
        charlist = list(yaml.safe_load(stream)["model"]["charset_train"])

    img = cv2.imread(str(RESOURCE_IMAGES[0]))
    h, w = img.shape[:2]
    crops = [
        img[int(h * 0.15):int(h * 0.35), int(w * 0.72):int(w * 0.77)],
        img[int(h * 0.15):int(h * 0.45), int(w * 0.62):int(w * 0.67)],
        img[int(h * 0.20):int(h * 0.50), int(w * 0.52):int(w * 0.57)],
        np.full((32, 160, 3), 255, dtype=np.uint8),
    ]

    summary = {}
    for tier in ("rec30", "rec50", "rec100"):
        path = models[tier]
        t0 = time.perf_counter()
        patched = patcher.prepare_parseq_model(path, enable_dynamic_batch=True, precision="fp32")
        patch_time = time.perf_counter() - t0
        cpu_rec = PARSEQ(str(path), charlist, device="cpu", batch_size=1)
        cuda_rec = PARSEQ(
            str(path),
            charlist,
            device="cuda",
            batch_size=32,
            strict_device=True,
        )
        cpu_texts = [cpu_rec.read(c) for c in crops]
        cuda_texts = cuda_rec.read_batch(crops)
        if cpu_texts != cuda_texts:
            raise AssertionError(
                f"CUDA vs CPU PARSeq mismatch on {tier}: cpu={cpu_texts} cuda={cuda_texts}"
            )
        summary[tier] = {
            "providers": cuda_rec.backend_profile.active_providers,
            "batch_size": cuda_rec.batch_size,
            "dynamic_batch": cuda_rec.dynamic_batch,
            "from_cache": patched.from_cache,
            "patch_or_lookup_sec": round(patch_time, 3),
            "exact_match": True,
        }
    return summary


def check_full_ocr_parity(models: Dict[str, Path]) -> Dict[str, object]:
    results = {}
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        src_dir = tmp / "inputs"
        src_dir.mkdir()
        for img_path in RESOURCE_IMAGES:
            shutil.copy2(img_path, src_dir / img_path.name)

        for enable_tcy in (False, True):
            label = "tcy_on" if enable_tcy else "standard"
            out_cpu = tmp / f"out_cpu_{label}"
            out_cuda = tmp / f"out_cuda_{label}"
            with contextlib.redirect_stdout(io.StringIO()):
                ocr.process(
                    _make_ocr_args(
                        sourcedir=src_dir,
                        output=out_cpu,
                        models=models,
                        device="cpu",
                        batch_size=1,
                        strict_device=False,
                        enable_tcy=enable_tcy,
                    )
                )
                ocr.process(
                    _make_ocr_args(
                        sourcedir=src_dir,
                        output=out_cuda,
                        models=models,
                        device="cuda",
                        batch_size=32,
                        strict_device=True,
                        enable_tcy=enable_tcy,
                    )
                )

            txt_exact = True
            xml_text_exact = True
            json_text_exact = True
            for img_path in RESOURCE_IMAGES:
                stem = img_path.stem
                if (out_cpu / f"{stem}.txt").read_text("utf-8") != (out_cuda / f"{stem}.txt").read_text("utf-8"):
                    txt_exact = False
                if _extract_xml_texts((out_cpu / f"{stem}.xml").read_text("utf-8")) != _extract_xml_texts(
                    (out_cuda / f"{stem}.xml").read_text("utf-8")
                ):
                    xml_text_exact = False
                j_cpu = json.loads((out_cpu / f"{stem}.json").read_text("utf-8"))["contents"]
                j_cuda = json.loads((out_cuda / f"{stem}.json").read_text("utf-8"))["contents"]
                t_cpu = [[item["text"], item["boundingBox"]] for page in j_cpu for item in page]
                t_cuda = [[item["text"], item["boundingBox"]] for page in j_cuda for item in page]
                if t_cpu != t_cuda:
                    json_text_exact = False

            if not (txt_exact and xml_text_exact and json_text_exact):
                raise AssertionError(
                    f"Full OCR parity mismatch ({label}): txt={txt_exact}, xml={xml_text_exact}, json={json_text_exact}"
                )
            results[label] = {
                "images_tested": len(RESOURCE_IMAGES),
                "txt_exact_match": txt_exact,
                "xml_line_text_exact_match": xml_text_exact,
                "json_text_and_bbox_exact_match": json_text_exact,
            }
    return results


def check_tablecenternet_cuda(models: Dict[str, Path]) -> Dict[str, object]:
    synthetic_table = np.full((640, 640, 3), 255, dtype=np.uint8)
    for y in (80, 220, 360, 520):
        cv2.line(synthetic_table, (60, y), (580, y), (0, 0, 0), 3)
    for x in (60, 230, 400, 580):
        cv2.line(synthetic_table, (x, 80), (x, 520), (0, 0, 0), 3)

    t_cpu = TableCenterNetRecognizer(models["table"], device="cpu").load()
    t_cuda = TableCenterNetRecognizer(
        models["table"], device="cuda", strict_device=True
    ).load()

    cells_cpu = t_cpu.recognize_rgb(synthetic_table)
    cells_cuda = t_cuda.recognize_rgb(synthetic_table)
    return {
        "providers": t_cuda.backend_profile.active_providers,
        "fallback_occurred": t_cuda.backend_profile.fallback_occurred,
        "cpu_cell_count": len(cells_cpu),
        "cuda_cell_count": len(cells_cuda),
    }


def benchmark_multipage(models: Dict[str, Path], num_pages: int = 12) -> Dict[str, object]:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        src_dir = tmp / "bench_pages"
        src_dir.mkdir()
        for idx in range(num_pages):
            src = RESOURCE_IMAGES[idx % len(RESOURCE_IMAGES)]
            shutil.copy2(src, src_dir / f"page_{idx + 1:02d}{src.suffix}")

        # Warm up dynamic-batch ONNX cache + CUDA JIT compilation once before timing steady-state
        warm_src = tmp / "warm"
        warm_src.mkdir()
        shutil.copy2(RESOURCE_IMAGES[0], warm_src / RESOURCE_IMAGES[0].name)
        with contextlib.redirect_stdout(io.StringIO()):
            ocr.process(
                _make_ocr_args(
                    sourcedir=warm_src,
                    output=tmp / "warm_out",
                    models=models,
                    device="cuda",
                    batch_size=32,
                    strict_device=True,
                )
            )

        modes = [
            ("cpu_batch1", "cpu", 1, False),
            ("cuda_batch1", "cuda", 1, True),
            ("cuda_dynbatch32_pipeline", "cuda", 32, True),
        ]
        timings: Dict[str, Dict[str, float]] = {}
        for mode_name, device, batch_size, strict in modes:
            out_dir = tmp / f"out_{mode_name}"
            args = _make_ocr_args(
                sourcedir=src_dir,
                output=out_dir,
                models=models,
                device=device,
                batch_size=batch_size,
                strict_device=strict,
            )
            t0 = time.perf_counter()
            with contextlib.redirect_stdout(io.StringIO()):
                ocr.process(args)
            elapsed = time.perf_counter() - t0
            timings[mode_name] = {
                "pages": num_pages,
                "elapsed_sec": round(elapsed, 3),
                "pages_per_sec": round(num_pages / elapsed, 3),
            }

        cpu_sec = timings["cpu_batch1"]["elapsed_sec"]
        cuda1_sec = timings["cuda_batch1"]["elapsed_sec"]
        cuda32_sec = timings["cuda_dynbatch32_pipeline"]["elapsed_sec"]
        timings["speedup_vs_cpu"] = {
            "cuda_batch1": round(cpu_sec / cuda1_sec, 2),
            "cuda_dynbatch32_pipeline": round(cpu_sec / cuda32_sec, 2),
            "dynbatch32_vs_cuda_batch1": round(cuda1_sec / cuda32_sec, 2),
        }
        return timings


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify and benchmark NDLOCR-Lite on GPU")
    parser.add_argument("--pages", type=int, default=12, help="Number of pages for multi-page benchmark")
    parser.add_argument("--json-out", type=str, default=None, help="Optional path to save JSON report")
    parser.add_argument("--skip-benchmark", action="store_true", help="Skip 12-page throughput benchmark")
    args = parser.parse_args()

    print("=== 1. Hardware & ORT Environment ===")
    nvidia_smi = _run_cmd_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"])
    nvcc_ver = _run_cmd_output(["nvcc", "--version"])
    print(f"nvidia-smi: {nvidia_smi}")
    print(f"nvcc: {nvcc_ver.splitlines()[-1] if nvcc_ver else nvcc_ver}")
    print(ort_provider.format_device_diagnostics())

    models = _resolve_models(SRC_DIR / "model")

    print("\n=== 2. PARSeq 3-Tier CUDA Dynamic-Batch Check ===")
    parseq_summary = check_parseq_tiers_cuda(models)
    print(json.dumps(parseq_summary, ensure_ascii=False, indent=2))

    print("\n=== 3. Full OCR Parity Check (CPU batch=1 vs CUDA batch=32, TCY off/on) ===")
    parity_summary = check_full_ocr_parity(models)
    print(json.dumps(parity_summary, ensure_ascii=False, indent=2))

    print("\n=== 4. TableCenterNet (DeformConv) CUDA Check ===")
    table_summary = check_tablecenternet_cuda(models)
    print(json.dumps(table_summary, ensure_ascii=False, indent=2))

    bench_summary = {}
    if not args.skip_benchmark:
        print(f"\n=== 5. {args.pages}-Page Throughput Benchmark ===")
        bench_summary = benchmark_multipage(models, num_pages=args.pages)
        print(json.dumps(bench_summary, ensure_ascii=False, indent=2))

    report = {
        "gpu": nvidia_smi,
        "ort_version": ort_provider.ort.__version__,
        "parseq": parseq_summary,
        "ocr_parity": parity_summary,
        "tablecenternet": table_summary,
        "benchmark": bench_summary,
    }
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
