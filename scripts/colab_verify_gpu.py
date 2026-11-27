"""End-to-end GPU verification and benchmark script for NDLOCR-Lite (designed for Colab T4 / CUDA)."""

from __future__ import annotations

import argparse
from argparse import Namespace
import contextlib
import difflib
import io
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Dict, List, Optional
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


def _query_process_vram_mib() -> Optional[int]:
    pid = str(os.getpid())
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if res.returncode == 0:
            for line in res.stdout.splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2 and parts[0] == pid:
                    return int(float(parts[1]))
    except Exception:
        pass
    return None


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
    precision: str = "fp32",
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
        precision=precision,
        strict_device=strict_device,
        check_device=False,
        json_only=False,
        enable_tcy=enable_tcy,
    )


def _extract_xml_texts(xml_text: str) -> List[str]:
    root = ET.fromstring(xml_text)
    return [line.get("STRING", "") for line in root.findall(".//LINE")]


def _sample_real_crops() -> List[np.ndarray]:
    img = cv2.imread(str(RESOURCE_IMAGES[0]))
    h, w = img.shape[:2]
    return [
        img[int(h * 0.15):int(h * 0.35), int(w * 0.72):int(w * 0.77)],
        img[int(h * 0.15):int(h * 0.45), int(w * 0.62):int(w * 0.67)],
        img[int(h * 0.20):int(h * 0.50), int(w * 0.52):int(w * 0.57)],
        np.full((32, 160, 3), 255, dtype=np.uint8),
    ]


def check_parseq_tiers_cuda(
    models: Dict[str, Path],
    *,
    precision: str = "fp32",
    batch_size: int = 64,
) -> Dict[str, object]:
    with (SRC_DIR / "config" / "NDLmoji.yaml").open("r", encoding="utf-8") as stream:
        charlist = list(yaml.safe_load(stream)["model"]["charset_train"])

    crops = _sample_real_crops()
    summary = {}
    for tier in ("rec30", "rec50", "rec100"):
        path = models[tier]
        t0 = time.perf_counter()
        patched = patcher.prepare_parseq_model(path, enable_dynamic_batch=True, precision=precision)
        patch_time = time.perf_counter() - t0
        cpu_rec = PARSEQ(str(path), charlist, device="cpu", batch_size=1)
        cuda_rec = PARSEQ(
            str(path),
            charlist,
            device="cuda",
            batch_size=batch_size,
            strict_device=True,
            precision=precision,
        )
        batch_tensor = cuda_rec.preprocess_batch(crops)
        raw_out = cuda_rec.session.run(
            cuda_rec.output_names, {cuda_rec.input_names[0]: batch_tensor}
        )[0]
        nan_count = int(np.size(raw_out) - np.count_nonzero(np.isfinite(raw_out)))
        if nan_count > 0:
            raise AssertionError(f"Non-finite outputs detected on {tier} ({precision}): nan_count={nan_count}")
        if raw_out.dtype != np.float32:
            raise AssertionError(f"Expected float32 output logits on {tier} ({precision}), got {raw_out.dtype}")

        cpu_texts = [cpu_rec.read(c) for c in crops]
        cuda_texts = cuda_rec.read_batch(crops)
        exact_match = cpu_texts == cuda_texts
        char_sim = difflib.SequenceMatcher(
            None, "\n".join(cpu_texts), "\n".join(cuda_texts)
        ).ratio()
        if precision == "fp32" and not exact_match:
            raise AssertionError(
                f"CUDA vs CPU PARSeq mismatch on {tier}: cpu={cpu_texts} cuda={cuda_texts}"
            )
        if precision == "fp16" and char_sim < 0.90:
            raise AssertionError(
                f"FP16 PARSeq char_similarity {char_sim:.4f} fell below 0.90 on {tier}: "
                f"cpu={cpu_texts} cuda={cuda_texts}"
            )
        summary[tier] = {
            "precision": cuda_rec.effective_precision,
            "providers": cuda_rec.backend_profile.active_providers,
            "batch_size": cuda_rec.batch_size,
            "dynamic_batch": cuda_rec.dynamic_batch,
            "from_cache": patched.from_cache,
            "patch_or_lookup_sec": round(patch_time, 3),
            "output_dtype": str(raw_out.dtype),
            "nan_count": nan_count,
            "exact_match_count": sum(a == b for a, b in zip(cpu_texts, cuda_texts)),
            "total_crops": len(crops),
            "exact_match": exact_match,
            "char_similarity": round(char_sim, 6),
        }
    return summary


def check_full_ocr_parity(
    models: Dict[str, Path],
    *,
    precision: str = "fp32",
    batch_size: int = 64,
) -> Dict[str, object]:
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
                        precision="fp32",
                    )
                )
                ocr.process(
                    _make_ocr_args(
                        sourcedir=src_dir,
                        output=out_cuda,
                        models=models,
                        device="cuda",
                        batch_size=batch_size,
                        strict_device=True,
                        enable_tcy=enable_tcy,
                        precision=precision,
                    )
                )

            txt_exact = True
            xml_text_exact = True
            json_text_exact = True
            total_lines = 0
            exact_lines = 0
            exact_boxes = 0
            line_count_diff = 0
            cpu_chars_all = []
            cuda_chars_all = []

            for img_path in RESOURCE_IMAGES:
                stem = img_path.stem
                cpu_txt = (out_cpu / f"{stem}.txt").read_text("utf-8")
                cuda_txt = (out_cuda / f"{stem}.txt").read_text("utf-8")
                cpu_chars_all.append(cpu_txt)
                cuda_chars_all.append(cuda_txt)
                if cpu_txt != cuda_txt:
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
                total_lines += len(t_cpu)
                line_count_diff += len(t_cuda) - len(t_cpu)
                for c_item, g_item in zip(t_cpu, t_cuda):
                    if c_item[0] == g_item[0]:
                        exact_lines += 1
                    if c_item[1] == g_item[1]:
                        exact_boxes += 1

            if precision == "fp32" and not (txt_exact and xml_text_exact and json_text_exact):
                raise AssertionError(
                    f"Full OCR parity mismatch ({label}): txt={txt_exact}, xml={xml_text_exact}, json={json_text_exact}"
                )
            char_similarity = difflib.SequenceMatcher(
                None, "\n".join(cpu_chars_all), "\n".join(cuda_chars_all)
            ).ratio()
            line_exact_rate = exact_lines / max(1, total_lines)
            if precision == "fp16":
                if line_count_diff != 0 or exact_boxes != total_lines:
                    raise AssertionError(
                        f"FP16 layout/box parity mismatch ({label}): "
                        f"line_count_diff={line_count_diff}, exact_boxes={exact_boxes}/{total_lines}"
                    )
                if char_similarity < 0.99 or line_exact_rate < 0.98:
                    raise AssertionError(
                        f"FP16 quality floor violated ({label}): "
                        f"char_similarity={char_similarity:.4f}, line_exact_rate={line_exact_rate:.4f}"
                    )
            results[label] = {
                "precision": precision,
                "images_tested": len(RESOURCE_IMAGES),
                "txt_exact_match": txt_exact,
                "xml_line_text_exact_match": xml_text_exact,
                "json_text_and_bbox_exact_match": json_text_exact,
                "line_exact_rate": round(line_exact_rate, 4),
                "box_exact_rate": round(exact_boxes / max(1, total_lines), 4),
                "char_similarity": round(char_similarity, 6),
                "line_count_diff": line_count_diff,
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


def benchmark_multipage(
    models: Dict[str, Path],
    num_pages: int = 12,
    *,
    precision: str = "fp32",
    batch_size: int = 64,
) -> Dict[str, object]:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        src_dir = tmp / "bench_pages"
        src_dir.mkdir()
        for idx in range(num_pages):
            src = RESOURCE_IMAGES[idx % len(RESOURCE_IMAGES)]
            shutil.copy2(src, src_dir / f"page_{idx + 1:02d}{src.suffix}")

        precisions_to_run = ["fp32", "fp16"] if precision == "both" else [precision]

        # Warm up dynamic-batch ONNX caches + CUDA JIT once before steady-state timing.
        warm_src = tmp / "warm"
        warm_src.mkdir()
        shutil.copy2(RESOURCE_IMAGES[0], warm_src / RESOURCE_IMAGES[0].name)
        for prec in precisions_to_run:
            with contextlib.redirect_stdout(io.StringIO()):
                ocr.process(
                    _make_ocr_args(
                        sourcedir=warm_src,
                        output=tmp / f"warm_out_{prec}",
                        models=models,
                        device="cuda",
                        batch_size=batch_size,
                        strict_device=True,
                        precision=prec,
                    )
                )

        modes = [
            ("cpu_batch1", "cpu", 1, False, "fp32"),
            ("cuda_batch1", "cuda", 1, True, "fp32"),
        ]
        for prec in precisions_to_run:
            modes.append((f"cuda_dynbatch{batch_size}_{prec}_pipeline", "cuda", batch_size, True, prec))

        timings: Dict[str, Dict[str, object]] = {}
        for mode_name, device, bs, strict, prec in modes:
            out_dir = tmp / f"out_{mode_name}"
            args = _make_ocr_args(
                sourcedir=src_dir,
                output=out_dir,
                models=models,
                device=device,
                batch_size=bs,
                strict_device=strict,
                precision=prec,
            )
            t_init0 = time.perf_counter()
            with contextlib.redirect_stdout(io.StringIO()):
                det, rec100, rec30, rec50 = ocr.initialize_ocr_models(args)
            init_sec = time.perf_counter() - t_init0
            vram_mib = _query_process_vram_mib()
            del det, rec100, rec30, rec50

            t0 = time.perf_counter()
            with contextlib.redirect_stdout(io.StringIO()):
                ocr.process(args)
            elapsed = time.perf_counter() - t0
            vram_after = _query_process_vram_mib() or vram_mib
            timings[mode_name] = {
                "pages": num_pages,
                "precision": prec,
                "init_sec": round(init_sec, 3),
                "elapsed_sec": round(elapsed, 3),
                "pages_per_sec": round(num_pages / elapsed, 3),
                "vram_mib": vram_after,
            }

        cpu_sec = float(timings["cpu_batch1"]["elapsed_sec"])
        cuda1_sec = float(timings["cuda_batch1"]["elapsed_sec"])
        speedup = {"cuda_batch1": round(cpu_sec / cuda1_sec, 2)}
        for prec in precisions_to_run:
            key = f"cuda_dynbatch{batch_size}_{prec}_pipeline"
            p_sec = float(timings[key]["elapsed_sec"])
            speedup[key] = round(cpu_sec / p_sec, 2)
            speedup[f"{key}_vs_cuda_batch1"] = round(cuda1_sec / p_sec, 2)
        timings["speedup_vs_cpu"] = speedup
        return timings


def benchmark_fp16_blocklists(
    models: Dict[str, Path],
    *,
    batch_size: int = 32,
) -> Dict[str, object]:
    """Compare FP16 op-blocklist variants (with attention-score protection) against FP32 on CUDA."""
    with (SRC_DIR / "config" / "NDLmoji.yaml").open("r", encoding="utf-8") as stream:
        charlist = list(yaml.safe_load(stream)["model"]["charset_train"])

    base_crops = _sample_real_crops()
    batch_crops = [base_crops[i % len(base_crops)] for i in range(batch_size)]
    variants = {
        "default": patcher.FP16_EXTRA_OP_BLOCK_LIST,
        "with_layernorm": ("LayerNormalization", "Softmax"),
    }
    results: Dict[str, object] = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        for tier in ("rec30", "rec50", "rec100"):
            path = models[tier]
            cpu_rec = PARSEQ(str(path), charlist, device="cpu", batch_size=1)
            ref_texts = [cpu_rec.read(c) for c in base_crops]

            fp32_dst = tmp / f"{tier}_fp32.onnx"
            patcher.patch_onnx_dynamic_batch(path, fp32_dst, precision="fp32", dynamic_batch=True)
            fp32_rec = PARSEQ(str(fp32_dst), charlist, device="cuda", batch_size=batch_size, strict_device=True)
            fp32_tensor = fp32_rec.preprocess_batch(batch_crops)
            for _ in range(2):
                fp32_rec.session.run(fp32_rec.output_names, {fp32_rec.input_names[0]: fp32_tensor})
            fp32_latencies = []
            for _ in range(10):
                t0 = time.perf_counter()
                fp32_rec.session.run(fp32_rec.output_names, {fp32_rec.input_names[0]: fp32_tensor})
                fp32_latencies.append((time.perf_counter() - t0) * 1000.0)
            fp32_med_ms = statistics.median(fp32_latencies)

            tier_res: Dict[str, object] = {"fp32_median_ms": round(fp32_med_ms, 2)}
            for variant_name, blocklist in variants.items():
                dst = tmp / f"{tier}_{variant_name}.onnx"
                t_build0 = time.perf_counter()
                patcher.patch_onnx_dynamic_batch(
                    path,
                    dst,
                    precision="fp16",
                    dynamic_batch=True,
                    fp16_extra_op_block_list=blocklist,
                    fp16_protect_attention_scores=True,
                )
                build_sec = time.perf_counter() - t_build0
                rec = PARSEQ(str(dst), charlist, device="cuda", batch_size=batch_size, strict_device=True)
                tensor = rec.preprocess_batch(batch_crops)
                # Warmup
                for _ in range(2):
                    rec.session.run(rec.output_names, {rec.input_names[0]: tensor})
                latencies = []
                last_out = None
                for _ in range(10):
                    t0 = time.perf_counter()
                    last_out = rec.session.run(rec.output_names, {rec.input_names[0]: tensor})[0]
                    latencies.append((time.perf_counter() - t0) * 1000.0)
                med_ms = statistics.median(latencies)
                nan_count = int(np.size(last_out) - np.count_nonzero(np.isfinite(last_out)))
                decoded = rec._decode_outputs(last_out[:len(base_crops)])
                tier_res[variant_name] = {
                    "blocklist": list(blocklist),
                    "protect_attention_scores": True,
                    "build_sec": round(build_sec, 3),
                    "median_ms": round(med_ms, 2),
                    "speedup_vs_fp32": round(fp32_med_ms / med_ms, 2),
                    "nan_count": nan_count,
                    "output_dtype": str(last_out.dtype),
                    "exact_match_count": sum(a == b for a, b in zip(ref_texts, decoded)),
                    "total_crops": len(base_crops),
                }
            results[tier] = tier_res
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify and benchmark NDLOCR-Lite on GPU")
    parser.add_argument("--pages", type=int, default=12, help="Number of pages for multi-page benchmark")
    parser.add_argument("--batch-size", type=int, default=64, help="PARSeq batch size for CUDA checks/benchmarks")
    parser.add_argument(
        "--precision",
        type=str,
        choices=["fp32", "fp16", "both"],
        default="fp32",
        help="Precision mode to verify and benchmark (fp32, fp16, or both)",
    )
    parser.add_argument(
        "--fp16-blocklist-bench",
        action="store_true",
        help="Compare full vs minimal FP32-island op blocklists on CUDA for PARSeq FP16",
    )
    parser.add_argument("--json-out", type=str, default=None, help="Optional path to save JSON report")
    parser.add_argument("--skip-benchmark", action="store_true", help="Skip 12-page throughput benchmark")
    args = parser.parse_args()

    print("=== 1. Hardware & ORT Environment ===")
    nvidia_smi = _run_cmd_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"])
    nvcc_ver = _run_cmd_output(["nvcc", "--version"])
    print(f"nvidia-smi: {nvidia_smi}")
    print(f"nvcc: {nvcc_ver.splitlines()[-1] if nvcc_ver else nvcc_ver}")
    print(ort_provider.format_device_diagnostics())

    avail = ort_provider.detect_available_backends()
    if not avail.get("cuda", {}).get("available", False):
        cuda_detail = avail.get("cuda", {}).get("detail", "CUDAExecutionProvider not available")
        print(
            f"[ERROR] This verification script requires CUDAExecutionProvider ({cuda_detail}).",
            file=sys.stderr,
        )
        sys.exit(2)

    models = _resolve_models(SRC_DIR / "model")
    precisions = ["fp32", "fp16"] if args.precision == "both" else [args.precision]

    print("\n=== 2. PARSeq 3-Tier CUDA Dynamic-Batch Check ===")
    parseq_summary = {}
    for prec in precisions:
        parseq_summary[prec] = check_parseq_tiers_cuda(
            models, precision=prec, batch_size=args.batch_size
        )
    print(json.dumps(parseq_summary, ensure_ascii=False, indent=2))

    print("\n=== 3. Full OCR Parity Check (CPU batch=1 vs CUDA, TCY off/on) ===")
    parity_summary = {}
    for prec in precisions:
        parity_summary[prec] = check_full_ocr_parity(
            models, precision=prec, batch_size=args.batch_size
        )
    print(json.dumps(parity_summary, ensure_ascii=False, indent=2))

    print("\n=== 4. TableCenterNet (DeformConv) CUDA Check ===")
    table_summary = check_tablecenternet_cuda(models)
    print(json.dumps(table_summary, ensure_ascii=False, indent=2))

    blocklist_summary = {}
    if args.fp16_blocklist_bench:
        print("\n=== 5. FP16 Op-Blocklist Comparison (full vs minimal) ===")
        blocklist_summary = benchmark_fp16_blocklists(models, batch_size=min(32, args.batch_size))
        print(json.dumps(blocklist_summary, ensure_ascii=False, indent=2))

    bench_summary = {}
    if not args.skip_benchmark:
        print(f"\n=== 6. {args.pages}-Page Throughput Benchmark ===")
        bench_summary = benchmark_multipage(
            models,
            num_pages=args.pages,
            precision=args.precision,
            batch_size=args.batch_size,
        )
        print(json.dumps(bench_summary, ensure_ascii=False, indent=2))

    report = {
        "gpu": nvidia_smi,
        "ort_version": ort_provider.ort.__version__,
        "parseq": parseq_summary,
        "ocr_parity": parity_summary,
        "tablecenternet": table_summary,
        "fp16_blocklist_bench": blocklist_summary,
        "benchmark": bench_summary,
    }
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
