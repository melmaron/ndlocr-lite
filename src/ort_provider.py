"""Unified ONNX Runtime execution provider management for CPU and multi-vendor GPUs."""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib
from pathlib import Path
import re
import sys
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple

import onnxruntime as ort


DEVICE_CHOICES: Tuple[str, ...] = (
    "auto",
    "cpu",
    "cuda",
    "migraphx",
    "rocm",
    "openvino",
    "webgpu",
    "dml",
)

AUTO_PRIORITY: Tuple[str, ...] = (
    "cuda",
    "migraphx",
    "rocm",
    "openvino",
    "webgpu",
    "dml",
    "cpu",
)

DEFORM_CONV_MIN_ORT_VERSION: Tuple[int, int, int] = (1, 25, 0)

_DEFAULT_BATCH_SIZES: Dict[str, int] = {
    "cuda": 32,
    "migraphx": 32,
    "rocm": 32,
    "openvino": 16,
    "webgpu": 16,
    "dml": 16,
    "cpu": 1,
}

_BACKEND_DISPLAY_NAMES: Dict[str, str] = {
    "cuda": "GPU (CUDA)",
    "migraphx": "GPU (MIGraphX)",
    "rocm": "GPU (ROCm)",
    "openvino": "GPU (OpenVINO)",
    "webgpu": "GPU (WebGPU)",
    "dml": "GPU (DirectML)",
    "cpu": "CPU",
}

_BACKEND_MONOLITHIC_PROVIDER: Dict[str, str] = {
    "cuda": "CUDAExecutionProvider",
    "migraphx": "MIGraphXExecutionProvider",
    "rocm": "ROCMExecutionProvider",
    "openvino": "OpenVINOExecutionProvider",
    "webgpu": "WebGpuExecutionProvider",
    "dml": "DmlExecutionProvider",
    "cpu": "CPUExecutionProvider",
}

_INSTALL_HINTS: Dict[str, str] = {
    "cuda": (
        "NVIDIA CUDA環境が必要です。`pip install onnxruntime-gpu`（CUDA 13系）"
        " または CUDA 12系インデックスから `onnxruntime-gpu` を導入し、"
        "対応する CUDA / cuDNN ランタイムを確認してください。"
    ),
    "migraphx": (
        "Linux AMD ROCm環境向けに `onnxruntime-migraphx` と AMDMIGraphX / ROCm "
        "ランタイムの導入が必要です。"
    ),
    "rocm": (
        "Linux AMD ROCm環境向けに `onnxruntime-rocm`（または `onnxruntime-migraphx`）"
        "と ROCm ランタイムの導入が必要です。"
    ),
    "openvino": (
        "Intel Arc / iGPU 環境向けに `pip install onnxruntime-ep-openvino` と"
        "最新の Intel GPU ドライバ（Linux では OpenCL / Level Zero）を導入してください。"
    ),
    "webgpu": (
        "Radeon / Arc / NVIDIA 共通GPU向けに `pip install onnxruntime-ep-webgpu` を導入してください"
        "（Linux では `libvulkan.so.1` と Vulkan ドライバ、Windows では DirectX 12 対応ドライバが必要です）。"
    ),
    "dml": (
        "Windows DirectML環境向けに `pip install onnxruntime-directml` を導入してください"
        "（注意: onnxruntime-directml 1.24系では v2 の表構造認識 TableCenterNet は非対応です）。"
    ),
}

_PLUGIN_SPECS: Tuple[Tuple[str, str], ...] = (
    ("webgpu", "onnxruntime_ep_webgpu"),
    ("openvino_ep", "onnxruntime_ep_openvino"),
)

_BACKEND_PLUGIN_NAME: Dict[str, str] = {
    "webgpu": "webgpu",
    "openvino": "openvino_ep",
}

_PLUGIN_LOCK = threading.Lock()
_REGISTERED_PLUGINS: Dict[str, str] = {}
_PLUGIN_ERRORS: Dict[str, str] = {}
_CUDA_DLLS_PRELOADED = False
_CUDA_DLLS_ERROR: Optional[str] = None
_WARNED_KEYS: set[Tuple[str, str, str]] = set()


@dataclass
class BackendProfile:
    """Resolved ONNX Runtime backend properties for a created session."""

    requested_device: str
    active_backend: str
    active_providers: List[str] = field(default_factory=list)
    device_id: int = 0
    default_batch_size: int = 1
    parallel_session_init: bool = True
    concurrent_session_run: bool = True
    fallback_occurred: bool = False
    warning_message: Optional[str] = None
    display_name: str = "CPU"


def parse_ort_version(version_str: Optional[str] = None) -> Tuple[int, int, int]:
    """Parse an ONNX Runtime version string into a comparable 3-tuple."""
    raw = version_str or getattr(ort, "__version__", "0.0.0")
    nums = [int(part) for part in re.findall(r"\d+", str(raw))[:3]]
    while len(nums) < 3:
        nums.append(0)
    return (nums[0], nums[1], nums[2])


def ort_supports_deform_conv(version_str: Optional[str] = None) -> bool:
    """Return True if the active ORT version supports standard ONNX DeformConv."""
    return parse_ort_version(version_str) >= DEFORM_CONV_MIN_ORT_VERSION


def normalize_device_name(device: Optional[str]) -> str:
    """Normalize user-supplied device names to canonical choices."""
    raw = (device or "auto").strip().casefold()
    aliases = {
        "": "auto",
        "directml": "dml",
        "gpu": "auto",
        "ov": "openvino",
    }
    normalized = aliases.get(raw, raw)
    if normalized not in DEVICE_CHOICES:
        return raw
    return normalized


def should_parallelize_model_init(device: Optional[str]) -> bool:
    """Only parallelize session creation on pure CPU to avoid GPU context races."""
    return normalize_device_name(device) == "cpu"


def preload_cuda_dlls_if_needed() -> None:
    """Best-effort DLL preloading for pip-installed CUDA/cuDNN wheels on Windows."""
    global _CUDA_DLLS_PRELOADED, _CUDA_DLLS_ERROR
    with _PLUGIN_LOCK:
        if _CUDA_DLLS_PRELOADED:
            return
        _CUDA_DLLS_PRELOADED = True
        preload_fn = getattr(ort, "preload_dlls", None)
        if not callable(preload_fn):
            return
        try:
            preload_fn()
        except Exception as exc:  # pragma: no cover - environment dependent
            _CUDA_DLLS_ERROR = str(exc)


def register_plugin_eps() -> Dict[str, str]:
    """Register installed ORT plugin execution provider libraries once per process."""
    register_fn = getattr(ort, "register_execution_provider_library", None)
    if not callable(register_fn):
        return dict(_REGISTERED_PLUGINS)

    with _PLUGIN_LOCK:
        for reg_name, module_name in _PLUGIN_SPECS:
            if reg_name in _REGISTERED_PLUGINS:
                continue
            try:
                module = importlib.import_module(module_name)
                lib_path = str(module.get_library_path())
            except ImportError:
                continue
            except Exception as exc:
                _PLUGIN_ERRORS[reg_name] = f"import/path error: {exc}"
                continue

            try:
                register_fn(reg_name, lib_path)
                _REGISTERED_PLUGINS[reg_name] = lib_path
                _PLUGIN_ERRORS.pop(reg_name, None)
            except Exception as exc:
                msg = str(exc)
                if "already registered" in msg.casefold():
                    _REGISTERED_PLUGINS[reg_name] = lib_path
                    _PLUGIN_ERRORS.pop(reg_name, None)
                else:
                    _PLUGIN_ERRORS[reg_name] = msg
    return dict(_REGISTERED_PLUGINS)


def _safe_ep_devices() -> List[Any]:
    get_devs = getattr(ort, "get_ep_devices", None)
    if not callable(get_devs):
        return []
    try:
        return list(get_devs())
    except Exception:
        return []


def _is_gpu_hw_device(ep_device: Any) -> bool:
    meta = getattr(ep_device, "ep_metadata", None) or {}
    ov_dev = str(meta.get("ov_device", "")).strip().upper()
    if ov_dev:
        return ov_dev.startswith("GPU")
    hw = getattr(ep_device, "device", None)
    hw_type = getattr(hw, "type", None)
    gpu_enum = getattr(getattr(ort, "OrtHardwareDeviceType", None), "GPU", None)
    if gpu_enum is not None and hw_type is not None:
        return hw_type == gpu_enum
    return "GPU" in str(hw_type).upper()


def _find_plugin_devices_for_backend(backend: str, ep_devices: Sequence[Any]) -> List[Any]:
    if backend == "webgpu":
        return [
            d for d in ep_devices
            if "webgpu" in str(getattr(d, "ep_name", "")).casefold()
        ]
    if backend == "openvino":
        return [
            d for d in ep_devices
            if str(getattr(d, "ep_name", "")) == "OpenVINOExecutionProvider"
            and _is_gpu_hw_device(d)
        ]
    return []


def detect_available_backends() -> Dict[str, Dict[str, Any]]:
    """Inspect installed EPs and discovered hardware devices for all supported backends."""
    register_plugin_eps()
    available_providers = set(ort.get_available_providers())
    ep_devices = _safe_ep_devices()
    results: Dict[str, Dict[str, Any]] = {}

    for backend in AUTO_PRIORITY:
        mono_provider = _BACKEND_MONOLITHIC_PROVIDER[backend]
        plugin_devs = _find_plugin_devices_for_backend(backend, ep_devices)
        plugin_reg_name = _BACKEND_PLUGIN_NAME.get(backend)
        plugin_is_loaded = bool(plugin_reg_name and plugin_reg_name in _REGISTERED_PLUGINS)
        if backend == "cpu":
            results[backend] = {
                "available": True,
                "mode": "builtin",
                "devices": ["CPU"],
                "detail": "CPUExecutionProvider",
            }
            continue

        if plugin_devs:
            desc = []
            for d in plugin_devs:
                meta = getattr(d, "ep_metadata", None) or {}
                hw = getattr(d, "device", None)
                vendor = getattr(hw, "vendor", "") if hw else ""
                ov_dev = meta.get("ov_device", "")
                label = " ".join(part for part in (str(d.ep_name), str(ov_dev), str(vendor)) if part)
                desc.append(label)
            results[backend] = {
                "available": True,
                "mode": "plugin",
                "devices": desc,
                "detail": "; ".join(desc),
            }
        elif (not plugin_is_loaded) and mono_provider in available_providers:
            results[backend] = {
                "available": True,
                "mode": "monolithic",
                "devices": [mono_provider],
                "detail": mono_provider,
            }
        else:
            extra_note = ""
            if backend == "openvino":
                ov_all = [
                    d for d in ep_devices
                    if str(getattr(d, "ep_name", "")) == "OpenVINOExecutionProvider"
                ]
                if ov_all or plugin_is_loaded:
                    extra_note = "OpenVINO plugin loaded, but no Intel GPU device detected (CPU/NPU only)"
                elif "openvino_ep" in _PLUGIN_ERRORS:
                    extra_note = f"plugin error: {_PLUGIN_ERRORS['openvino_ep']}"
            elif backend == "webgpu":
                if "webgpu" in _PLUGIN_ERRORS:
                    extra_note = f"plugin error: {_PLUGIN_ERRORS['webgpu']}"
                elif plugin_is_loaded:
                    extra_note = "WebGPU plugin loaded, but no compatible GPU adapter detected"

            results[backend] = {
                "available": False,
                "mode": "none",
                "devices": [],
                "detail": extra_note or f"{mono_provider} not available",
            }

    return results


def _build_session_options(cpu_single_thread: bool = False) -> ort.SessionOptions:
    opt = ort.SessionOptions()
    opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if cpu_single_thread:
        opt.intra_op_num_threads = 1
        opt.inter_op_num_threads = 1
    return opt


def _make_profile(
    requested_device: str,
    active_backend: str,
    active_providers: List[str],
    device_id: int,
    *,
    fallback_occurred: bool = False,
    warning_message: Optional[str] = None,
) -> BackendProfile:
    is_cpu = active_backend == "cpu"
    return BackendProfile(
        requested_device=requested_device,
        active_backend=active_backend,
        active_providers=list(active_providers),
        device_id=int(device_id),
        default_batch_size=_DEFAULT_BATCH_SIZES.get(active_backend, 1),
        parallel_session_init=is_cpu,
        concurrent_session_run=is_cpu,
        fallback_occurred=fallback_occurred,
        warning_message=warning_message,
        display_name=_BACKEND_DISPLAY_NAMES.get(active_backend, active_backend.upper()),
    )


def _create_session_for_backend(
    model_path: str | Path,
    backend: str,
    *,
    device_id: int,
    cpu_single_thread: bool,
) -> Tuple[ort.InferenceSession, List[str]]:
    model_str = str(model_path)
    register_plugin_eps()
    available_providers = set(ort.get_available_providers())
    ep_devices = _safe_ep_devices()

    if backend == "cpu":
        opt = _build_session_options(cpu_single_thread=cpu_single_thread)
        session = ort.InferenceSession(model_str, opt, providers=["CPUExecutionProvider"])
        return session, list(session.get_providers())

    if backend == "cuda":
        preload_cuda_dlls_if_needed()
        if "CUDAExecutionProvider" not in available_providers:
            raise RuntimeError("CUDAExecutionProvider is not available in installed onnxruntime.")
        opt = _build_session_options(cpu_single_thread=False)
        providers = [("CUDAExecutionProvider", {"device_id": int(device_id)}), "CPUExecutionProvider"]
        session = ort.InferenceSession(model_str, opt, providers=providers)
        active = list(session.get_providers())
        if "CUDAExecutionProvider" not in active:
            raise RuntimeError(f"CUDAExecutionProvider fell back to {active}.")
        return session, active

    if backend == "migraphx":
        if "MIGraphXExecutionProvider" not in available_providers:
            raise RuntimeError("MIGraphXExecutionProvider is not available.")
        opt = _build_session_options(cpu_single_thread=False)
        providers = [("MIGraphXExecutionProvider", {"device_id": int(device_id)}), "CPUExecutionProvider"]
        session = ort.InferenceSession(model_str, opt, providers=providers)
        active = list(session.get_providers())
        if "MIGraphXExecutionProvider" not in active:
            raise RuntimeError(f"MIGraphXExecutionProvider fell back to {active}.")
        return session, active

    if backend == "rocm":
        if "ROCMExecutionProvider" not in available_providers:
            raise RuntimeError("ROCMExecutionProvider is not available.")
        opt = _build_session_options(cpu_single_thread=False)
        providers = [("ROCMExecutionProvider", {"device_id": int(device_id)}), "CPUExecutionProvider"]
        session = ort.InferenceSession(model_str, opt, providers=providers)
        active = list(session.get_providers())
        if "ROCMExecutionProvider" not in active:
            raise RuntimeError(f"ROCMExecutionProvider fell back to {active}.")
        return session, active

    if backend == "openvino":
        gpu_devs = _find_plugin_devices_for_backend("openvino", ep_devices)
        if gpu_devs:
            chosen = gpu_devs[device_id] if 0 <= device_id < len(gpu_devs) else gpu_devs[0]
            opt = _build_session_options(cpu_single_thread=False)
            opt.add_provider_for_devices([chosen], {})
            session = ort.InferenceSession(model_str, sess_options=opt)
            active = list(session.get_providers())
            if "OpenVINOExecutionProvider" not in active:
                raise RuntimeError(f"OpenVINOExecutionProvider plugin did not activate: {active}")
            return session, active
        if "openvino_ep" in _REGISTERED_PLUGINS:
            raise RuntimeError("OpenVINO plugin loaded, but no Intel GPU device detected (CPU/NPU only).")
        if "OpenVINOExecutionProvider" in available_providers:
            opt = _build_session_options(cpu_single_thread=False)
            ov_device = "GPU" if int(device_id) == 0 else f"GPU.{int(device_id)}"
            providers = [("OpenVINOExecutionProvider", {"device_type": ov_device}), "CPUExecutionProvider"]
            session = ort.InferenceSession(model_str, opt, providers=providers)
            active = list(session.get_providers())
            if "OpenVINOExecutionProvider" not in active:
                raise RuntimeError(f"OpenVINOExecutionProvider fell back to {active}.")
            return session, active
        raise RuntimeError("OpenVINO GPU device is not available.")

    if backend == "webgpu":
        webgpu_devs = _find_plugin_devices_for_backend("webgpu", ep_devices)
        if webgpu_devs:
            chosen = webgpu_devs[device_id] if 0 <= device_id < len(webgpu_devs) else webgpu_devs[0]
            ep_name = str(chosen.ep_name)
            opt = _build_session_options(cpu_single_thread=False)
            opt.add_provider_for_devices([chosen], {})
            session = ort.InferenceSession(model_str, sess_options=opt)
            active = list(session.get_providers())
            if ep_name in active or any("webgpu" in p.casefold() for p in active):
                return session, active
            raise RuntimeError(f"WebGPU plugin EP did not activate: {active}")
        if "webgpu" in _REGISTERED_PLUGINS:
            raise RuntimeError("WebGPU plugin loaded, but no compatible GPU adapter detected.")
        if "WebGpuExecutionProvider" in available_providers:
            opt = _build_session_options(cpu_single_thread=False)
            providers = [("WebGpuExecutionProvider", {"deviceId": int(device_id)}), "CPUExecutionProvider"]
            session = ort.InferenceSession(model_str, opt, providers=providers)
            active = list(session.get_providers())
            if "WebGpuExecutionProvider" not in active:
                raise RuntimeError(f"WebGpuExecutionProvider fell back to {active}.")
            return session, active
        raise RuntimeError("WebGPU Execution Provider is not available.")

    if backend == "dml":
        if "DmlExecutionProvider" not in available_providers:
            raise RuntimeError("DmlExecutionProvider is not available.")
        opt = _build_session_options(cpu_single_thread=False)
        opt.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opt.enable_mem_pattern = False
        providers = [("DmlExecutionProvider", {"device_id": int(device_id)}), "CPUExecutionProvider"]
        session = ort.InferenceSession(model_str, opt, providers=providers)
        active = list(session.get_providers())
        if "DmlExecutionProvider" not in active:
            raise RuntimeError(f"DmlExecutionProvider fell back to {active}.")
        return session, active

    raise ValueError(
        f"Unsupported device '{backend}'. Supported choices: {', '.join(DEVICE_CHOICES)}"
    )


def _emit_warning_once(key: Tuple[str, str, str], message: str) -> None:
    if key in _WARNED_KEYS:
        return
    _WARNED_KEYS.add(key)
    print(message, file=sys.stderr)


def _format_fallback_warning(
    requested: str,
    fallback_backend: str,
    reason: str,
    model_label: str,
) -> str:
    hint = _INSTALL_HINTS.get(requested, "")
    lines = [
        f"[WARN] 指定されたデバイス '{requested}' を {model_label} で有効化できなかったため、'{fallback_backend}' へフォールバックしました。",
        f"       (Requested device '{requested}' failed for {model_label}: {reason})",
    ]
    if hint:
        lines.append(f"       対処法: {hint}")
    return "\n".join(lines)


def create_inference_session(
    model_path: str | Path,
    *,
    device: str = "auto",
    device_id: int = 0,
    strict_device: bool = False,
    cpu_single_thread: bool = False,
    model_label: str = "model",
) -> Tuple[ort.InferenceSession, BackendProfile]:
    """Create an ONNX Runtime InferenceSession for the requested device or 'auto'."""
    requested = normalize_device_name(device)
    if requested not in DEVICE_CHOICES:
        raise ValueError(
            f"Unsupported device '{device}'. Choose from: {', '.join(DEVICE_CHOICES)}"
        )

    if requested == "auto":
        availability = detect_available_backends()
        failures: List[str] = []
        for candidate in AUTO_PRIORITY:
            if candidate == "cpu":
                continue
            if not availability.get(candidate, {}).get("available"):
                continue
            try:
                session, active_providers = _create_session_for_backend(
                    model_path,
                    candidate,
                    device_id=device_id,
                    cpu_single_thread=cpu_single_thread,
                )
                return session, _make_profile(
                    requested_device="auto",
                    active_backend=candidate,
                    active_providers=active_providers,
                    device_id=device_id,
                )
            except Exception as exc:
                failures.append(f"{candidate}: {exc}")

        session, active_providers = _create_session_for_backend(
            model_path,
            "cpu",
            device_id=device_id,
            cpu_single_thread=cpu_single_thread,
        )
        warning_msg = None
        if failures:
            warning_msg = (
                f"[WARN] 自動選択されたGPUバックエンドの初期化に失敗したため CPU で実行します ({model_label}): "
                + "; ".join(failures)
            )
            _emit_warning_once(("auto", "cpu", ";".join(failures)), warning_msg)
        return session, _make_profile(
            requested_device="auto",
            active_backend="cpu",
            active_providers=active_providers,
            device_id=device_id,
            fallback_occurred=bool(failures),
            warning_message=warning_msg,
        )

    if requested == "cpu":
        session, active_providers = _create_session_for_backend(
            model_path,
            "cpu",
            device_id=device_id,
            cpu_single_thread=cpu_single_thread,
        )
        return session, _make_profile(
            requested_device="cpu",
            active_backend="cpu",
            active_providers=active_providers,
            device_id=device_id,
        )

    # Explicit GPU backend requested
    try:
        session, active_providers = _create_session_for_backend(
            model_path,
            requested,
            device_id=device_id,
            cpu_single_thread=cpu_single_thread,
        )
        if requested == "dml" and not ort_supports_deform_conv() and "table" in model_label.casefold():
            _emit_warning_once(
                ("dml", "deformconv", ort.__version__),
                "[WARN] onnxruntime-directml (ORT < 1.25) は TableCenterNet の DeformConv に非対応です。",
            )
        return session, _make_profile(
            requested_device=requested,
            active_backend=requested,
            active_providers=active_providers,
            device_id=device_id,
        )
    except Exception as exc:
        if strict_device:
            hint = _INSTALL_HINTS.get(requested, "")
            raise RuntimeError(
                f"Requested device '{requested}' failed for {model_label}: {exc}"
                + (f" | {hint}" if hint else "")
            ) from exc

        session, active_providers = _create_session_for_backend(
            model_path,
            "cpu",
            device_id=device_id,
            cpu_single_thread=cpu_single_thread,
        )
        warning_msg = _format_fallback_warning(requested, "cpu", str(exc), model_label)
        _emit_warning_once((requested, "cpu", str(exc)), warning_msg)
        return session, _make_profile(
            requested_device=requested,
            active_backend="cpu",
            active_providers=active_providers,
            device_id=device_id,
            fallback_occurred=True,
            warning_message=warning_msg,
        )


def resolve_auto_backend_name() -> str:
    """Return the highest-priority currently available backend name for 'auto'."""
    availability = detect_available_backends()
    for candidate in AUTO_PRIORITY:
        if availability.get(candidate, {}).get("available"):
            return candidate
    return "cpu"


def format_device_diagnostics() -> str:
    """Return a human-readable diagnostics report for `--check-device`."""
    availability = detect_available_backends()
    auto_choice = resolve_auto_backend_name()
    ep_devices = _safe_ep_devices()
    lines = [
        "=== NDLOCR-Lite GPU / ONNX Runtime Diagnostics ===",
        f"ONNX Runtime version : {getattr(ort, '__version__', 'unknown')}",
        f"TableCenterNet (DeformConv, ORT>=1.25): {'SUPPORTED' if ort_supports_deform_conv() else 'UNSUPPORTED'}",
        f"Available providers  : {ort.get_available_providers()}",
        f"Registered plugins   : {_REGISTERED_PLUGINS or 'none'}",
    ]
    if _PLUGIN_ERRORS:
        lines.append(f"Plugin load errors   : {_PLUGIN_ERRORS}")
    if ep_devices:
        lines.append("Discovered EP devices:")
        for idx, dev in enumerate(ep_devices):
            hw = getattr(dev, "device", None)
            meta = getattr(dev, "ep_metadata", None) or {}
            hw_desc = (
                f"type={getattr(hw, 'type', '?')}, vendor={getattr(hw, 'vendor', '?')}, id={getattr(hw, 'device_id', '?')}"
                if hw is not None
                else "no-hw-info"
            )
            lines.append(f"  [{idx}] ep={dev.ep_name} | {hw_desc} | meta={meta}")
    lines.append("Backend summary:")
    for backend in AUTO_PRIORITY:
        info = availability[backend]
        status = "READY" if info["available"] else "unavailable"
        lines.append(f"  - {backend:9s}: {status:11s} ({info['mode']}) {info['detail']}")
    lines.append(f"Selected by '--device auto': {auto_choice} ({_BACKEND_DISPLAY_NAMES.get(auto_choice, auto_choice)})")
    return "\n".join(lines)
