"""Automatic dynamic-batch ONNX graph patcher and local cache for PARSeq models."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import sys
import tempfile
from typing import Dict, List, Optional, Tuple

import numpy as np
import onnxruntime as ort


PATCHER_SCHEMA_VERSION = 1
SUPPORTED_PRECISIONS: Tuple[str, ...] = ("fp32",)


@dataclass
class PatchedModelResult:
    """Result returned by :func:`prepare_parseq_model`."""

    model_path: Path
    dynamic_batch: bool
    precision: str
    from_cache: bool
    fallback_reason: Optional[str] = None


def _import_onnx():
    import onnx
    from onnx import helper, numpy_helper

    return onnx, helper, numpy_helper


def optimized_cache_dir(cache_dir: Optional[Path] = None) -> Path:
    """Return directory for cached dynamic-batch ONNX models."""
    if cache_dir is not None:
        return Path(cache_dir)
    env_override = os.environ.get("NDLOCR_CACHE_DIR")
    if env_override:
        return Path(env_override) / "optimized"
    from model_download import model_cache_dir

    return model_cache_dir() / "optimized"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_cache_path(
    model_path: Path,
    src_hash: str,
    precision: str = "fp32",
    dynamic_batch: bool = True,
    cache_dir: Optional[Path] = None,
) -> Path:
    """Build deterministic cache path including hash, batch mode, precision, and schema version."""
    batch_tag = "dynbatch" if dynamic_batch else "static"
    filename = (
        f"{model_path.stem}_{src_hash[:12]}_{batch_tag}_{precision}_v{PATCHER_SCHEMA_VERSION}.onnx"
    )
    return optimized_cache_dir(cache_dir) / filename


def is_dynamic_batch_model(model_path: str | Path) -> bool:
    """Return True if the ONNX model's first input already has a dynamic batch dimension."""
    path = Path(model_path)
    try:
        onnx, _, _ = _import_onnx()
        model = onnx.load(str(path), load_external_data=False)
        if not model.graph.input:
            return False
        dims = model.graph.input[0].type.tensor_type.shape.dim
        if not dims:
            return False
        dim0 = dims[0]
        return bool(dim0.dim_param) or dim0.dim_value == 0
    except Exception:
        return False


def is_valid_dynamic_batch_cache(model_path: str | Path) -> bool:
    """Return True if a cached dynamic-batch model can be parsed and is actually dynamic.

    Cache files can be truncated by an interrupted copy, a full disk, or an
    antivirus scanner. A corrupted cache must never be handed to ONNX Runtime
    (it raises ``InvalidProtobuf``/``INVALID_GRAPH`` and aborts OCR), so the
    caller drops and rebuilds it instead.
    """
    try:
        path = Path(model_path)
        if not path.is_file() or path.stat().st_size == 0:
            return False
        try:
            onnx, _, _ = _import_onnx()
        except ImportError:
            # Without the onnx package we cannot validate; let ONNX Runtime try.
            return True
        model = onnx.load(str(path), load_external_data=False)
        # A cache cut at the trailing opset_import field still parses as protobuf,
        # but ONNX Runtime rejects it ("Missing opset in the model"); reject here.
        if not model.opset_import or not model.graph.node:
            return False
        if not model.graph.input:
            return False
        dims = model.graph.input[0].type.tensor_type.shape.dim
        if not dims:
            return False
        dim0 = dims[0]
        return bool(dim0.dim_param) or dim0.dim_value == 0
    except Exception:
        return False


def _graph_has_subgraphs(graph) -> bool:
    """Return True if any node carries a control-flow subgraph attribute."""
    graph_attr_type = 5  # onnx.AttributeProto.GRAPH
    for node in graph.node:
        for attribute in node.attribute:
            if attribute.type == graph_attr_type:
                return True
    return False


def _prune_unused_constants(model) -> None:
    graph = model.graph
    if _graph_has_subgraphs(graph):
        # Outer-scope initializers may be referenced only from inside an
        # If/Loop/Scan body; without a recursive liveness walk, keep them.
        return
    used = {output.name for output in graph.output if output.name}
    used.update(inp.name for inp in graph.input if inp.name)
    for node in reversed(list(graph.node)):
        if node.op_type == "Constant" and node.output and node.output[0] not in used:
            continue
        used.update(inp for inp in node.input if inp)

    kept_nodes = [
        node for node in graph.node
        if not (node.op_type == "Constant" and node.output and node.output[0] not in used)
    ]
    del graph.node[:]
    graph.node.extend(kept_nodes)

    kept_inits = [init for init in graph.initializer if init.name in used]
    del graph.initializer[:]
    graph.initializer.extend(kept_inits)


def patch_onnx_dynamic_batch(
    src_path: str | Path,
    dst_path: str | Path,
    precision: str = "fp32",
) -> None:
    """Rewrite static batch=1 PARSeq ONNX graph to support dynamic batch sizes."""
    if precision not in SUPPORTED_PRECISIONS:
        raise ValueError(
            f"Unsupported precision '{precision}'. Supported: {SUPPORTED_PRECISIONS}"
        )

    onnx, helper, numpy_helper = _import_onnx()
    src = Path(src_path)
    dst = Path(dst_path)

    model = onnx.load(str(src))
    graph = model.graph

    if not graph.input or not graph.output:
        raise ValueError("ONNX model must define at least one input and output.")

    inp = graph.input[0]
    inp_dims = inp.type.tensor_type.shape.dim
    if len(inp_dims) != 4 or (inp_dims[0].dim_value not in (0, 1) and not inp_dims[0].dim_param):
        raise ValueError(
            f"Expected batch=1 NCHW input tensor, got {[d.dim_param or d.dim_value for d in inp_dims]}"
        )

    out = graph.output[0]
    out_dims = out.type.tensor_type.shape.dim
    if len(out_dims) != 3 or (out_dims[0].dim_value not in (0, 1) and not out_dims[0].dim_param):
        raise ValueError(
            f"Expected batch=1 3D output tensor, got {[d.dim_param or d.dim_value for d in out_dims]}"
        )

    init_map: Dict[str, np.ndarray] = {
        init.name: numpy_helper.to_array(init) for init in graph.initializer
    }
    const_val: Dict[str, np.ndarray] = {}
    for node in graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value":
                    const_val[node.output[0]] = numpy_helper.to_array(attr.t)
    all_consts: Dict[str, np.ndarray] = {**init_map, **const_val}
    producer = {output: node for node in graph.node for output in node.output}

    # Detect model hidden dimension and number of decoder attention heads from graph constants.
    d_model = 192
    n_heads = 6
    head_dim = 32
    for name, arr in all_consts.items():
        if "text_embed" in name and arr.ndim == 2 and arr.shape[0] > 1000:
            d_model = int(arr.shape[1])
            break
    for node in graph.node:
        if node.op_type == "Reshape" and "/self_attn" in node.name and node.name.endswith("/Reshape_3"):
            sh = node.input[1]
            if sh in all_consts:
                vals = all_consts[sh].tolist()
                if len(vals) == 3 and vals[1] * vals[2] == d_model:
                    n_heads = int(vals[1])
                    head_dim = int(vals[2])
                    break

    inp.type.tensor_type.shape.dim[0].ClearField("dim_value")
    inp.type.tensor_type.shape.dim[0].dim_param = "batch_size"
    out.type.tensor_type.shape.dim[0].ClearField("dim_value")
    out.type.tensor_type.shape.dim[0].dim_param = "batch_size"
    del graph.value_info[:]

    prefix_nodes = []

    def add_const(name: str, arr: np.ndarray) -> str:
        tensor = numpy_helper.from_array(arr, name=f"{name}_val")
        prefix_nodes.append(
            helper.make_node("Constant", inputs=[], outputs=[name], value=tensor, name=f"{name}_node")
        )
        return name

    c_0 = add_const("_dyn_c0", np.array([0], dtype=np.int64))
    c_1 = add_const("_dyn_c1", np.array([1], dtype=np.int64))
    c_1_1 = add_const("_dyn_c1_1", np.array([1, 1], dtype=np.int64))
    c_0_scalar = add_const("_dyn_s0", np.array(0, dtype=np.int64))
    c_1_scalar = add_const("_dyn_s1", np.array(1, dtype=np.int64))
    c_sq0_axes = add_const("_dyn_sq0_axes", np.array([0], dtype=np.int64))
    c_sq1_axes = add_const("_dyn_sq1_axes", np.array([1], dtype=np.int64))

    prefix_nodes.extend([
        helper.make_node("Shape", [inp.name], ["_dyn_in_shape"], name="_dyn_shape"),
        helper.make_node("Slice", ["_dyn_in_shape", c_0, c_1, c_0], ["_dyn_B1d"], name="_dyn_B1d_slice"),
        helper.make_node("Squeeze", ["_dyn_B1d", c_sq0_axes], ["_dyn_B_scalar"], name="_dyn_B_scalar_sq"),
        helper.make_node("Range", [c_0_scalar, "_dyn_B_scalar", c_1_scalar], ["_dyn_B_range"], name="_dyn_B_range_node"),
        helper.make_node("Unsqueeze", ["_dyn_B_range", c_1], ["_dyn_B_col"], name="_dyn_B_col_unsq"),
        helper.make_node("Concat", ["_dyn_B1d", c_1], ["_dyn_B_1_shape"], axis=0, name="_dyn_B_1_concat"),
        helper.make_node("Concat", ["_dyn_B1d", c_1_1], ["_dyn_B_1_1_shape"], axis=0, name="_dyn_B_1_1_concat"),
    ])

    shape_2d_cache: Dict[int, str] = {}

    def get_b_t_shape(t_val: int) -> str:
        if t_val not in shape_2d_cache:
            c_t = add_const(f"_dyn_cT_{t_val}", np.array([t_val], dtype=np.int64))
            out_name = f"_dyn_shape_B_{t_val}"
            prefix_nodes.append(
                helper.make_node("Concat", ["_dyn_B1d", c_t], [out_name], axis=0, name=f"_dyn_concat_B_{t_val}")
            )
            shape_2d_cache[t_val] = out_name
        return shape_2d_cache[t_val]

    rshape_cache: Dict[Tuple[int, ...], str] = {}

    def make_shape_const(values: List[int]) -> str:
        key = tuple(int(v) for v in values)
        if key not in rshape_cache:
            rshape_cache[key] = add_const(
                f"_dyn_rshape_{len(rshape_cache)}",
                np.array(key, dtype=np.int64),
            )
        return rshape_cache[key]

    extra_nodes_before: Dict[int, List] = {}
    for idx, node in enumerate(graph.node):
        if node.op_type == "Reshape" and len(node.input) >= 2 and node.input[1] in all_consts:
            vals = [int(v) for v in all_consts[node.input[1]].tolist()]
            src_op = producer.get(node.input[0])
            src_type = src_op.op_type if src_op else ""

            if "/encoder/" in node.name and len(vals) >= 3 and vals[0] == 1:
                node.input[1] = make_shape_const([-1] + vals[1:])
            elif "/self_attn" in node.name or "/cross_attn" in node.name:
                if len(vals) == 3 and vals[1] == n_heads and vals[2] == head_dim:
                    node.input[1] = make_shape_const([vals[0], -1, vals[2]])
                elif len(vals) == 4 and vals[:3] == [1, 1, 1]:
                    node.input[1] = make_shape_const([-1] + vals[1:])
                elif len(vals) == 3 and vals[0] == n_heads and src_type == "Expand":
                    node.input[1] = make_shape_const([-1] + vals[1:])
                elif len(vals) == 2 and vals[1] == d_model:
                    node.input[1] = make_shape_const([-1, vals[1]])
                elif len(vals) == 3 and vals[1] == 1 and vals[2] == d_model and src_type == "Gemm":
                    node.input[1] = make_shape_const([vals[0], -1, vals[2]])
            elif node.name == "/Reshape" and vals == [1, 1]:
                node.input[1] = "_dyn_B_1_shape"

        elif (
            node.op_type == "Squeeze"
            and len(node.input) == 1
            and not any(attr.name == "axes" for attr in node.attribute)
        ):
            node.input.append(c_sq1_axes)

        elif node.op_type == "Expand" and len(node.input) >= 2:
            in0, in1 = node.input[0], node.input[1]
            if (
                in0 in all_consts
                and all_consts[in0].ndim == 3
                and all_consts[in0].shape[0] == 1
                and all_consts[in0].shape[2] == d_model
            ):
                node.input[1] = "_dyn_B_1_1_shape"
            elif (
                in0 in all_consts
                and in1 in all_consts
                and all_consts[in0].shape == (1,)
                and all_consts[in1].tolist() == [1]
            ):
                node.input[1] = "_dyn_B1d"
            elif (
                in0 in all_consts
                and all_consts[in0].shape == (1, 1)
                and all_consts[in0].tolist() == [[0]]
            ):
                node.op_type = "Identity"
                del node.input[:]
                node.input.append("_dyn_B_col")
                del node.attribute[:]
            elif (
                in0 in all_consts
                and all_consts[in0].shape == (1,)
                and in1 in producer
                and producer[in1].op_type == "Where"
            ):
                node.input[1] = "_dyn_B_1_shape"

        elif node.op_type == "Concat" and len(node.input) == 2:
            axis = next((attr.i for attr in node.attribute if attr.name == "axis"), 0)
            if axis == 0 and node.input[0] in all_consts and all_consts[node.input[0]].tolist() == [1, 1]:
                node.input[0] = "_dyn_B_1_shape"
            elif axis == 1 and node.input[0] in all_consts and all_consts[node.input[0]].shape == (1, 1):
                exp_out = f"_dyn_bos_exp_{idx}"
                extra_nodes_before.setdefault(idx, []).append(
                    helper.make_node(
                        "Expand",
                        [node.input[0], "_dyn_B_1_shape"],
                        [exp_out],
                        name=f"_dyn_bos_exp_node_{idx}",
                    )
                )
                node.input[0] = exp_out

        elif (
            node.op_type == "ScatterND"
            and node.input[0] in all_consts
            and all_consts[node.input[0]].ndim == 2
            and all_consts[node.input[0]].shape[0] == 1
        ):
            seq_len = int(all_consts[node.input[0]].shape[1])
            exp_data = f"_dyn_scatter_init_{idx}"
            extra_nodes_before.setdefault(idx, []).append(
                helper.make_node(
                    "Expand",
                    [node.input[0], get_b_t_shape(seq_len)],
                    [exp_data],
                    name=f"_dyn_scatter_init_node_{idx}",
                )
            )
            node.input[0] = exp_data

    new_nodes = list(prefix_nodes)
    for idx, node in enumerate(graph.node):
        if idx in extra_nodes_before:
            new_nodes.extend(extra_nodes_before[idx])
        new_nodes.append(node)
    del graph.node[:]
    graph.node.extend(new_nodes)

    _prune_unused_constants(model)
    dst.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(dst))
    onnx.checker.check_model(str(dst))


def verify_dynamic_batch_equivalence(
    orig_path: str | Path,
    patched_path: str | Path,
    *,
    atol: float = 2e-2,
    batch_size: int = 2,
    disable_ort_optimizations: bool = False,
) -> float:
    """Verify patched dynamic-batch ONNX matches original batch=1 outputs and argmax."""
    opt_orig = ort.SessionOptions()
    opt_patched = ort.SessionOptions()
    # Dynamic-batch unrolled decoders trigger O(N^2) reuse planning in ORT; disable for patched model.
    opt_patched.enable_mem_reuse = False
    if disable_ort_optimizations:
        opt_orig.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        opt_patched.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    else:
        opt_orig.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opt_patched.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    orig_sess = ort.InferenceSession(str(orig_path), opt_orig, providers=["CPUExecutionProvider"])
    patched_sess = ort.InferenceSession(str(patched_path), opt_patched, providers=["CPUExecutionProvider"])

    in_meta = orig_sess.get_inputs()[0]
    in_name = in_meta.name
    height, width = int(in_meta.shape[2]), int(in_meta.shape[3])
    dtype = np.float16 if "float16" in str(patched_sess.get_inputs()[0].type) else np.float32

    rng = np.random.default_rng(12345)
    samples = rng.uniform(-1.0, 1.0, size=(batch_size, 3, height, width)).astype(np.float32)

    orig_outputs = np.concatenate(
        [orig_sess.run(None, {in_name: samples[i:i + 1]})[0] for i in range(batch_size)],
        axis=0,
    )
    patched_in_name = patched_sess.get_inputs()[0].name
    patched_outputs = patched_sess.run(None, {patched_in_name: samples.astype(dtype)})[0].astype(np.float32)

    if orig_outputs.shape != patched_outputs.shape:
        raise ValueError(
            f"Output shape mismatch: orig={orig_outputs.shape}, patched={patched_outputs.shape}"
        )

    orig_argmax = np.argmax(orig_outputs, axis=-1)
    patched_argmax = np.argmax(patched_outputs, axis=-1)
    if not np.array_equal(orig_argmax, patched_argmax):
        raise ValueError("Argmax mismatch between original batch=1 and patched dynamic-batch model.")

    max_abs_diff = float(np.max(np.abs(orig_outputs - patched_outputs)))
    if not np.isfinite(max_abs_diff) or max_abs_diff > atol:
        raise ValueError(
            f"Max abs diff {max_abs_diff:.4e} exceeded threshold {atol:.4e}."
        )
    if not np.isfinite(orig_outputs).all() or not np.isfinite(patched_outputs).all():
        raise ValueError("Non-finite values in batch=1 or dynamic-batch outputs.")
    return max_abs_diff


def prepare_parseq_model(
    model_path: str | Path,
    *,
    enable_dynamic_batch: bool = True,
    precision: str = "fp32",
    cache_dir: Optional[Path] = None,
    verify: bool = True,
) -> PatchedModelResult:
    """Prepare and cache a dynamic-batch PARSeq ONNX model, falling back cleanly on failure."""
    src = Path(model_path)
    if precision not in SUPPORTED_PRECISIONS:
        reason = f"Unsupported precision '{precision}'; falling back to batch=1"
        print(f"[WARN] {reason}", file=sys.stderr)
        return PatchedModelResult(
            model_path=src,
            dynamic_batch=is_dynamic_batch_model(src),
            precision="fp32",
            from_cache=False,
            fallback_reason=reason,
        )

    if is_dynamic_batch_model(src):
        return PatchedModelResult(
            model_path=src,
            dynamic_batch=True,
            precision=precision,
            from_cache=False,
        )

    if not enable_dynamic_batch:
        return PatchedModelResult(
            model_path=src,
            dynamic_batch=False,
            precision=precision,
            from_cache=False,
        )

    try:
        _import_onnx()
    except ImportError as exc:
        reason = f"onnx package not available ({exc}); falling back to batch=1"
        print(f"[WARN] {reason}", file=sys.stderr)
        return PatchedModelResult(
            model_path=src,
            dynamic_batch=False,
            precision=precision,
            from_cache=False,
            fallback_reason=reason,
        )

    try:
        src_hash = _file_sha256(src)
        cached = build_cache_path(
            src, src_hash, precision=precision, dynamic_batch=True, cache_dir=cache_dir
        )
        if cached.is_file():
            if is_valid_dynamic_batch_cache(cached):
                return PatchedModelResult(
                    model_path=cached,
                    dynamic_batch=True,
                    precision=precision,
                    from_cache=True,
                )
            print(
                f"[WARN] Corrupted dynamic-batch cache detected; rebuilding: {cached.name}",
                file=sys.stderr,
            )
            try:
                cached.unlink()
            except OSError:
                pass
        cached.parent.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        reason = f"Dynamic batch cache unavailable for {src.name}: {exc}"
        print(f"[WARN] {reason}", file=sys.stderr)
        return PatchedModelResult(
            model_path=src,
            dynamic_batch=False,
            precision=precision,
            from_cache=False,
            fallback_reason=reason,
        )

    tmp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=cached.parent,
            prefix=f"{src.stem}_",
            suffix=".part.onnx",
            delete=False,
        ) as tmp_file:
            tmp_path = Path(tmp_file.name)

        patch_onnx_dynamic_batch(src, tmp_path, precision=precision)
        if verify:
            verify_dynamic_batch_equivalence(src, tmp_path)
        tmp_path.replace(cached)
        return PatchedModelResult(
            model_path=cached,
            dynamic_batch=True,
            precision=precision,
            from_cache=False,
        )
    except Exception as exc:
        reason = f"Dynamic batch patch skipped for {src.name}: {exc}"
        print(f"[WARN] {reason}", file=sys.stderr)
        return PatchedModelResult(
            model_path=src,
            dynamic_batch=False,
            precision=precision,
            from_cache=False,
            fallback_reason=reason,
        )
    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
