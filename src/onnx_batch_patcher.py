"""Automatic dynamic-batch and mixed-precision FP16 ONNX patcher and local cache for PARSeq models."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import heapq
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import onnxruntime as ort


PATCHER_SCHEMA_VERSION = 1
SUPPORTED_PRECISIONS: Tuple[str, ...] = ("fp32", "fp16")
FP16_EXTRA_OP_BLOCK_LIST: Tuple[str, ...] = ("Softmax",)


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


def _file_blake2b(path: Path) -> str:
    digest = hashlib.blake2b(digest_size=16)
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


def _build_sidecar_path(
    src: Path,
    *,
    precision: str,
    dynamic_batch: bool,
    cache_dir: Optional[Path] = None,
) -> Path:
    resolved = str(src.resolve())
    path_hash = hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:12]
    batch_tag = "dynbatch" if dynamic_batch else "static"
    filename = (
        f"{src.stem}_{path_hash}_{batch_tag}_{precision}_v{PATCHER_SCHEMA_VERSION}.meta.json"
    )
    return optimized_cache_dir(cache_dir) / filename


def invalidate_parseq_cache(
    model_path: str | Path,
    *,
    precision: str,
    dynamic_batch: bool,
    cache_dir: Optional[Path] = None,
    cached_path: Optional[str | Path] = None,
) -> None:
    """Remove sidecar and cached ONNX artifacts for a PARSeq model without touching the source."""
    src = Path(model_path)
    try:
        sidecar = _build_sidecar_path(
            src,
            precision=precision,
            dynamic_batch=dynamic_batch,
            cache_dir=cache_dir,
        )
        sidecar.unlink(missing_ok=True)
    except OSError:
        pass
    if cached_path is not None:
        cached = Path(cached_path)
        try:
            if cached.resolve() != src.resolve():
                cached.unlink(missing_ok=True)
        except OSError:
            pass


def _inspect_sidecar_cache(
    src: Path,
    *,
    precision: str,
    dynamic_batch: bool,
    cache_dir: Optional[Path] = None,
) -> Tuple[Optional[Path], Optional[Path]]:
    """Verify sidecar metadata and BLAKE2b digests without running SHA-256 or onnx.load.

    Returns ``(hit_path, corrupted_cached_path)``:
    - ``(cached, None)`` on a clean hit.
    - ``(None, cached)`` when a sidecar entry exists for the verified source, but the cached
      artifact drifted in size, mtime, or BLAKE2b content digest (so the caller must discard
      and rebuild it rather than trusting protobuf structure alone).
    - ``(None, None)`` on miss, corrupt/legacy sidecar, or replaced source model.
    """
    try:
        resolved_src = str(src.resolve())
        src_stat = src.stat()
        sidecar = _build_sidecar_path(
            src, precision=precision, dynamic_batch=dynamic_batch, cache_dir=cache_dir
        )
        if not sidecar.is_file():
            return None, None
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        if not isinstance(meta, dict):
            return None, None
        if meta.get("schema_version") != PATCHER_SCHEMA_VERSION:
            return None, None
        if meta.get("src_path") != resolved_src:
            return None, None
        if meta.get("precision") != precision:
            return None, None
        if bool(meta.get("dynamic_batch")) != bool(dynamic_batch):
            return None, None
        if meta.get("src_size") != src_stat.st_size:
            return None, None
        if meta.get("src_mtime_ns") != src_stat.st_mtime_ns:
            return None, None
        expected_src_b2b = meta.get("src_blake2b")
        if (
            not isinstance(expected_src_b2b, str)
            or not expected_src_b2b
            or _file_blake2b(src) != expected_src_b2b
        ):
            return None, None
        cache_name = meta.get("cache_name")
        if (
            not isinstance(cache_name, str)
            or not cache_name
            or Path(cache_name).name != cache_name
        ):
            return None, None
        cached = sidecar.parent / cache_name
        if not cached.is_file():
            return None, None
        expected_cache_b2b = meta.get("cache_blake2b")
        if not isinstance(expected_cache_b2b, str) or not expected_cache_b2b:
            return None, None
        cache_stat = cached.stat()
        if (
            cache_stat.st_size <= 0
            or meta.get("cache_size") != cache_stat.st_size
            or meta.get("cache_mtime_ns") != cache_stat.st_mtime_ns
            or _file_blake2b(cached) != expected_cache_b2b
        ):
            return None, cached
        return cached, None
    except Exception:
        return None, None


def _try_sidecar_cache_hit(
    src: Path,
    *,
    precision: str,
    dynamic_batch: bool,
    cache_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Fast-path cache lookup using sidecar stat and BLAKE2b metadata."""
    hit, _ = _inspect_sidecar_cache(
        src,
        precision=precision,
        dynamic_batch=dynamic_batch,
        cache_dir=cache_dir,
    )
    return hit


def _write_sidecar_metadata(
    src: Path,
    cached: Path,
    *,
    src_hash: str,
    precision: str,
    dynamic_batch: bool,
    cache_dir: Optional[Path] = None,
) -> None:
    tmp_meta: Optional[Path] = None
    try:
        src_stat = src.stat()
        cache_stat = cached.stat()
        src_blake2b = _file_blake2b(src)
        cache_blake2b = _file_blake2b(cached)
        sidecar = _build_sidecar_path(
            src, precision=precision, dynamic_batch=dynamic_batch, cache_dir=cache_dir
        )
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": PATCHER_SCHEMA_VERSION,
            "src_path": str(src.resolve()),
            "src_size": int(src_stat.st_size),
            "src_mtime_ns": int(src_stat.st_mtime_ns),
            "src_hash": src_hash,
            "src_blake2b": src_blake2b,
            "cache_name": cached.name,
            "cache_size": int(cache_stat.st_size),
            "cache_mtime_ns": int(cache_stat.st_mtime_ns),
            "cache_blake2b": cache_blake2b,
            "dynamic_batch": bool(dynamic_batch),
            "precision": precision,
        }
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=sidecar.parent,
            prefix=f"{sidecar.stem}_",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_meta = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False)
        tmp_meta.replace(sidecar)
        tmp_meta = None
    except OSError:
        pass
    finally:
        if tmp_meta is not None:
            try:
                tmp_meta.unlink(missing_ok=True)
            except OSError:
                pass


def _model_has_dynamic_batch(model) -> bool:
    if not model.graph.input:
        return False
    dims = model.graph.input[0].type.tensor_type.shape.dim
    if not dims:
        return False
    dim0 = dims[0]
    return bool(dim0.dim_param) or dim0.dim_value == 0


def _model_has_fp16_tensors(model) -> bool:
    onnx, _, _ = _import_onnx()
    fp16_type = onnx.TensorProto.FLOAT16
    if any(init.data_type == fp16_type for init in model.graph.initializer):
        return True
    for node in model.graph.node:
        if node.op_type == "Cast" and any(
            attr.name == "to" and attr.i == fp16_type for attr in node.attribute
        ):
            return True
        if node.op_type == "Constant" and any(
            attr.name == "value" and attr.t.data_type == fp16_type for attr in node.attribute
        ):
            return True
    return False


def _model_output_is_fp32(model) -> bool:
    onnx, _, _ = _import_onnx()
    if not model.graph.output:
        return False
    return model.graph.output[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT


def is_dynamic_batch_model(model_path: str | Path) -> bool:
    """Return True if the ONNX model's first input already has a dynamic batch dimension."""
    path = Path(model_path)
    try:
        onnx, _, _ = _import_onnx()
        model = onnx.load(str(path), load_external_data=False)
        return _model_has_dynamic_batch(model)
    except Exception:
        return False


def is_valid_cached_model(
    model_path: str | Path,
    *,
    require_dynamic_batch: bool = True,
    expected_precision: str = "fp32",
) -> bool:
    """Validate a cached ONNX model's integrity, batch mode, and precision contract."""
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
        if require_dynamic_batch and not _model_has_dynamic_batch(model):
            return False
        if expected_precision == "fp16":
            if not (_model_has_fp16_tensors(model) and _model_output_is_fp32(model)):
                return False
        return True
    except Exception:
        return False


def is_valid_dynamic_batch_cache(model_path: str | Path) -> bool:
    """Return True if a cached dynamic-batch FP32 model can be parsed and is actually dynamic.

    Cache files can be truncated by an interrupted copy, a full disk, or an
    antivirus scanner. A corrupted cache must never be handed to ONNX Runtime
    (it raises ``InvalidProtobuf``/``INVALID_GRAPH`` and aborts OCR), so the
    caller drops and rebuilds it instead.
    """
    return is_valid_cached_model(
        model_path,
        require_dynamic_batch=True,
        expected_precision="fp32",
    )


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


def _cast_to_attr(node) -> Optional[int]:
    for attr in node.attribute:
        if attr.name == "to":
            return int(attr.i)
    return None


def _toposort_graph_nodes(nodes: Sequence) -> List:
    """Deterministically topologically sort ONNX nodes using Kahn's algorithm."""
    prod: Dict[str, int] = {}
    for idx, node in enumerate(nodes):
        for out_name in node.output:
            if out_name:
                prod[out_name] = idx

    indeg = [0] * len(nodes)
    adj: List[List[int]] = [[] for _ in range(len(nodes))]
    for idx, node in enumerate(nodes):
        deps = {
            prod[inp]
            for inp in node.input
            if inp and inp in prod and prod[inp] != idx
        }
        indeg[idx] = len(deps)
        for dep_idx in deps:
            adj[dep_idx].append(idx)

    heap = [idx for idx, deg in enumerate(indeg) if deg == 0]
    heapq.heapify(heap)
    ordered = []
    while heap:
        u = heapq.heappop(heap)
        ordered.append(nodes[u])
        for v in adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                heapq.heappush(heap, v)

    if len(ordered) != len(nodes):
        raise ValueError(
            f"Cycle detected in ONNX graph after FP16 conversion ({len(ordered)}/{len(nodes)} sorted)."
        )
    return ordered


def _find_attention_score_nodes(model) -> List[str]:
    """Identify attention-score MatMul and masking nodes feeding Softmax to keep in FP32."""
    prod = {out: node for node in model.graph.node for out in node.output if out}
    blocked: set[str] = set()
    walk_ops = {"Add", "Where", "Reshape", "Transpose", "Cast"}
    for node in model.graph.node:
        if node.op_type != "Softmax" or not node.input:
            continue
        stack = [node.input[0]]
        seen: set[str] = set()
        while stack:
            tensor_name = stack.pop()
            if not tensor_name or tensor_name in seen or tensor_name not in prod:
                continue
            seen.add(tensor_name)
            p_node = prod[tensor_name]
            if p_node.op_type == "MatMul":
                if p_node.name:
                    blocked.add(p_node.name)
            elif p_node.op_type in walk_ops:
                if p_node.name:
                    blocked.add(p_node.name)
                for inp in p_node.input:
                    stack.append(inp)
    return sorted(blocked)


def convert_model_to_fp16(
    model,
    *,
    extra_op_block_list: Sequence[str] = FP16_EXTRA_OP_BLOCK_LIST,
    node_block_list: Optional[Sequence[str]] = None,
    protect_attention_scores: bool = True,
    keep_io_types: bool = True,
):
    """Convert an ONNX ModelProto to mixed-precision FP16 with FP32 IO and cleanup passes.

    Uses ``onnxruntime.transformers.float16.convert_float_to_float16`` and fixes its
    duplicate ``*_cast_to_fp32`` output bug, collapses redundant ``fp32 -> fp16 -> fp32``
    bridges between adjacent FP32-island ops, deduplicates ``value_info``, and
    topologically sorts the resulting graph nodes.
    """
    onnx, _, _ = _import_onnx()
    from onnxruntime.transformers import float16 as ort_fp16

    op_block_list = sorted(set(ort_fp16.DEFAULT_OP_BLOCK_LIST) | set(extra_op_block_list))
    combined_node_block: set[str] = set(node_block_list or ())
    if protect_attention_scores:
        combined_node_block.update(_find_attention_score_nodes(model))
    converted = ort_fp16.convert_float_to_float16(
        model,
        keep_io_types=keep_io_types,
        op_block_list=op_block_list,
        node_block_list=sorted(combined_node_block) if combined_node_block else None,
    )
    graph = converted.graph

    # 1. Deduplicate Cast nodes with identical output names generated by ort_fp16.
    seen_outputs: set[str] = set()
    cast_sigs: Dict[str, Tuple[str, Optional[int]]] = {}
    dedup_nodes = []
    for node in graph.node:
        if len(node.output) == 1 and node.output[0] in seen_outputs:
            out_name = node.output[0]
            sig = (node.input[0], _cast_to_attr(node)) if len(node.input) == 1 else None
            if node.op_type == "Cast" and sig is not None and cast_sigs.get(out_name) == sig:
                continue
            raise ValueError(
                f"Duplicate non-identical node output '{out_name}' produced during FP16 conversion."
            )
        for out_name in node.output:
            if out_name:
                seen_outputs.add(out_name)
        if node.op_type == "Cast" and len(node.input) == 1 and len(node.output) == 1:
            cast_sigs[node.output[0]] = (node.input[0], _cast_to_attr(node))
        dedup_nodes.append(node)

    # 2. Collapse redundant fp32 -> fp16 -> fp32 bridges between adjacent FP32 nodes.
    if not _graph_has_subgraphs(graph):
        graph_outputs = {out.name for out in graph.output if out.name}
        fp16_type = onnx.TensorProto.FLOAT16
        fp32_type = onnx.TensorProto.FLOAT

        fp16_casts: Dict[str, Tuple[int, str]] = {}
        for idx, node in enumerate(dedup_nodes):
            if (
                node.op_type == "Cast"
                and node.name.endswith("_cast_to_fp16_node")
                and len(node.input) == 1
                and len(node.output) == 1
                and _cast_to_attr(node) == fp16_type
            ):
                fp16_casts[node.output[0]] = (idx, node.input[0])

        fp32_bridge: Dict[str, str] = {}
        drop_indices: set[int] = set()
        for idx, node in enumerate(dedup_nodes):
            if (
                node.op_type == "Cast"
                and len(node.input) == 1
                and len(node.output) == 1
                and _cast_to_attr(node) == fp32_type
                and node.input[0] in fp16_casts
                and node.output[0] not in graph_outputs
            ):
                _, src_fp32 = fp16_casts[node.input[0]]
                fp32_bridge[node.output[0]] = src_fp32
                drop_indices.add(idx)

        if fp32_bridge:
            surviving = []
            for idx, node in enumerate(dedup_nodes):
                if idx in drop_indices:
                    continue
                for i, inp_name in enumerate(node.input):
                    if inp_name in fp32_bridge:
                        node.input[i] = fp32_bridge[inp_name]
                surviving.append(node)

            used_inputs = set(graph_outputs)
            for node in surviving:
                used_inputs.update(inp for inp in node.input if inp)

            dedup_nodes = [
                node
                for node in surviving
                if not (
                    node.op_type == "Cast"
                    and node.name.endswith("_cast_to_fp16_node")
                    and len(node.output) == 1
                    and node.output[0] not in used_inputs
                )
            ]

    # 3. Topologically sort nodes.
    ordered_nodes = _toposort_graph_nodes(dedup_nodes)
    del graph.node[:]
    graph.node.extend(ordered_nodes)

    # 4. Deduplicate value_info and drop orphaned entries for removed Cast nodes.
    valid_names = {inp.name for inp in graph.input if inp.name}
    valid_names.update(out.name for out in graph.output if out.name)
    valid_names.update(init.name for init in graph.initializer if init.name)
    for node in ordered_nodes:
        valid_names.update(out_name for out_name in node.output if out_name)

    seen_vi: set[str] = set()
    kept_vi = []
    for vi in graph.value_info:
        if vi.name and vi.name in valid_names and vi.name not in seen_vi:
            seen_vi.add(vi.name)
            kept_vi.append(vi)
    del graph.value_info[:]
    graph.value_info.extend(kept_vi)

    return converted


def _rewrite_graph_dynamic_batch(model) -> None:
    """Rewrite static batch=1 PARSeq GraphProto in-place to support dynamic batch sizes."""
    onnx, helper, numpy_helper = _import_onnx()
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


def patch_onnx_dynamic_batch(
    src_path: str | Path,
    dst_path: str | Path,
    precision: str = "fp32",
    *,
    dynamic_batch: bool = True,
    fp16_extra_op_block_list: Sequence[str] = FP16_EXTRA_OP_BLOCK_LIST,
    fp16_protect_attention_scores: bool = True,
) -> None:
    """Patch PARSeq ONNX graph for dynamic batch and/or mixed-precision FP16."""
    if precision not in SUPPORTED_PRECISIONS:
        raise ValueError(
            f"Unsupported precision '{precision}'. Supported: {SUPPORTED_PRECISIONS}"
        )

    onnx, _, _ = _import_onnx()
    src = Path(src_path)
    dst = Path(dst_path)

    model = onnx.load(str(src))
    if dynamic_batch and not _model_has_dynamic_batch(model):
        _rewrite_graph_dynamic_batch(model)
        _prune_unused_constants(model)

    if precision == "fp16":
        model = convert_model_to_fp16(
            model,
            extra_op_block_list=fp16_extra_op_block_list,
            protect_attention_scores=fp16_protect_attention_scores,
            keep_io_types=True,
        )
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
    precision: str = "fp32",
    dynamic_batch: bool = True,
) -> float:
    """Verify patched PARSeq ONNX matches original outputs within precision tolerances."""
    opt_orig = ort.SessionOptions()
    opt_patched = ort.SessionOptions()
    # Dynamic-batch unrolled decoders trigger O(N^2) reuse planning in ORT; disable for patched model.
    opt_patched.enable_mem_reuse = not dynamic_batch
    if precision == "fp16":
        # Suppress noisy CPU EP constant-folding warnings on FP16 Gemm nodes during verification.
        opt_patched.log_severity_level = 3
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

    if precision == "fp16":
        out_meta = patched_sess.get_outputs()
        if out_meta and str(out_meta[0].type) != "tensor(float)":
            raise ValueError(
                f"Expected FP32 output logits for mixed-precision FP16 model, got {out_meta[0].type}"
            )

    effective_bs = int(batch_size) if dynamic_batch else 1
    if precision == "fp16":
        # Smooth deterministic document-like gradients avoid random-noise argmax flips
        # across PARSeq's 40 autoregressive decoding steps on AVX2 vs AVX-512 CPUs.
        xs = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, None, :]
        ys = np.linspace(-1.0, 1.0, height, dtype=np.float32)[None, :, None]
        samples = np.empty((effective_bs, 3, height, width), dtype=np.float32)
        for i in range(effective_bs):
            base = 0.85 - 0.15 * np.cos((i + 1) * np.pi * xs) * np.cos(np.pi * ys)
            samples[i, 0] = base
            samples[i, 1] = base - 0.05 * (i % 2)
            samples[i, 2] = base - 0.05 * ((i + 1) % 2)
        samples = np.clip(samples, -1.0, 1.0)
    else:
        rng = np.random.default_rng(12345)
        samples = rng.uniform(-1.0, 1.0, size=(effective_bs, 3, height, width)).astype(np.float32)

    orig_outputs = np.concatenate(
        [orig_sess.run(None, {in_name: samples[i:i + 1]})[0] for i in range(effective_bs)],
        axis=0,
    )
    patched_in_name = patched_sess.get_inputs()[0].name
    patched_outputs = patched_sess.run(None, {patched_in_name: samples.astype(dtype)})[0].astype(np.float32)

    if orig_outputs.shape != patched_outputs.shape:
        raise ValueError(
            f"Output shape mismatch: orig={orig_outputs.shape}, patched={patched_outputs.shape}"
        )
    if not np.isfinite(orig_outputs).all() or not np.isfinite(patched_outputs).all():
        raise ValueError("Non-finite values in batch=1 or patched outputs.")

    max_abs_diff = float(np.max(np.abs(orig_outputs - patched_outputs)))

    if precision == "fp16":
        if dynamic_batch and effective_bs > 1:
            patched_b1 = np.concatenate(
                [
                    patched_sess.run(None, {patched_in_name: samples[i:i + 1].astype(dtype)})[0].astype(np.float32)
                    for i in range(effective_bs)
                ],
                axis=0,
            )
            self_diff = float(np.max(np.abs(patched_b1 - patched_outputs)))
            if not np.isfinite(self_diff) or self_diff > 0.1:
                raise ValueError(
                    f"Dynamic-batch FP16 self-consistency diff {self_diff:.4e} exceeded 1e-1."
                )

        fp16_atol = atol if atol != 2e-2 else 2.0
        if not np.isfinite(max_abs_diff) or max_abs_diff > fp16_atol:
            raise ValueError(
                f"FP16 max abs diff {max_abs_diff:.4e} exceeded threshold {fp16_atol:.4e}."
            )
        orig_flat = orig_outputs.reshape(-1).astype(np.float64)
        patched_flat = patched_outputs.reshape(-1).astype(np.float64)
        denom = float(np.linalg.norm(orig_flat) * np.linalg.norm(patched_flat))
        cos_sim = float(np.dot(orig_flat, patched_flat) / denom) if denom > 0.0 else 0.0
        if not np.isfinite(cos_sim) or cos_sim < 0.99:
            raise ValueError(
                f"FP16 output cosine similarity {cos_sim:.4f} fell below 0.99."
            )
        return max_abs_diff

    orig_argmax = np.argmax(orig_outputs, axis=-1)
    patched_argmax = np.argmax(patched_outputs, axis=-1)
    if not np.array_equal(orig_argmax, patched_argmax):
        raise ValueError("Argmax mismatch between original batch=1 and patched dynamic-batch model.")

    if not np.isfinite(max_abs_diff) or max_abs_diff > atol:
        raise ValueError(
            f"Max abs diff {max_abs_diff:.4e} exceeded threshold {atol:.4e}."
        )
    return max_abs_diff


def prepare_parseq_model(
    model_path: str | Path,
    *,
    enable_dynamic_batch: bool = True,
    precision: str = "fp32",
    cache_dir: Optional[Path] = None,
    verify: bool = True,
) -> PatchedModelResult:
    """Prepare and cache a dynamic-batch and/or FP16 PARSeq ONNX model, falling back cleanly."""
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

    if not enable_dynamic_batch and precision == "fp32":
        return PatchedModelResult(
            model_path=src,
            dynamic_batch=is_dynamic_batch_model(src),
            precision="fp32",
            from_cache=False,
        )

    sidecar_hit, corrupted_cached = _inspect_sidecar_cache(
        src,
        precision=precision,
        dynamic_batch=enable_dynamic_batch,
        cache_dir=cache_dir,
    )
    if sidecar_hit is not None:
        return PatchedModelResult(
            model_path=sidecar_hit,
            dynamic_batch=bool(enable_dynamic_batch),
            precision=precision,
            from_cache=True,
        )
    if corrupted_cached is not None:
        print(
            f"[WARN] Corrupted dynamic-batch cache detected; rebuilding: {corrupted_cached.name}",
            file=sys.stderr,
        )
        invalidate_parseq_cache(
            src,
            precision=precision,
            dynamic_batch=enable_dynamic_batch,
            cache_dir=cache_dir,
            cached_path=corrupted_cached,
        )

    if precision == "fp32" and is_dynamic_batch_model(src):
        return PatchedModelResult(
            model_path=src,
            dynamic_batch=True,
            precision="fp32",
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
            precision="fp32",
            from_cache=False,
            fallback_reason=reason,
        )

    try:
        src_hash = _file_sha256(src)
        cached = build_cache_path(
            src,
            src_hash,
            precision=precision,
            dynamic_batch=enable_dynamic_batch,
            cache_dir=cache_dir,
        )
        if cached.is_file():
            if is_valid_cached_model(
                cached,
                require_dynamic_batch=enable_dynamic_batch,
                expected_precision=precision,
            ):
                _write_sidecar_metadata(
                    src,
                    cached,
                    src_hash=src_hash,
                    precision=precision,
                    dynamic_batch=enable_dynamic_batch,
                    cache_dir=cache_dir,
                )
                return PatchedModelResult(
                    model_path=cached,
                    dynamic_batch=bool(enable_dynamic_batch),
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
            precision="fp32",
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

        patch_onnx_dynamic_batch(
            src,
            tmp_path,
            precision=precision,
            dynamic_batch=enable_dynamic_batch,
        )
        if verify:
            verify_dynamic_batch_equivalence(
                src,
                tmp_path,
                precision=precision,
                dynamic_batch=enable_dynamic_batch,
            )
        tmp_path.replace(cached)
        _write_sidecar_metadata(
            src,
            cached,
            src_hash=src_hash,
            precision=precision,
            dynamic_batch=enable_dynamic_batch,
            cache_dir=cache_dir,
        )
        return PatchedModelResult(
            model_path=cached,
            dynamic_batch=bool(enable_dynamic_batch),
            precision=precision,
            from_cache=False,
        )
    except Exception as exc:
        reason = f"Dynamic batch patch skipped for {src.name}: {exc}"
        print(f"[WARN] {reason}", file=sys.stderr)
        return PatchedModelResult(
            model_path=src,
            dynamic_batch=is_dynamic_batch_model(src),
            precision="fp32",
            from_cache=False,
            fallback_reason=reason,
        )
    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
