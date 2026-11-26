"""Equivalence tests for the vectorized AABB pre-filter in ``decode_official``.

The optimized decoder replaces the per-pair ``aabb_intersects`` call with a
vectorized overlap pre-filter over the loop-invariant candidate AABBs.  To pin
that optimization down, the pre-optimization algorithm is reproduced verbatim
below (``_reference_decode_official``) and compared against the shipped decoder
on deterministic Synthetic TableCenterNet outputs.
"""
from __future__ import annotations

from pathlib import Path
import sys
import time
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import tablecenternet as tc


def _reference_decode_official(
    hm_logits: np.ndarray,
    reg: np.ndarray,
    ct2cn: np.ndarray,
    cn2ct: np.ndarray,
    lc: np.ndarray,
    span: np.ndarray,
    *,
    center_k: int,
    corner_k: int,
    center_thresh: float,
    corner_thresh: float,
) -> tc.DecodedBatch:
    """Reference decoder: calls ``aabb_intersects`` for every valid pair."""
    heat = tc.sigmoid(hm_logits)

    center_scores, center_inds, _, _, center_polygons = tc.polygons_decode(
        heat[:, 0:1], ct2cn, reg, center_k
    )
    corner_scores, corner_inds, corner_xs, corner_ys, corner_polygons = tc.polygons_decode(
        heat[:, 1:2], cn2ct, reg, corner_k
    )

    h, w = heat.shape[2:]
    center_ys = center_inds // w
    center_xs = center_inds % w
    cell_spans = tc.gather_map(span, center_ys, center_xs)

    corner_ys_idx = corner_inds // w
    corner_xs_idx = corner_inds % w
    corner_logics = tc.gather_map(lc, corner_ys_idx, corner_xs_idx)

    corrected = center_polygons.copy()
    logic = np.zeros((len(center_scores), 4), dtype=np.int32)
    corner_counts = np.zeros((len(center_scores), 2), dtype=np.int32)

    valid_center_indices = np.flatnonzero(center_scores >= center_thresh)
    valid_corner_indices = np.flatnonzero(corner_scores >= corner_thresh)
    valid_corner_polys = corner_polygons[valid_corner_indices]

    for i in valid_center_indices:
        origin = center_polygons[i]
        fixed = corrected[i]
        lc_logic = [None, None, None, None]
        first_fixes = 0
        replacement_fixes = 0

        for local_j, j in enumerate(valid_corner_indices):
            candidate_poly = valid_corner_polys[local_j]
            if not tc.aabb_intersects(origin, candidate_poly):
                continue
            if not tc.any_corner_strictly_inside(origin, candidate_poly):
                continue

            cx = float(corner_xs[j])
            cy = float(corner_ys[j])
            ci = tc.nearest_corner_index(origin, cx, cy)

            origin_xy = origin[ci]
            current_xy = fixed[ci]
            candidate_xy = np.array([cx, cy], dtype=np.float32)

            if np.array_equal(current_xy, origin_xy):
                first_fixes += 1
                fixed[ci] = candidate_xy
                lc_logic[ci] = corner_logics[j].copy()
            else:
                old_d2 = float(np.sum((origin_xy - current_xy) ** 2))
                new_d2 = float(np.sum((origin_xy - candidate_xy) ** 2))
                if old_d2 >= new_d2:
                    replacement_fixes += 1
                    fixed[ci] = candidate_xy
                    lc_logic[ci] = corner_logics[j].copy()

        corner_counts[i] = [first_fixes, replacement_fixes]

        fallback_logic = tc.sample_lc_at_polygon(lc, fixed)
        for j in range(4):
            if lc_logic[j] is None:
                lc_logic[j] = fallback_logic[j]

        logic[i] = tc.get_logic_coords(
            np.asarray(lc_logic, dtype=np.float32), cell_spans[i]
        )

    scores = center_scores.copy()
    modified = False
    for i in range(len(scores)):
        if scores[i] < center_thresh:
            break
        if int(corner_counts[i].sum()) <= tc.CELL_MIN_OPTIMIZE_COUNT:
            scores[i] *= tc.CELL_DECAY
            modified = True

    if modified:
        order = np.argsort(scores)[::-1]
        corrected = corrected[order]
        scores = scores[order]
        logic = logic[order]
        corner_counts = corner_counts[order]

    return tc.DecodedBatch(
        cells=corrected,
        scores=scores,
        logic=logic,
        corner_counts=corner_counts,
    )


def _square_vectors(half: float) -> np.ndarray:
    """Corner offset vectors producing a TL, TR, BR, BL axis-aligned square."""
    half = float(half)
    return np.array([half, half, -half, half, -half, -half, half, -half], dtype=np.float32)


def _synthetic_batch(
    seed: int,
    size: int,
    center_points,
    corner_points,
    *,
    center_half: float = 6.0,
    corner_half: float = 2.0,
):
    """Deterministic hm/reg/ct2cn/cn2ct/lc/sp batch with known peak locations."""
    rng = np.random.default_rng(seed)
    hm = np.full((1, 2, size, size), -10.0, dtype=np.float32)
    for rank, (y, x) in enumerate(center_points):
        hm[0, 0, y, x] = np.float32(4.0 + rank * 1e-4)
    for rank, (y, x) in enumerate(corner_points):
        hm[0, 1, y, x] = np.float32(4.0 + rank * 1e-4)

    reg = np.zeros((1, 2, size, size), dtype=np.float32)
    ct2cn = np.broadcast_to(
        _square_vectors(center_half)[None, :, None, None], (1, 8, size, size)
    ).copy()
    cn2ct = np.broadcast_to(
        _square_vectors(corner_half)[None, :, None, None], (1, 8, size, size)
    ).copy()
    lc = rng.integers(1, 4, size=(1, 2, size, size)).astype(np.float32)
    span = rng.integers(1, 3, size=(1, 2, size, size)).astype(np.float32)
    return hm, reg, ct2cn, cn2ct, lc, span


_KWARGS = dict(
    center_k=3000,
    corner_k=5000,
    center_thresh=0.2,
    corner_thresh=0.3,
)


def _empty_candidate_batch():
    # Corner heatmap stays below corner_thresh: valid_corner_indices is empty.
    return _synthetic_batch(11, 24, [(6, 6), (6, 12), (12, 6)], [])


def _single_candidate_batch():
    # Exactly one corner above threshold; only the first center overlaps it.
    return _synthetic_batch(12, 24, [(6, 6), (6, 18), (18, 6)], [(5, 5)])


def _dense_candidate_batch():
    # 36 centers x 64 corners = 2304 evaluated pairs, hundreds of AABB hits.
    centers = [(8 + 8 * row, 8 + 8 * col) for row in range(6) for col in range(6)]
    corners = [(4 + 7 * row, 4 + 7 * col) for row in range(8) for col in range(8)]
    return _synthetic_batch(13, 64, centers, corners)


def _random_geometry_batch(seed: int, size: int = 32):
    """Seeded non-square polygons with sub-pixel offsets and random vectors."""
    rng = np.random.default_rng(seed)
    hm = np.full((1, 2, size, size), -10.0, dtype=np.float32)
    center_points = [(6 + 10 * row, 6 + 10 * col) for row in range(3) for col in range(3)]
    corner_points = [(4 + 8 * row, 5 + 10 * col) for row in range(4) for col in range(3)]
    for rank, (y, x) in enumerate(center_points):
        hm[0, 0, y, x] = np.float32(4.0 + rank * 1e-4)
    for rank, (y, x) in enumerate(corner_points):
        hm[0, 1, y, x] = np.float32(4.0 + rank * 1e-4)

    reg = rng.uniform(-1.5, 1.5, size=(1, 2, size, size)).astype(np.float32)
    ct2cn = rng.uniform(-14.0, 14.0, size=(1, 8, size, size)).astype(np.float32)
    cn2ct = rng.uniform(-8.0, 8.0, size=(1, 8, size, size)).astype(np.float32)
    lc = rng.uniform(0.5, 6.5, size=(1, 2, size, size)).astype(np.float32)
    span = rng.uniform(1.0, 4.0, size=(1, 2, size, size)).astype(np.float32)
    return hm, reg, ct2cn, cn2ct, lc, span


def _assert_decoded_equal(
    actual: tc.DecodedBatch, expected: tc.DecodedBatch, *, label: str = ""
) -> None:
    for name in ("cells", "scores", "logic", "corner_counts"):
        actual_arr = np.asarray(getattr(actual, name))
        expected_arr = np.asarray(getattr(expected, name))
        assert actual_arr.shape == expected_arr.shape, f"{label}{name}: shape mismatch"
        assert actual_arr.dtype == expected_arr.dtype, f"{label}{name}: dtype mismatch"
        assert np.array_equal(actual_arr, expected_arr), f"{label}{name}: values mismatch"


def test_empty_candidates_match_reference():
    batch = _empty_candidate_batch()
    expected = _reference_decode_official(*batch, **_KWARGS)
    actual = tc.decode_official(*batch, **_KWARGS)

    _assert_decoded_equal(actual, expected)
    # Sanity check that the empty-candidate path was actually exercised.
    assert int(actual.corner_counts.sum()) == 0


def test_empty_topk_corner_channel_matches_reference():
    # corner_k=0 yields corner_polygons shaped (0, 4, 2); reducing along axis=1
    # must keep corner_min/corner_max at (0, 2) and produce no candidates.
    batch = _empty_candidate_batch()
    kwargs = dict(_KWARGS, corner_k=0)
    expected = _reference_decode_official(*batch, **kwargs)
    actual = tc.decode_official(*batch, **kwargs)

    _assert_decoded_equal(actual, expected)
    assert int(actual.corner_counts.sum()) == 0


def test_single_candidate_matches_reference():
    batch = _single_candidate_batch()
    expected = _reference_decode_official(*batch, **_KWARGS)
    actual = tc.decode_official(*batch, **_KWARGS)

    _assert_decoded_equal(actual, expected)
    assert int(actual.corner_counts[:, 0].sum()) == 1
    assert int(actual.corner_counts[:, 1].sum()) == 0


def test_dense_candidates_match_reference():
    batch = _dense_candidate_batch()

    start = time.perf_counter()
    expected = _reference_decode_official(*batch, **_KWARGS)
    reference_seconds = time.perf_counter() - start

    start = time.perf_counter()
    actual = tc.decode_official(*batch, **_KWARGS)
    vectorized_seconds = time.perf_counter() - start

    _assert_decoded_equal(actual, expected)

    # Both first-fix and replacement branches must be reached.
    assert int(actual.corner_counts[:, 0].sum()) > 100
    assert int(actual.corner_counts[:, 1].sum()) > 0

    # Timing is reported only; asserting on wall-clock time would be flaky.
    print(
        "\ndense candidates: reference={:.1f} ms, vectorized={:.1f} ms "
        "(speedup {:.1f}x)".format(
            reference_seconds * 1e3,
            vectorized_seconds * 1e3,
            reference_seconds / vectorized_seconds,
        )
    )


def test_seeded_random_geometry_matches_reference():
    # Seeded, reproducible inputs: non-square polygons, sub-pixel offsets and
    # vectors that make both the first-fix and replacement branches fire.
    for seed in (21, 22, 23):
        batch = _random_geometry_batch(seed)
        expected = _reference_decode_official(*batch, **_KWARGS)
        actual = tc.decode_official(*batch, **_KWARGS)

        _assert_decoded_equal(actual, expected, label=f"seed={seed} ")
        assert int(actual.corner_counts.sum()) > 0


def test_vectorized_filter_does_not_call_aabb_intersects():
    batch = _single_candidate_batch()
    sentinel = AssertionError("aabb_intersects must not be called by decode_official")
    with mock.patch.object(tc, "aabb_intersects", side_effect=sentinel):
        actual = tc.decode_official(*batch, **_KWARGS)

    # The single-candidate path still relocates one corner through the
    # vectorized overlap index list.
    assert int(actual.corner_counts[:, 0].sum()) == 1
