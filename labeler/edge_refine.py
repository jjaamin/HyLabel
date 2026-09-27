"""Push a predicted contour out onto the image edge it should have landed on.

SAM tends to stop just inside an object, so the mask boundary sits a few pixels
in from the visible edge. Refinement walks outward along each boundary point's
normal, looking for a stronger edge within a search range, and moves the point
there — but only where the evidence is convincing. Points with no clear edge are
left undecided and filled in from the neighbours that did find one, so a weak
stretch of boundary follows the confident parts around it instead of being
pushed somewhere arbitrary.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

DEFAULT_SEARCH_RANGE = 20
MIN_SEARCH_RANGE = 1
MAX_SEARCH_RANGE = 40

# Points either side of i used to estimate the tangent. Wider than one point
# because a pixel-stepped contour's immediate neighbours only ever differ by a
# single step, which quantises the normal to eight directions.
_TANGENT_SPAN = 3

# Contours shorter than this have no reliable tangent or neighbourhood to
# interpolate from, so they are passed through untouched.
_MIN_CONTOUR = 12

# A candidate has to beat the response where the contour already sits by this
# much. Without it, points already on the edge drift outward on noise.
_MIN_GAIN = 1.5

# ...and reach this fraction of the strong-edge level measured on this contour.
# Taking the level from the contour itself is what makes the test adaptive: a
# low-contrast object sets a low bar, a crisp one a high bar.
_STRONG_PCTL = 80
_FLOOR_FRAC = 0.35

# A candidate further out is worth slightly less, so a decent near edge wins
# over a marginally stronger distant one.
_DIST_PENALTY = 0.35

# How far above its own ray's noise a peak has to stand, in robust sigmas. This
# is the test that keeps a featureless stretch from inventing an edge: measured
# on flat noise it drops the share of points accepted from 0.75 to 0.07, while
# leaving a real edge at 1.00.
_NOISE_SIGMAS = 4.0
_MAD_TO_SIGMA = 1.4826

# Guards the all-zero case, where every response ties at 0 and every ratio test
# passes trivially.
_EPS = 1e-6

# A real edge is found by a run of neighbouring points; noise is found by one
# point on its own. An accepted point needs at least _COHERENCE_MIN accepted
# points (itself included) in a window of _COHERENCE_WIN, or it is discarded as
# a speckle before anything is interpolated from it.
_COHERENCE_WIN = 5
_COHERENCE_MIN = 3

# Share of a contour that must have found an edge on its own before the result
# is interpolated right around the loop. Below it the few points that did agree
# are treated as local findings: they still move, but their offset falls off
# within _LOW_COVERAGE_SUPPORT points instead of dragging the whole boundary
# out. Scattered false positives in a textureless region are what this is for —
# they inflated a disk on flat noise by 38% when every gap was bridged.
_MIN_COVERAGE = 0.25
_LOW_COVERAGE_SUPPORT = 5.0

# Circular filters applied to the displacement field: the median kills lone
# points that locked onto something their neighbours did not see, the mean takes
# the stepping out of the result.
_MEDIAN_WIN = 5
_SMOOTH_WIN = 5

_BLUR_SIGMA = 1.2


def clamp_range(value: int) -> int:
    return max(MIN_SEARCH_RANGE, min(MAX_SEARCH_RANGE, int(value)))


def image_gradients(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(gx, gy) of a lightly blurred greyscale copy of image.

    Computed once per image by the caller and reused for every refinement — it
    costs far more than the contour walk that consumes it.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    blurred = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), _BLUR_SIGMA)
    gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    return gx, gy


# ── sampling ──────────────────────────────────────────────────────────────────

def _sample(img: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Bilinear sample of img at float coordinates, clamped to the image."""
    h, w = img.shape
    x = np.clip(xs, 0.0, w - 1.001)
    y = np.clip(ys, 0.0, h - 1.001)
    x0 = np.floor(x).astype(np.intp)
    y0 = np.floor(y).astype(np.intp)
    x1 = x0 + 1
    y1 = y0 + 1
    fx = (x - x0).astype(np.float32)
    fy = (y - y0).astype(np.float32)
    return (img[y0, x0] * (1 - fx) * (1 - fy)
            + img[y0, x1] * fx * (1 - fy)
            + img[y1, x0] * (1 - fx) * fy
            + img[y1, x1] * fx * fy)


# ── circular helpers ──────────────────────────────────────────────────────────

def _wrap_filter(values: np.ndarray, window: int, kind: str) -> np.ndarray:
    """Median or mean over a window that wraps around the closed contour."""
    n = len(values)
    if window < 3 or n < window:
        return values
    pad = window // 2
    ext = np.concatenate([values[-pad:], values, values[:pad]])
    view = np.lib.stride_tricks.sliding_window_view(ext, window)
    return np.median(view, axis=-1) if kind == "median" else view.mean(axis=-1)


def _interp_circular(values: np.ndarray) -> Optional[np.ndarray]:
    """Fill NaNs from the finite entries, treating the array as a closed loop.

    Returns None when nothing was decided — the caller then leaves the contour
    alone rather than inventing a boundary out of no evidence at all.
    """
    known = np.isfinite(values)
    if not known.any():
        return None
    if known.all():
        return values
    n = len(values)
    idx = np.arange(n)
    ki = idx[known]
    kv = values[known]
    # One period either side so a gap spanning the seam interpolates across it
    # instead of clamping to the first and last known points.
    ext_i = np.concatenate([ki - n, ki, ki + n])
    ext_v = np.concatenate([kv, kv, kv])
    return np.interp(idx, ext_i, ext_v)


def _dist_to_anchor(known: np.ndarray) -> np.ndarray:
    """Distance from each index to the nearest True, measured around the loop."""
    n = len(known)
    idx = np.arange(n)
    anchors = idx[known]
    if len(anchors) == 0:
        return np.full(n, np.inf)
    ext = np.concatenate([anchors - n, anchors, anchors + n])
    pos = np.searchsorted(ext, idx)
    left = ext[np.maximum(pos - 1, 0)]
    right = ext[np.minimum(pos, len(ext) - 1)]
    return np.minimum(np.abs(idx - left), np.abs(idx - right)).astype(np.float64)


# ── the walk ──────────────────────────────────────────────────────────────────

def _outward_normals(pts: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Unit normals pointing away from the mask, one per contour point.

    The perpendicular of the tangent gives the axis; which of the two ways along
    it leads out of the mask is settled by probing both sides and taking the
    majority. A closed contour has one consistent winding, so the vote is a
    property of the whole contour, and it comes out right for a hole too, where
    "away from the mask" points into the hole.
    """
    n = len(pts)
    span = min(_TANGENT_SPAN, max(1, n // 4))
    tangent = np.roll(pts, -span, axis=0) - np.roll(pts, span, axis=0)
    length = np.hypot(tangent[:, 0], tangent[:, 1])
    length[length == 0.0] = 1.0
    tx = tangent[:, 0] / length
    ty = tangent[:, 1] / length
    nx, ny = ty, -tx

    h, w = mask.shape
    probe = 2.0
    sx = np.clip(np.round(pts[:, 0] + nx * probe).astype(np.intp), 0, w - 1)
    sy = np.clip(np.round(pts[:, 1] + ny * probe).astype(np.intp), 0, h - 1)
    if (mask[sy, sx] == 0).mean() < 0.5:
        nx, ny = -nx, -ny
    return np.stack([nx, ny], axis=1)


def _refine_contour(pts: np.ndarray, mask: np.ndarray, gx: np.ndarray,
                    gy: np.ndarray, search: int) -> Tuple[np.ndarray, int, int]:
    """Move pts outward onto the strongest nearby edge.

    Returns (points, moved, interpolated) where moved counts the points that
    found an edge themselves and interpolated the ones that took their offset
    from their neighbours.
    """
    n = len(pts)
    normals = _outward_normals(pts, mask)
    steps = np.arange(0, search + 1, dtype=np.float32)

    px = pts[:, 0:1] + normals[:, 0:1] * steps[None, :]
    py = pts[:, 1:2] + normals[:, 1:2] * steps[None, :]
    # Response is the gradient projected on the normal, not its magnitude: we
    # are looking for an edge we cross on the way out, and an edge running
    # alongside the search ray should not attract the point.
    resp = np.abs(_sample(gx, px, py) * normals[:, 0:1]
                  + _sample(gy, px, py) * normals[:, 1:2])

    rows = np.arange(n)
    penalty = 1.0 - _DIST_PENALTY * (steps[1:] / float(search))
    best = np.argmax(resp[:, 1:] * penalty[None, :], axis=1) + 1
    peak = resp[rows, best]
    base = resp[:, 0]

    # A ray that was still climbing when it ran out of range has not found an
    # edge — the edge is further out than the user allowed. Testing that is what
    # a three-point crest check was meant to do, but the distance penalty shifts
    # the argmax a sample off the true crest on a broad edge, and the strict
    # version then threw away 94% of a clean square's boundary.
    last = max(search - 1, 0)
    still_rising = (best == search) & (resp[:, search] > resp[:, last])

    # Three ways of being unconvincing, all measured on the data rather than set
    # in advance: no better than where we already are, weak next to the rest of
    # this contour, or indistinguishable from this ray's own noise.
    stronger = peak >= _MIN_GAIN * base
    floor = _FLOOR_FRAC * float(np.percentile(peak, _STRONG_PCTL))
    ray_med = np.median(resp, axis=1)
    ray_mad = np.median(np.abs(resp - ray_med[:, None]), axis=1)
    significant = peak >= ray_med + _NOISE_SIGMAS * _MAD_TO_SIGMA * ray_mad

    accept = (~still_rising) & (peak > _EPS) & stronger & (peak >= floor) & significant

    if n >= _COHERENCE_WIN:
        in_window = _wrap_filter(
            accept.astype(np.float64), _COHERENCE_WIN, "mean") * _COHERENCE_WIN
        accept = accept & (in_window >= _COHERENCE_MIN - 0.5)

    offsets = np.where(accept, best.astype(np.float64), np.nan)
    filled = _interp_circular(offsets)
    if filled is None:
        return pts, 0, 0

    if accept.mean() < _MIN_COVERAGE:
        # Too little of the contour agreed to speak for the rest of it. Keep
        # what was measured, fade it out over a few points, and leave the
        # unsupported stretches where SAM put them.
        falloff = np.clip(
            1.0 - _dist_to_anchor(accept) / _LOW_COVERAGE_SUPPORT, 0.0, 1.0)
        filled = filled * falloff

    filled = _wrap_filter(filled, _MEDIAN_WIN, "median")
    filled = _wrap_filter(filled, _SMOOTH_WIN, "mean")
    filled = np.clip(filled, 0.0, float(search))

    moved = int(accept.sum())
    return pts + normals * filled[:, None], moved, n - moved


def refine_mask(mask: np.ndarray, gx: np.ndarray, gy: np.ndarray,
                search_range: int = DEFAULT_SEARCH_RANGE
                ) -> Tuple[np.ndarray, Dict[str, int]]:
    """Refined copy of mask, plus counts describing what happened.

    The mask comes back with the same 0/1 shape it went in with. Holes are kept:
    they are refined like any other boundary and punched back out afterwards.
    """
    m = (mask > 0).astype(np.uint8)
    stats = {"points": 0, "moved": 0, "interpolated": 0, "contours": 0}
    if not m.any():
        return m, stats
    search = clamp_range(search_range)

    contours, hierarchy = cv2.findContours(
        m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return m, stats

    outer: List[np.ndarray] = []
    holes: List[np.ndarray] = []
    for i, contour in enumerate(contours):
        pts = contour.reshape(-1, 2).astype(np.float64)
        if len(pts) >= _MIN_CONTOUR:
            pts, moved, interpolated = _refine_contour(pts, m, gx, gy, search)
            stats["points"] += len(pts)
            stats["moved"] += moved
            stats["interpolated"] += interpolated
            stats["contours"] += 1
        # hierarchy[0][i][3] is the parent: -1 marks an outer contour.
        target = outer if hierarchy[0][i][3] == -1 else holes
        target.append(np.round(pts).astype(np.int32))

    out = np.zeros_like(m)
    if outer:
        cv2.fillPoly(out, outer, 1)
    if holes:
        cv2.fillPoly(out, holes, 0)
    return out, stats
