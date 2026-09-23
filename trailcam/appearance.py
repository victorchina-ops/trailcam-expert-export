"""Colour appearance descriptors for associating people across event frames.

A descriptor holds two L1-normalised joint HSV histograms (16 H x 4 S x 4 V =
256 bins, index ``(h * 4 + s) * 4 + v``) over the upper (15-50 % of box height)
and lower (50-90 %) body, restricted to the person mask when given and ignoring
dark pixels (V < 20). It is weak similarity evidence between frames of one
event, never an identity claim; night-IR (greyscale) crops only carry V.

HSV follows the OpenCV definition (hue 0 for achromatic pixels) but is binned
with exact integer arithmetic, so the numpy (CPU) and torch (GPU) paths give
bit-identical histograms. A working CUDA device is used when available; any
CUDA problem falls back to CPU with the same result.
"""
from __future__ import annotations

import math
import numbers
import re
import warnings

import numpy as np

VERSION = "hsv_v1"
PARTS = ("upper", "lower")
POLICY = {
    "regions": {"upper": (0.15, 0.50), "lower": (0.50, 0.90)},  # fractions of clamped box height
    "bins": (16, 4, 4),  # H, S, V bin counts; any change needs a new VERSION
    "min_value": 20,     # pixels with V (0-255) below this are ignored
    "min_pixels": 50,    # a part with fewer usable pixels is None (not assessable)
    "decimals": 5,
}
BIN_COUNT = POLICY["bins"][0] * POLICY["bins"][1] * POLICY["bins"][2]
GPU_MIN_PIXELS = 65536  # speed only: smaller crops are faster in numpy; results are identical
_DEVICE_PATTERN = re.compile(r"auto|gpu|cuda|cuda:\d+|\d+")
_BACKENDS: dict = {}  # normalised device request -> (resolved device, torch module or None)


def descriptor(rgb, box, mask=None, device="auto") -> dict:
    """Appearance of the person in ``box``; see the module docstring.

    ``rgb``: HxWx3 RGB array with values 0-255 (HxWx4 and HxW greyscale also
    accepted; non-finite float pixels are ignored). ``box``: [x1, y1, x2, y2]
    pixels, clamped to the image; None, None/non-finite coordinates or empty
    boxes give an invalid descriptor. ``mask``: None, an image-sized (H, W)
    array (nonzero = person) or polygon(s) of [x, y] image points such as
    ``mask_polygon_xy``. ``device``: auto | cpu | cuda[:N] | N.
    A part is None with < min_pixels usable pixels; ``valid`` is true when any
    part is usable; ``pixels`` records the usable pixel count per part.
    """
    image = np.asarray(rgb)
    if image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[2] not in (3, 4)):
        raise ValueError("rgb must be an HxWx3 RGB array (HxWx4 or HxW greyscale accepted)")
    if image.dtype.kind not in "biuf":
        raise ValueError(f"rgb must hold numbers, not {image.dtype}")
    backend = _backend(device)  # validates the device even when the box turns out empty
    height, width = image.shape[:2]
    shape = _parse_mask(mask, height, width)
    counts = [[0] * BIN_COUNT for _ in PARTS]
    window = _window(box, width, height)
    if window is not None:
        c0, c1, rows = window
        r0, r1 = min(r[0] for r in rows), max(r[1] for r in rows)
        if c1 > c0 and r1 > r0:
            crop = image[r0:r1, c0:c1]
            crop = crop[..., :3] if crop.ndim == 3 else np.repeat(crop[..., None], 3, axis=2)
            keep = None if shape is None else (
                shape[r0:r1, c0:c1] if isinstance(shape, np.ndarray) else _fill_polygons(shape, r0, r1, c0, c1))
            if crop.dtype != np.uint8:
                crop = crop.astype(np.float64)
                finite = np.isfinite(crop).all(axis=2)  # NaN/inf pixels carry no colour: ignore them
                keep = finite if keep is None else keep & finite
                crop = np.clip(np.rint(np.where(finite[..., None], crop, 0.0)), 0, 255).astype(np.uint8)
            counts = _count(crop, keep, [(a - r0, b - r0) for a, b in rows], backend)
    result = {"version": VERSION}
    for part, part_counts in zip(PARTS, counts):
        total = sum(part_counts)
        result[part] = ([round(c / total, POLICY["decimals"]) for c in part_counts]
                        if total >= POLICY["min_pixels"] else None)
    result["valid"] = any(result[part] is not None for part in PARTS)
    result["pixels"] = {part: sum(c) for part, c in zip(PARTS, counts)}
    return result


def distance(a, b) -> float | None:
    """Mean Hellinger (Bhattacharyya) distance over parts valid in both, in [0, 1].

    0 = identical colour distributions, 1 = disjoint. None when either
    descriptor is missing, invalid, of another version, or no part is shared.
    """
    if not all(isinstance(d, dict) and d.get("valid") is True and d.get("version") == VERSION for d in (a, b)):
        return None
    values = []
    for part in PARTS:
        p, q = _histogram(a, part), _histogram(b, part)
        if p is None or q is None:
            continue
        # Normalising by the sums absorbs the 5-decimal rounding (as OpenCV does);
        # _histogram scales each to max 1 first, so no scale can overflow or underflow.
        coefficient = float(np.sqrt(p * q).sum()) / math.sqrt(float(p.sum()) * float(q.sum()))
        values.append(math.sqrt(max(0.0, 1.0 - coefficient)))
    return round(sum(values) / len(values), 6) if values else None


def resolve_device(device="auto") -> str:
    """Device descriptors will use: a CUDA device that passed a real kernel probe, else "cpu".

    Resolved once per process and request. "auto" falls back silently when
    PyTorch or CUDA is absent and warns when CUDA is present but broken;
    explicit CUDA requests always warn on fallback.
    """
    return _backend(device)[1]


def _histogram(d, part):
    """A part's bins scaled to max 1, or None unless BIN_COUNT finite non-negative numbers, not all 0."""
    values = d.get(part)
    if not isinstance(values, (list, tuple)) or len(values) != BIN_COUNT:
        return None
    # Vectorised validation: distance() runs for every candidate pair in an
    # event, so per-element Python checks dominated association time.
    try:
        array = np.asarray(values)
    except (TypeError, ValueError, OverflowError):
        return None
    if array.shape != (BIN_COUNT,) or array.dtype.kind not in "iuf":
        return None  # strings, booleans and mixed objects are not histogram values
    with np.errstate(all="ignore"):
        array = array.astype(np.float64)
    if not np.all(np.isfinite(array)) or np.any(array < 0):
        return None
    top = float(array.max())
    return array / top if top > 0 else None


def _pixel_edge(value):
    return int(math.floor(value + 0.5))


def _coordinate(value):
    """Box coordinate as float; None and values beyond float range count as non-finite."""
    if value is None:
        return math.nan
    if isinstance(value, (str, bytes)) or np.ndim(value) != 0:
        raise ValueError("box must be [x1, y1, x2, y2] numbers")
    try:
        return float(value)
    except OverflowError:
        return math.nan
    except (TypeError, ValueError) as exc:
        raise ValueError("box must be [x1, y1, x2, y2] numbers") from exc


def _window(box, width, height):
    """Clamped box as (col_start, col_end, [(row_start, row_end) per part]) or None."""
    if box is None:
        return None
    try:
        items = [] if isinstance(box, (str, bytes)) else list(box)
    except TypeError as exc:
        raise ValueError("box must be [x1, y1, x2, y2]") from exc
    if len(items) != 4:
        raise ValueError("box must be [x1, y1, x2, y2]")
    values = [_coordinate(v) for v in items]
    if not all(math.isfinite(v) for v in values):
        return None
    x1, x2 = (min(max(v, 0.0), float(width)) for v in values[0::2])
    y1, y2 = (min(max(v, 0.0), float(height)) for v in values[1::2])
    if x2 <= x1 or y2 <= y1:
        return None
    rows = [(_pixel_edge(y1 + a * (y2 - y1)), _pixel_edge(y1 + b * (y2 - y1)))
            for a, b in (POLICY["regions"][part] for part in PARTS)]
    return _pixel_edge(x1), _pixel_edge(x2), rows


def _parse_mask(mask, height, width):
    """None, a boolean (height, width) raster, or a list of (N, 2) float polygons."""
    if mask is None:
        return None
    if isinstance(mask, np.ndarray):
        if mask.shape != (height, width):
            raise ValueError(f"mask shape {mask.shape} does not match image {(height, width)}; "
                             "pass polygons as lists of [x, y] points")
        return mask != 0
    items = list(mask)
    if items and np.ndim(items[0]) == 1 and len(items[0]) == 2:
        items = [items]  # one polygon of [x, y] points
    polygons = []
    for item in items:
        points = np.asarray(item, dtype=np.float64)
        if points.size and (points.ndim != 2 or points.shape[1] != 2):
            raise ValueError("mask polygons must be lists of [x, y] points")
        if not np.isfinite(points).all():
            raise ValueError("Nonfinite mask polygon coordinate")
        polygons.append(points.reshape(-1, 2))
    return polygons


def _fill_polygons(polygons, r0, r1, c0, c1):
    """Union of even-odd polygon fills sampled at pixel centres of rows r0:r1, cols c0:c1."""
    rows, cols = r1 - r0, c1 - c0
    inside = np.zeros((rows, cols), dtype=bool)
    centres = np.arange(r0, r1, dtype=np.float64) + 0.5
    for points in polygons:
        if len(points) < 3:
            continue
        xa, ya = points[:, 0], points[:, 1]
        xb, yb = np.roll(xa, -1), np.roll(ya, -1)
        down = yb < ya  # orient every edge top to bottom: vertex order cannot change a crossing
        x0, y0, x1, y1 = (np.where(down, b, a) for a, b in ((xa, xb), (ya, yb), (xb, xa), (yb, ya)))
        row, edge = np.nonzero((y0 <= centres[:, None]) != (y1 <= centres[:, None]))
        if not len(row):
            continue
        with np.errstate(over="ignore", invalid="ignore"):  # only absurd (~1e308) coordinates overflow
            t = (centres[row] - y0[edge]) / (y1[edge] - y0[edge])
            crossing = x0[edge] + t * (x1[edge] - x0[edge])
        crossing = np.where(np.isnan(crossing), x0[edge], crossing)  # 0 * inf happens only at t == 0
        # Pixels from this column rightwards have the crossing to their left.
        first = np.clip(np.floor(crossing - c0 - 0.5) + 1, 0, cols).astype(np.int64)
        toggles = np.zeros((rows, cols + 1), dtype=np.int32)
        np.add.at(toggles, (row, first), 1)
        inside |= np.cumsum(toggles[:, :cols], axis=1) % 2 == 1
    return inside


def _joint_bins(r, g, b, xp):
    """Per-pixel joint bin index and V with exact integer HSV; xp is numpy or torch."""
    h_bins, s_bins, v_bins = POLICY["bins"]
    value = xp.maximum(xp.maximum(r, g), b)
    chroma = value - xp.minimum(xp.minimum(r, g), b)
    # Hue in sixths of the circle, scaled by chroma: [0, 6 * chroma).
    hue = xp.where(value == r, g - b, xp.where(value == g, b - r + 2 * chroma, r - g + 4 * chroma))
    hue = xp.where(hue < 0, hue + 6 * chroma, hue)
    h = (h_bins * hue) // (6 * xp.where(chroma > 0, chroma, chroma + 1))
    s = (s_bins * chroma) // xp.where(value > 0, value, value + 1)
    s = xp.where(s >= s_bins, s - 1, s)
    return (h * s_bins + s) * v_bins + (v_bins * value) // 256, value


def _count_numpy(crop, keep, parts):
    pixels = crop.astype(np.int32)
    bins, value = _joint_bins(pixels[..., 0], pixels[..., 1], pixels[..., 2], np)
    usable = value >= POLICY["min_value"]
    if keep is not None:
        usable &= keep
    return [np.bincount(bins[a:b][usable[a:b]], minlength=BIN_COUNT).tolist() for a, b in parts]


def _count_torch(torch, device, crop, keep, parts):
    pixels = torch.from_numpy(np.require(crop, np.uint8, ["C", "W"])).to(device=device, dtype=torch.int32)
    bins, value = _joint_bins(pixels[..., 0], pixels[..., 1], pixels[..., 2], torch)
    usable = value >= POLICY["min_value"]
    if keep is not None:
        usable &= torch.from_numpy(np.require(keep, bool, ["C", "W"])).to(device)
    bins = torch.where(usable, bins, BIN_COUNT)  # overflow bin instead of a syncing boolean index
    return torch.stack([torch.bincount(bins[a:b].flatten(), minlength=BIN_COUNT + 1)[:BIN_COUNT]
                        for a, b in parts]).cpu().tolist()


def _count(crop, keep, parts, backend):
    key, resolved, torch = backend
    if resolved != "cpu" and crop.shape[0] * crop.shape[1] >= GPU_MIN_PIXELS:
        try:
            return _count_torch(torch, resolved, crop, keep, parts)
        except Exception as exc:  # any GPU problem: numpy gives the identical result
            _BACKENDS[key] = ("cpu", None)
            warnings.warn("CUDA appearance descriptor failed; using CPU from now on: "
                          f"{type(exc).__name__}: {exc}", RuntimeWarning, stacklevel=3)
    return _count_numpy(crop, keep, parts)


def _import_torch():
    import torch  # lazy: descriptors must work without PyTorch installed
    return torch


def _backend(device):
    """(normalised request, resolved device, torch module or None), cached per request."""
    key = "cpu" if device is None else str(device).strip().lower()
    if key == "cpu":
        return key, "cpu", None
    if key not in _BACKENDS:
        if not _DEVICE_PATTERN.fullmatch(key):
            raise ValueError("device must be auto, cpu, gpu, cuda, cuda:N or a CUDA index such as 0")
        try:
            torch = _import_torch()
            available, reason = bool(torch.cuda.is_available()), "CUDA is unavailable"
        except Exception as exc:  # a missing or broken PyTorch install means CPU
            torch, available, reason = None, False, f"PyTorch is unavailable: {type(exc).__name__}: {exc}"
        resolved = "cpu"
        if available:
            try:
                from .vision import select_device  # shared real-kernel CUDA probe
                resolved, reason = select_device(torch, key)
            except Exception as exc:
                resolved, reason = "cpu", f"CUDA check failed: {type(exc).__name__}: {exc}"
        if resolved == "cpu" and (available or key != "auto"):
            warnings.warn("Appearance descriptors use CPU: " + str(reason), RuntimeWarning, stacklevel=3)
        _BACKENDS[key] = (resolved, None if resolved == "cpu" else torch)
    return (key, *_BACKENDS[key])
