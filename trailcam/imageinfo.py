"""Capture metadata, camera data-strip rows, and camera ids for one image.

Pure CPU helpers (numpy and the standard library only; no model loading and
no device selection). Outputs are plain JSON types. Capture times are naive
local camera-clock times: EXIF carries no reliable time zone on trail cameras,
and `clock_suspect` is informational only.
"""
from __future__ import annotations

from datetime import datetime, timedelta
import math
import os
from pathlib import Path, PurePosixPath
import re

import numpy as np

RULE_VERSION = "imageinfo_v1"
POLICY = {
    # Clock plausibility (informational; relative times within a camera still count).
    "clock_min_year": 2015, "clock_max_days_after_mtime": 1.0, "clock_year_start_minutes": 10,
    # Data strip: a band at the top or bottom edge whose rows are mostly very dark.
    "strip_dark_level": 30,            # gray < 30/255 is "very dark"
    "strip_text_level": 128,           # gray >= 128 is bright text on the bar
    "strip_row_dark_share": 0.70,      # band rows are >= 70% very dark (text pixels excluded)
    "strip_text_row_dark_share": 0.40, # ...and >= 40% very dark overall; the inner row >= 70% overall
    "strip_max_fraction": 0.12, "strip_min_fraction": 0.01, "strip_min_image_side": 32,
    # Sharp horizontal boundary with the scene (gray levels, on the working copy).
    "strip_edge_min_contrast": 20.0,   # scene rows minus band rows, and the jump across the edge
    "strip_edge_jump_share": 0.60,     # the jump holds most of the contrast (not a gradient)
    "strip_edge_column_share": 0.50,   # the step is present across most of the width
    "strip_edge_window": 0.01,         # scene reference rows, fraction of height (>= 2 rows)
    "strip_work_height": 1000, "strip_work_width": 512,
}
EXIF_IFD = 0x8769
TAG_MAKE, TAG_MODEL, TAG_DATETIME = 0x010F, 0x0110, 0x0132
# (time_source, datetime tag, matching sub-second tag), in order of preference.
TIME_TAGS = (("exif_original", 0x9003, 0x9291), ("exif_digitized", 0x9004, 0x9292),
             ("exif_datetime", TAG_DATETIME, 0x9290))
_DATETIME = re.compile(r"(\d{4})[:\-/.](\d{1,2})[:\-/.](\d{1,2})[T ]+(\d{1,2})[:.](\d{1,2})[:.](\d{1,2})"
                       r"(?:[.,](\d+))?", re.ASCII)
_DIGITS = re.compile(r"[0-9]+")


def _text(value):
    """EXIF ASCII value as a clean string, or None for missing/non-text values."""
    if isinstance(value, bytes):
        value = value.decode("utf-8", "replace")
    if not isinstance(value, str):
        return None
    value = "".join(ch for ch in value.split("\x00", 1)[0] if ch.isprintable()).strip()
    return value or None


def _parse_datetime(value):
    """(datetime, inline fraction digits) from an EXIF date string; (None, None) if invalid."""
    text = _text(value)
    match = _DATETIME.match(text) if text else None
    if not match:
        return None, None
    try:
        return datetime(*(int(g) for g in match.groups()[:6])), match.group(7)
    except ValueError:  # e.g. "0000:00:00 00:00:00" or month 13
        return None, None


def _subsec(value):
    text = _text(value)
    return text if text and _DIGITS.fullmatch(text) else None


def _exif_tables(image):
    """(IFD0, Exif IFD) as plain dicts; any decoding failure yields empty tables."""
    ifd0, sub = {}, {}
    try:
        exif = image.getexif()
    except Exception:  # Malformed EXIF must never stop the image.
        return ifd0, sub
    try:
        ifd0 = dict(exif)
    except Exception:
        pass
    try:
        sub = dict(exif.get_ifd(EXIF_IFD))
    except Exception:
        pass
    return ifd0, sub


def _lookup(tag, *tables):
    for table in tables:
        value = table.get(tag)
        if value is not None:
            return value
    return None


def _file_mtime(path):
    if path is None or (isinstance(path, (str, bytes)) and not path):  # "" would stat the cwd
        return None
    try:
        return datetime.fromtimestamp(Path(os.fsdecode(path)).stat().st_mtime)
    except (OSError, ValueError, OverflowError, TypeError):
        return None


def _iso(moment):
    return moment.isoformat(timespec="microseconds" if moment.microsecond else "seconds")


def sequence_number(path) -> int | None:
    """Last run of ASCII digits in the filename stem (IMAG0571 -> 571), else None."""
    if path is None:
        return None
    if isinstance(path, bytes):
        path = os.fsdecode(path)
    runs = _DIGITS.findall(PurePosixPath(str(path).replace("\\", "/")).stem)
    try:
        return int(runs[-1]) if runs else None
    except ValueError:  # beyond Python's int-from-string digit limit: not a counter
        return None


def clock_suspect(capture, mtime=None) -> bool:
    """Implausible camera clock: before 2015, > 1 day after file mtime, or a year-start reset default."""
    if capture is None:
        return False
    if capture.year < POLICY["clock_min_year"]:
        return True
    if mtime is not None and capture - mtime > timedelta(days=POLICY["clock_max_days_after_mtime"]):
        return True
    return (capture.month == 1 and capture.day == 1 and capture.hour == 0
            and capture.minute < POLICY["clock_year_start_minutes"])


def read_capture_metadata(image, path) -> dict:
    """Capture time from EXIF (original, digitized, DateTime) else file mtime (local).

    `image` is the opened PIL image (EXIF-transposed copies keep EXIF); either
    argument may be None. `time_source` is None only when no time was found.
    """
    ifd0, sub = _exif_tables(image) if image is not None else ({}, {})
    capture, source, subsec = None, None, None
    for name, tag, subsec_tag in TIME_TAGS:
        tables = (ifd0, sub) if tag == TAG_DATETIME else (sub, ifd0)
        moment, inline = _parse_datetime(_lookup(tag, *tables))
        if moment is None:
            continue
        subsec = _subsec(_lookup(subsec_tag, sub, ifd0)) or inline
        if subsec:
            moment = moment.replace(microsecond=int(subsec[:6].ljust(6, "0")))
        capture, source = moment, name
        break
    mtime = _file_mtime(path)
    if capture is None and mtime is not None:
        capture, source = mtime, "file_mtime"
    return {"capture_time": _iso(capture) if capture else None, "time_source": source,
            "subsec": subsec, "camera_make": _text(ifd0.get(TAG_MAKE)),
            "camera_model": _text(ifd0.get(TAG_MODEL)), "sequence_number": sequence_number(path),
            "clock_suspect": clock_suspect(capture, mtime)}


def _gray(pixels):
    """ITU-R 601 luma (299 R + 587 G + 114 B) / 1000 as float64.

    Integer weights keep grey pixels exact (v, v, v -> v), so the 30/128
    thresholds classify RGB and single-channel input identically; elementwise,
    so results are platform independent.
    """
    if pixels.ndim == 2:
        return pixels.astype(np.float64)
    if pixels.shape[2] >= 3:
        rgb = pixels[..., :3].astype(np.float64)
        return (rgb[..., 0] * 299 + rgb[..., 1] * 587 + rgb[..., 2] * 114) / 1000
    return pixels[..., 0].astype(np.float64)


def _band_rows(array, from_top, row_step, col_step):
    """Band height in original rows at one edge, or 0 when no valid strip is present."""
    height = array.shape[0]
    level = POLICY["strip_dark_level"]
    strict, loose = POLICY["strip_row_dark_share"], POLICY["strip_text_row_dark_share"]
    max_rows = int(POLICY["strip_max_fraction"] * height)
    work_rows = -(-height // row_step)
    limit = -(-max_rows // row_step)  # longest band in working rows (rounded up; exact check below)
    window = max(2, int(round(POLICY["strip_edge_window"] * work_rows)))
    count = min(work_rows, limit + 2 + window)

    def rows_at(offsets):  # offsets are counted from the image edge inward
        rows = offsets if from_top else height - 1 - offsets
        return _gray(array[rows, ::col_step])

    gray = rows_at(row_step * np.arange(count))
    dark = (gray < level).mean(axis=1)
    text = (gray >= POLICY["strip_text_level"]).mean(axis=1)
    # Bar rows: very dark apart from bright text; the band grows from the edge.
    scan = (dark >= loose) & (dark >= strict * (1.0 - text))
    scan = scan[:limit + 1]
    stop = int(np.argmin(scan)) if not scan.all() else len(scan)
    strict_rows = np.flatnonzero(dark[:stop] >= strict)
    if not len(strict_rows):
        return 0
    band = int(strict_rows[-1]) + 1  # the band ends on a text-free, >= 70% very dark row
    if band > limit or band + 1 >= count:
        return 0
    # Sharp horizontal edge: skip one possibly transitional row at the boundary.
    inside, outside = gray[max(0, band - 2):band], gray[band + 1:band + 1 + window]
    means = gray.mean(axis=1)
    min_contrast = POLICY["strip_edge_min_contrast"]
    contrast = float(outside.mean() - inside.mean())
    jump = float(means[band + 1] - means[band - 1])
    columns = float(((outside.mean(axis=0) - inside.mean(axis=0)) >= .5 * min_contrast).mean())
    if not (contrast >= min_contrast and jump >= min_contrast and columns >= POLICY["strip_edge_column_share"]
            and jump >= POLICY["strip_edge_jump_share"] * contrast):  # written so NaN rejects
        return 0
    rows = band * row_step
    if row_step > 1:  # refine the boundary between two working rows at full resolution
        offsets = (band - 1) * row_step + np.arange(1, row_step)
        between = (rows_at(offsets) < level).mean(axis=1) >= strict
        rows = int(offsets[np.argmin(between)]) if not between.all() else rows
    minimum = max(2, math.ceil(POLICY["strip_min_fraction"] * height))
    return rows if minimum <= rows <= max_rows else 0


def detect_data_strip(rgb) -> dict:
    """Rows at the top/bottom that belong to a camera info bar; 0 when absent.

    A strip is an edge band (at most 12% of height) whose rows are >= 70% very
    dark (< 30/255) once bright text pixels are set aside, ending on a row that
    is >= 70% very dark outright, and whose boundary with the scene is a sharp
    horizontal step across most of the width. Dark ground or sky that fades
    gradually, or has an irregular outline, is not a strip. Works on a strided
    grayscale copy of the edge rows; the boundary is refined at full resolution.
    Accepts real-valued HxW, HxWx3 or HxWx4 arrays with 0-255 values (RGB or BGR).
    """
    none = {"top": 0, "bottom": 0}
    try:
        array = np.asarray(rgb)
        if array.dtype.kind not in "biuf" or array.ndim not in (2, 3) or (array.ndim == 3 and array.shape[2] < 1):
            return none  # non-numeric (object, string, complex) or wrongly shaped input
        height, width = array.shape[:2]
        if min(height, width) < POLICY["strip_min_image_side"]:
            return none
        row_step = max(1, -(-height // POLICY["strip_work_height"]))
        col_step = max(1, -(-width // POLICY["strip_work_width"]))
        with np.errstate(all="ignore"):  # NaN/inf/huge floats simply fail the tests (even under -W error)
            return {"top": _band_rows(array, True, row_step, col_step),
                    "bottom": _band_rows(array, False, row_step, col_step)}
    except (TypeError, ValueError):  # ragged input
        return none


def in_data_strip(box, strip, height, fraction=0.5) -> bool:
    """True when >= `fraction` of the box area (clipped to the image rows) is in strip rows.

    Invalid or degenerate boxes, a missing strip, or an invalid height give
    False; `fraction` <= 0 means any overlap.
    """
    if isinstance(box, (str, bytes)):  # "0099" would otherwise unpack as four numbers
        return False
    try:
        x1, y1, x2, y2 = (float(v) for v in box)
        height, fraction = float(height), float(fraction)
        top, bottom = (max(0.0, float(strip.get(k) or 0)) for k in ("top", "bottom"))
    except (TypeError, ValueError, AttributeError):
        return False
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2, height, fraction, top, bottom)) or height <= 0:
        return False
    y1, y2 = max(y1, 0.0), min(y2, height)
    if x2 <= x1 or y2 <= y1:
        return False
    top = min(top, height)
    bottom_start = max(height - bottom, top)
    inside = max(0.0, min(y2, top) - y1) + max(0.0, y2 - max(y1, bottom_start))
    return inside > 0 and inside >= fraction * (y2 - y1) * (1 - 1e-9)


def camera_id_for(relative_path: str, mode: str = "folder", pattern=None) -> str:
    """Camera id: parent folder (POSIX, "." at root), regex group `camera`, or "all".

    Regex mode searches the POSIX-normalised path and falls back to the folder
    when the pattern does not match (or matches an empty group).
    """
    if isinstance(relative_path, bytes):
        relative_path = os.fsdecode(relative_path)
    text = "" if relative_path is None else str(relative_path).replace("\\", "/")
    mode = str(mode).lower()
    if mode == "single":
        return "all"
    if mode == "regex":
        if pattern is None:
            raise ValueError("Camera id mode 'regex' needs a pattern with a named group 'camera'")
        try:
            compiled = pattern if isinstance(pattern, re.Pattern) else re.compile(pattern)
        except (re.error, TypeError) as exc:  # TypeError: not a str/bytes pattern
            raise ValueError("Invalid camera id pattern: " + str(exc)) from exc
        if not isinstance(compiled.pattern, str) or "camera" not in compiled.groupindex:
            raise ValueError("Camera id pattern must be a text regex with a named group 'camera'")
        match = compiled.search(text)
        if match and match.group("camera"):
            return match.group("camera")
    elif mode != "folder":
        raise ValueError("Camera id mode must be folder, regex, or single")
    return PurePosixPath(text).parent.as_posix() if text else "."
