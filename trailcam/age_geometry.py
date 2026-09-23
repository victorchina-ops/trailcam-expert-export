"""Adult/child estimates from standing-height geometry; unknown is the default.

Pure post-processing (numpy only, no models, no device selection): results are
identical whether inference ran on CUDA or on the CPU fallback. Labels are
geometric estimates of apparent standing height, never identity claims.

Geometry (``person_geometry``). ``height_px`` runs from the crown to the foot.
Only keypoints with conf >= .4 that lie inside the box grown by 0.25 * box
height on every side are used; a confident keypoint outside that zone (pose
matched to the wrong box, garbage coordinates) is dropped and makes the person
incomplete. Crown = min y of nose/eyes/ears raised by 0.5 * (shoulder_y -
eye_y) (ear_y stands in for eye_y when no eye is confident); without that
evidence the box top is used, because an unextended face keypoint sits below
the crown and would bias ratios toward "child". Foot = lowest usable ankle
(x = mean of usable ankles), else box bottom / box centre. ``complete``
requires head keypoints, a usable ankle, a box clear of the left/right/top
border, the optional top data strip (``strip_top``) and the bottom data strip /
bottom border (2 px margin), usable shoulder and hip keypoints (at least one
each) with hip below shoulder by >= 0.15 * height, and extended legs (foot -
hip >= 1.2 * torso; seated or crouching people are not standing-height
observations). Non-finite results give no geometry.

Calibration (``fit_camera``). On one camera's complete people fit
``height_px = a*foot_y + b*foot_x + c``: least-median-of-relative-residuals
start (deterministic elemental subsets, scored on at most 20000 evenly spaced
rows in bounded-memory chunks) followed by Tukey-bisquare IRLS on relative
residuals r = h/h_pred - 1 (c = 0.20, so children and seated people,
r <~ -0.22, get zero weight). Inliers are |r| < 0.20. The slope terms are only
fitted when the foot range spans >= 0.25 median heights (x also needs
|corr(x, y)| <= 0.9); otherwise a constant-height model is used and accepted
only when its residual MAD <= 0.06. Status ``unreliable`` when the residual MAD
exceeds 0.10, fewer than half the people are inliers, or a predicted height is
non-positive anywhere in the observed foot box.

Classification (``classify``), first applicable rule:
1. Incomplete person -> unknown, method ``none``.
2. ``ok`` calibration and foot_y within the calibrated foot range extended by
   0.5 * h_pred (foot_x too when b != 0): ratio = h / h_pred; child if
   ratio <= 0.82, adult if ratio >= 0.88, else unknown. Ratios outside
   [0.40, 1.50] are implausible for a standing person -> unknown.
3. Relative height: eligible peers are the *other* complete people of the
   same image whose |foot_y - peer foot_y| <= 0.08 * image height (inclusive).
   ratio = h / max(peer height); child if ratio <= 0.75, else unknown.
   Similar-height peers alone cannot establish an adult reference: two
   similarly sized children must not label each other as adults. Ratios outside [0.30, 3.00] (a height pair no two
   standing people form) -> implausible, unknown.
4. No eligible peers -> unknown, method ``none``, ratio None.
Ratios are rounded to 4 decimals before the threshold comparison, so the
reported ratio always agrees with the label.
"""
from __future__ import annotations

import math

import numpy as np

RULE_VERSION = "age_geometry_v2"
HEAD, EYES, EARS = (0, 1, 2, 3, 4), (1, 2), (3, 4)
SHOULDERS, HIPS, ANKLES = (5, 6), (11, 12), (15, 16)
POLICY = {
    "keypoint_confidence": 0.40, "crown_extension": 0.50, "border_margin_px": 2.0,
    "keypoint_box_tolerance": 0.25,
    # Hip-to-ankle / shoulder-to-hip: standing adults ~1.7, toddlers ~1.4-1.6, adults on a
    # 45 cm bench ~0.9. At 1.20 a seated adult who still passes measures well above the 0.82
    # child cut (1.10 put that edge case exactly on the cut), so sitting is not read as "child".
    "min_torso_per_height": 0.15, "min_leg_per_torso": 1.20,
    "min_people_floor": 5, "min_span_per_height": 0.25, "max_xy_correlation": 0.90,
    "lmeds_subsets": 500, "lmeds_seed": 0, "lmeds_max_rows": 20000, "lmeds_chunk_elements": 2_000_000,
    "tukey_c": 0.20, "max_iterations": 100,
    "max_residual_mad": 0.10, "max_residual_mad_constant": 0.06, "min_inlier_fraction": 0.50,
    "calibration_child_max": 0.82, "calibration_adult_min": 0.88,
    "calibration_ratio_min": 0.40, "calibration_ratio_max": 1.50,
    "calibration_extrapolation_per_height": 0.50,
    "relative_child_max": 0.75, "relative_adult_min": 0.90,
    "relative_ratio_min": 0.30, "relative_ratio_max": 3.00,
    "peer_foot_window_per_image_height": 0.08,
}


def _finite(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _is_true(value):
    return value is True or (isinstance(value, np.bool_) and bool(value))


def _ident(value):
    """Identifier as a JSON-safe comparable value: str / int / finite float / None, else its str()."""
    value = value.item() if isinstance(value, np.generic) else value
    if value is None or isinstance(value, (str, int)) or (isinstance(value, float) and math.isfinite(value)):
        return value
    return str(value)


def _seq(value):
    """Items of a list / tuple / >= 1-D array; None for str, dict, set, scalars."""
    if isinstance(value, np.ndarray):
        return list(value) if value.ndim else None
    return list(value) if isinstance(value, (list, tuple)) else None


def _items(values):
    """Items of any iterable collection (not str/dict); [] for None and scalars."""
    if values is None or isinstance(values, (str, bytes, dict)):
        return []
    try:
        return list(values)
    except TypeError:
        return []


def _mean(values):
    return sum(values) / len(values)


def _box(box):
    values = _seq(box)
    if values is None or len(values) != 4:
        return None
    values = [_finite(v) for v in values]
    if None in values or not (0 < values[2] - values[0] < math.inf and 0 < values[3] - values[1] < math.inf):
        return None
    return values


def _keypoints(points):
    if points is None:
        return None, "no_keypoints"
    rows = _seq(points)
    rows = [_seq(k) for k in rows] if rows is not None else None
    if rows is None or len(rows) != 17 or any(k is None or len(k) != 3 for k in rows):
        return None, "invalid_keypoints"
    rows = [[_finite(v) for v in k] for k in rows]
    if any(None in k for k in rows):
        return None, "invalid_keypoints"
    return rows, None


def person_geometry(p):
    """Standing-height geometry of one person; ``complete`` gates its use for age."""
    p = p if isinstance(p, dict) else {}
    out = {"person_id": _ident(p.get("person_id")), "image_id": _ident(p.get("image_id")),
           "camera_id": _ident(p.get("camera_id")), "image_height": _finite(p.get("image_height")),
           "height_px": None, "foot_x": None, "foot_y": None, "complete": False,
           "reasons": [], "head_source": None, "foot_source": None}
    reasons = out["reasons"]
    box = _box(p.get("box"))
    if box is None:
        reasons.append("invalid_box")
        return out
    x1, y1, x2, y2 = box
    kps, problem = _keypoints(p.get("keypoints"))
    if problem:
        reasons.append(problem)
    threshold, slack = POLICY["keypoint_confidence"], POLICY["keypoint_box_tolerance"] * (y2 - y1)
    usable, outside = {}, False
    for i in (HEAD + SHOULDERS + HIPS + ANKLES if kps else ()):
        x, y, conf = kps[i]
        if conf < threshold:
            continue
        if x1 - slack <= x <= x2 + slack and y1 - slack <= y <= y2 + slack:
            usable[i] = (x, y)
        else:
            outside = True
    if outside:
        reasons.append("keypoints_outside_box")

    def ys(indices):
        return [usable[i][1] for i in indices if i in usable]

    head, eyes, shoulders, hips = ys(HEAD), ys(EYES) or ys(EARS), ys(SHOULDERS), ys(HIPS)
    ankles = [usable[i] for i in ANKLES if i in usable]
    top, out["head_source"] = y1, "box_top"
    if head and eyes and shoulders and _mean(shoulders) > _mean(eyes):
        top = min(head) - POLICY["crown_extension"] * (_mean(shoulders) - _mean(eyes))
        out["head_source"] = "keypoint_crown"
    if ankles:
        foot_x, foot_y, out["foot_source"] = _mean([a[0] for a in ankles]), max(a[1] for a in ankles), "ankles"
    else:
        foot_x, foot_y, out["foot_source"] = x1 + (x2 - x1) / 2, y2, "box_bottom"
    height = foot_y - top
    if not all(math.isfinite(v) for v in (foot_x, foot_y, height)):
        reasons.append("nonfinite_geometry")
        out.update(head_source=None, foot_source=None)
        return out
    out.update(foot_x=round(foot_x, 2), foot_y=round(foot_y, 2))
    if height <= 0:
        reasons.append("nonpositive_height")
        return out
    out["height_px"] = round(height, 2)
    if kps is not None:
        if not head:
            reasons.append("no_head_keypoints")
        if not ankles:
            reasons.append("no_confident_ankle")
    width, image_height = _finite(p.get("image_width")), out["image_height"]
    if width is None or image_height is None or width <= 0 or image_height <= 0:
        reasons.append("invalid_image_dimensions")
    else:
        margin = POLICY["border_margin_px"]
        strip_top, strip_bottom = (max(0.0, _finite(p.get(k)) or 0.0) for k in ("strip_top", "strip_bottom"))
        for touching, code in ((x1 <= margin, "touches_left_border"),
                               (x2 >= width - margin, "touches_right_border"),
                               (y1 <= strip_top + margin, "touches_top_border"),
                               (y2 >= image_height - strip_bottom - margin, "touches_bottom_or_data_strip")):
            if touching:
                reasons.append(code)
    if kps is not None:
        if not shoulders or not hips:
            reasons.append("no_torso_keypoints")
        else:
            torso = _mean(hips) - _mean(shoulders)
            legs = POLICY["min_leg_per_torso"]
            if torso < POLICY["min_torso_per_height"] * height:
                reasons.append("torso_not_upright")
            elif ankles and legs is not None and foot_y - _mean(hips) < legs * torso:
                reasons.append("legs_not_extended")
    out["complete"] = not reasons
    return out


def _fit_row(g):
    if not isinstance(g, dict) or not _is_true(g.get("complete")):
        return None
    h, y, x = (_finite(g.get(k)) for k in ("height_px", "foot_y", "foot_x"))
    return (y, x, h) if None not in (h, y, x) and h > 0 else None


def _relative(h, pred):
    positive = pred > 0
    return np.where(positive, h / np.where(positive, pred, 1.0) - 1.0, np.inf)


def _lmeds(X, h):
    """Start minimising the median |relative residual| over elemental subsets."""
    n, p = X.shape
    if p == 1:
        picks = np.unique(np.linspace(0, n - 1, min(n, POLICY["lmeds_subsets"])).round().astype(int))
        betas = np.sort(h)[picks][:, None]
    else:
        rng = np.random.RandomState(POLICY["lmeds_seed"])  # legacy stream: stable across numpy versions
        subsets = rng.randint(0, n, size=(POLICY["lmeds_subsets"], p))  # O(p) per subset, unlike choice()
        subsets = subsets[(np.diff(np.sort(subsets, axis=1), axis=1) > 0).all(axis=1)]  # distinct rows only
        A, t = X[subsets], h[subsets]
        usable = np.abs(np.linalg.det(A)) > 1e-6
        if not usable.any():
            return None
        try:
            betas = np.linalg.solve(A[usable], t[usable][..., None])[..., 0]
        except np.linalg.LinAlgError:
            return None
    # Rows are sorted by (y, x, h): evenly spaced picks are a y-stratified sample.
    # Candidates are scored a chunk at a time so memory stays bounded for any n.
    keep = np.unique(np.linspace(0, n - 1, min(n, POLICY["lmeds_max_rows"])).round().astype(int))
    Xs, hs = X[keep], h[keep][:, None]
    step = max(1, POLICY["lmeds_chunk_elements"] // len(keep))
    score = np.concatenate([np.median(np.abs(_relative(hs, Xs @ betas[i:i + step].T)), axis=0)
                            for i in range(0, len(betas), step)])
    return betas[int(np.argmin(score))] if np.isfinite(score).any() else None


def _irls(X, h, beta):
    """Tukey-bisquare IRLS on relative residuals; weights 0 beyond ``tukey_c``."""
    for _ in range(POLICY["max_iterations"]):
        pred = X @ beta
        u = _relative(h, pred) / POLICY["tukey_c"]
        w = np.where(np.abs(u) < 1, (1 - u * u) ** 2, 0.0)
        if np.count_nonzero(w) <= X.shape[1]:
            return None
        s = np.sqrt(w) / np.where(pred > 0, pred, 1.0)
        try:
            new = np.linalg.lstsq(X * s[:, None], h * s, rcond=None)[0]
        except np.linalg.LinAlgError:
            return None
        if not np.all(np.isfinite(new)):
            return None
        done = np.max(np.abs(new - beta)) <= 1e-10 * max(1.0, float(np.max(np.abs(new))))
        beta = new
        if done:
            break
    return beta


def fit_camera(geoms, min_people=20):
    """Robust per-camera ``h = a*foot_y + b*foot_x + c`` over complete people."""
    rows = sorted(r for r in map(_fit_row, _items(geoms)) if r is not None)
    n = len(rows)
    result = {"status": "insufficient", "n": n, "coef": None, "residual_mad": None,
              "inliers": 0, "inlier_fraction": None, "model": None, "foot_y_range": None,
              "foot_x_range": None, "reasons": [], "rule_version": RULE_VERSION}
    if n < max(_finite(min_people) or 0, POLICY["min_people_floor"]):
        result["reasons"].append("fewer_complete_people_than_minimum")
        return result
    with np.errstate(all="ignore"):  # garbage rows must not warn; they are down-weighted
        return _fit(rows, result)


def _fit(rows, result):
    n = len(rows)
    y, x, h = (np.array(column, dtype=float) for column in zip(*rows))
    result.update(foot_y_range=[round(float(y.min()), 2), round(float(y.max()), 2)],
                  foot_x_range=[round(float(x.min()), 2), round(float(x.max()), 2)])
    scale = float(np.median(h))
    span = POLICY["min_span_per_height"] * scale
    use_y = float(np.ptp(y)) >= span
    use_x = (use_y and float(np.ptp(x)) >= span
             and abs(float(np.corrcoef(x, y)[0, 1])) <= POLICY["max_xy_correlation"])
    model = "linear_xy" if use_x else "linear_y" if use_y else "constant"
    # Median centring keeps typical rows well conditioned even beside one absurd row.
    y0, x0 = float(np.median(y)), float(np.median(x))
    columns = [(y - y0) / scale] if use_y else []
    columns += [(x - x0) / scale] if use_x else []
    X = np.column_stack(columns + [np.ones(n)])
    beta = _lmeds(X, h)
    beta = _irls(X, h, beta) if beta is not None else None
    result["model"] = model
    if beta is None:
        result.update(status="unreliable", reasons=["fit_failed"])
        return result
    a = beta[0] / scale if use_y else 0.0
    b = beta[1] / scale if use_x else 0.0
    coef = [round(float(v), 6) + 0.0 for v in (a, b, beta[-1] - a * y0 - b * x0)]
    if not all(math.isfinite(v) for v in coef):
        result.update(status="unreliable", reasons=["fit_failed"])
        return result
    r = _relative(h, coef[0] * y + coef[1] * x + coef[2])
    inliers = np.abs(r) < POLICY["tukey_c"]
    count = int(inliers.sum())
    mad = round(float(np.median(np.abs(r[inliers]))), 4) if count else None
    corners = [coef[0] * yy + coef[1] * xx + coef[2] for yy in (y.min(), y.max()) for xx in (x.min(), x.max())]
    reasons = []
    if count < POLICY["min_inlier_fraction"] * n:
        reasons.append("low_inlier_fraction")
    limit = POLICY["max_residual_mad_constant" if model == "constant" else "max_residual_mad"]
    if mad is None or mad > limit:
        reasons.append("constant_model_residual_too_large" if model == "constant" else "residual_mad_above_limit")
    if not min(corners) > 0:
        reasons.append("nonpositive_prediction_in_range")
    result.update(status="unreliable" if reasons else "ok", coef=coef, residual_mad=mad,
                  inliers=count, inlier_fraction=round(count / n, 4), reasons=reasons)
    return result


def _calibrated_height(geom, calibration):
    """Return (h_pred, reason); reason set when the calibration cannot be applied here."""
    coef = _seq(calibration.get("coef"))
    coef = [_finite(v) for v in coef] if coef is not None and len(coef) == 3 else None
    if coef is None or None in coef:
        return None, "invalid_calibration"
    fy, fx = _finite(geom.get("foot_y")), _finite(geom.get("foot_x"))
    if fx is None and coef[1] != 0:
        return None, "missing_foot_x"
    pred = coef[0] * fy + coef[1] * (fx or 0.0) + coef[2]
    if not math.isfinite(pred):
        return None, "invalid_calibration"
    if pred <= 0:
        return None, "nonpositive_calibrated_height"
    margin = POLICY["calibration_extrapolation_per_height"] * pred
    for value, key, used in ((fy, "foot_y_range", True), (fx, "foot_x_range", coef[1] != 0)):
        bounds = _seq(calibration.get(key))
        bounds = [_finite(v) for v in bounds] if bounds is not None and len(bounds) == 2 else None
        if used and bounds and None not in bounds and not bounds[0] - margin <= value <= bounds[1] + margin:
            return None, "outside_calibrated_range"
    return pred, None


def expected_height(geom, calibration):
    """Calibrated standing-adult height (pixels) at this person's foot position, or None.

    Only for an ``ok`` calibration and a foot position inside the calibrated
    range (plus the usual extrapolation margin); used for distance-like
    decisions such as the near-zone count, never for age by itself.
    """
    if not isinstance(calibration, dict) or calibration.get("status") != "ok" or not isinstance(geom, dict):
        return None
    try:
        pred, reason = _calibrated_height(geom, calibration)
    except (TypeError, ValueError):
        return None
    return round(pred, 3) if reason is None and pred is not None else None


def _peer_heights(geom, peers, foot_y, window):
    heights, image_id, person_id = [], _ident(geom.get("image_id")), _ident(geom.get("person_id"))
    for peer in _items(peers):
        if not isinstance(peer, dict) or peer is geom or not _is_true(peer.get("complete")):
            continue
        if _ident(peer.get("image_id")) != image_id:
            continue
        if person_id is not None and _ident(peer.get("person_id")) == person_id:
            continue
        ph, py = _finite(peer.get("height_px")), _finite(peer.get("foot_y"))
        if ph is not None and py is not None and ph > 0 and abs(py - foot_y) <= window:
            heights.append(ph)
    return heights


def classify(geom, calibration, peers):
    """Adult/child/unknown from camera calibration, else same-image relative height."""
    geom = geom if isinstance(geom, dict) else {}
    result = {"age": "unknown", "method": "none", "ratio": None, "reference_px": None, "reasons": []}
    reasons = result["reasons"]
    h, foot_y = _finite(geom.get("height_px")), _finite(geom.get("foot_y"))
    if not _is_true(geom.get("complete")) or h is None or h <= 0 or foot_y is None:
        reasons.append("incomplete_person")
        return result

    def decide(method, reference, child_max, adult_min, low, high, adult_allowed=True):
        ratio = h / reference
        ratio = round(ratio, 4) if math.isfinite(ratio) else None
        result.update(method=method, ratio=ratio, reference_px=round(reference, 2))
        if ratio is None or not low <= ratio <= high:
            reasons.append("implausible_ratio")
        elif ratio <= child_max:
            result["age"] = "child"
        elif ratio < adult_min:
            reasons.append("ratio_between_thresholds")
        elif adult_allowed:
            result["age"] = "adult"
        else:
            reasons.append("no_verified_adult_reference")
        return result

    status = calibration.get("status") if isinstance(calibration, dict) else None
    status = str(status) if isinstance(status, str) else None
    if status == "ok":
        pred, problem = _calibrated_height(geom, calibration)
        if problem is None:
            return decide("camera_calibration", pred, POLICY["calibration_child_max"],
                          POLICY["calibration_adult_min"], POLICY["calibration_ratio_min"],
                          POLICY["calibration_ratio_max"])
        reasons.append(problem)
    else:
        reasons.append("calibration_" + status if status else "no_calibration")
    image_height = _finite(geom.get("image_height"))
    if image_height is None or image_height <= 0:
        reasons.append("no_image_height_for_peers")
        return result
    heights = _peer_heights(geom, peers, foot_y, POLICY["peer_foot_window_per_image_height"] * image_height)
    if not heights:
        reasons.append("no_eligible_peers")
        return result
    return decide("relative_height", max(heights), POLICY["relative_child_max"],
                  POLICY["relative_adult_min"], POLICY["relative_ratio_min"], POLICY["relative_ratio_max"],
                  adult_allowed=False)
