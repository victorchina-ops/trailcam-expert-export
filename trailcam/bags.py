"""Large-bag evidence from bag boxes measured against the associated body.

Pure post-processing: no model loading and no device code, so results are
identical on CUDA and CPU-only machines. A bag is "large" (trekking/frame
pack, suitcase, duffel) or a daypack; ``None`` means the evidence cannot
separate the two. Spatial association supports, but never proves, carrying.
"""
from __future__ import annotations

import math
import operator

RULE_VERSION = "bags_v2"
LUGGAGE_LABELS = ("suitcase", "duffel bag")
KEYPOINTS = {"left_shoulder": 5, "right_shoulder": 6, "left_hip": 11, "right_hip": 12}
# bags_v2 features (priors from anthropometry and retail pack sizes; T is the
# shoulder-to-hip line distance, about 0.29 x stature):
#   F1 = bag height / T            daypack ~0.98-1.22, trekking pack ~1.56-1.83
#   F2 = (shoulder_y - bag top) / T bag rising above the shoulders
#   F3 = bag height / person height daypack ~0.28-0.35, trekking pack ~0.45-0.53
# A large verdict needs two agreeing cues or a full-body height ratio; the v1
# single-cue rules labelled most daypacks as large on the local labels.
POLICY = {
    "keypoint_confidence": 0.50,         # shoulder/hip keypoints used at >= this
    "luggage_confidence": 0.35,          # suitcase / duffel bag considered at >= this
    "luggage_min_torso_ratio": 0.80,     # luggage long side >= this x T to count as large
    "backpack_confidence": 0.25,         # weaker backpack detections are ignored
    "large_torso_ratio": 1.40,           # F1 (with F2) for large
    "large_top_above_shoulder": 0.25,    # F2 (with F1) for large
    "large_height_ratio": 0.50,          # F3 alone for large, only with the full body visible
    "full_body_per_torso": 2.6,          # person height >= this x T means legs are visible
    "daypack_torso_ratio": 1.15,         # F1 <= this and F2 <= daypack_top -> daypack
    "daypack_top_above_shoulder": 0.15,
    "daypack_height_ratio": 0.30,        # F3 <= this -> daypack when no torso is measurable
    "min_torso_per_person_height": 0.15, # shorter torso (bent/lying, bad keypoints) unused
    "duplicate_iou": 0.50,               # overlapping bag boxes are one physical bag
    "version": RULE_VERSION,
}
# Values are reported *and* compared at this precision: thresholds stay
# inclusive for float32 detector outputs (float32 .35 = .3499999940) and the
# rounded features always reproduce the verdict.
_DECIMALS = 6
_RANK = {"large": 0, "uncertain": 1, "daypack": 2}


def _is_bool(value):
    return type(value).__name__ in ("bool", "bool_")  # Python and numpy bools


def _num(value):
    """Finite float from a numeric scalar; None for bools, arrays and junk."""
    if value is None or _is_bool(value) or getattr(value, "ndim", 0):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _round(value):
    """Reporting precision; non-finite values become None (strict JSON)."""
    return round(float(value), _DECIMALS) if value is not None and math.isfinite(value) else None


def _at_least(value, threshold):
    return value is not None and round(value, _DECIMALS) >= threshold


def _items(value):
    """Items of a list-like input; [] for None, None when not list-like."""
    if value is None:
        return []
    if isinstance(value, (str, bytes, dict)):
        return None
    try:
        return list(value)
    except TypeError:
        return None


def _box(value):
    """Four finite coordinates (rounded) with positive, finite size, else None."""
    values = _items(value)
    if not values or len(values) != 4:
        return None
    box = [_round(_num(v)) for v in values]
    if None in box or not (0 < box[2] - box[0] < math.inf and 0 < box[3] - box[1] < math.inf):
        return None
    return box


def _bag_box(bag):
    box = bag.get("box")
    return _box(box if box is not None else bag.get("xyxy"))


def _iou(a, b):
    width, height = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    if width <= 0 or height <= 0:
        return 0.0
    inter = width * height
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _same_bag(a, b):
    """IoU >= POLICY["duplicate_iou"]: two boxes of one physical bag."""
    return _at_least(_iou(a, b), POLICY["duplicate_iou"])


def _label(bag):
    return " ".join(str(bag.get("label", "")).replace("_", " ").lower().split())


def _triples(keypoints):
    """17 ``[x, y, conf]`` rows of finite floats or None, else None (bad shape)."""
    rows = _items(keypoints)
    if rows is None or len(rows) != 17:
        return None
    out = []
    for row in rows:
        if isinstance(row, (dict, set, frozenset)):
            return None
        values = _items(row)
        if values is None or len(values) < 3:
            return None
        out.append([_num(v) for v in values[:3]])
    return out


def _mean(values):
    return sum(v / len(values) for v in values) if values else None  # no overflow


def body_lines(keypoints):
    """Shoulder/hip line y from COCO keypoints 5/6 and 11/12 at conf >= .4.

    Returns ``{"shoulder_y", "hip_y", "torso_px", "used", "reasons"}``; each
    line is the mean y of its usable keypoints. The torso needs both lines
    with hips below shoulders; an inverted torso discards both lines.
    """
    out = {"shoulder_y": None, "hip_y": None, "torso_px": None, "used": [], "reasons": []}
    if keypoints is None:
        out["reasons"].append("keypoints_missing")
        return out
    rows = _triples(keypoints)
    if rows is None:
        out["reasons"].append("keypoints_invalid_shape")
        return out
    ys = {}
    for name, index in KEYPOINTS.items():
        x, y, conf = rows[index]
        if x is not None and y is not None and _at_least(conf, POLICY["keypoint_confidence"]):
            ys[name] = y
            out["used"].append(name)
    shoulder_y = _mean([ys[n] for n in ("left_shoulder", "right_shoulder") if n in ys])
    hip_y = _mean([ys[n] for n in ("left_hip", "right_hip") if n in ys])
    torso = hip_y - shoulder_y if shoulder_y is not None and hip_y is not None else None
    if torso is not None and not 0 < torso < math.inf:
        out["used"] = []
        out["reasons"].append("torso_inverted_keypoints_discarded" if torso <= 0
                              else "torso_nonfinite_keypoints_discarded")
        return out
    out.update(shoulder_y=shoulder_y, hip_y=hip_y, torso_px=torso)
    if shoulder_y is None:
        out["reasons"].append("shoulders_not_visible")
    if hip_y is None:
        out["reasons"].append("hips_not_visible")
    return out


def _backpack_verdict(box, person_height, shoulder_y, torso):
    """Geometry rules for one backpack; returns (verdict, rule, features)."""
    bag_height = box[3] - box[1]
    height_ratio = bag_height / person_height if person_height else None
    torso_ratio = bag_height / torso if torso else None
    above_px = shoulder_y - box[1] if shoulder_y is not None else None
    above_torso = above_px / torso if above_px is not None and torso else None
    full_body = bool(torso and person_height
                     and round(person_height / torso, _DECIMALS) >= POLICY["full_body_per_torso"])
    features = {"bag_height_px": _round(bag_height), "height_ratio": _round(height_ratio),
                "torso_ratio": _round(torso_ratio), "top_above_shoulder_px": _round(above_px),
                "top_above_shoulder_per_torso": _round(above_torso), "full_body": full_body}
    if _at_least(torso_ratio, POLICY["large_torso_ratio"]) and _at_least(above_torso, POLICY["large_top_above_shoulder"]):
        return True, "backpack_tall_and_above_shoulders_large", features
    if full_body and _at_least(height_ratio, POLICY["large_height_ratio"]):
        return True, "backpack_full_body_height_ratio_large", features
    if torso_ratio is not None:
        if round(torso_ratio, _DECIMALS) <= POLICY["daypack_torso_ratio"] and (
                above_torso is None or round(above_torso, _DECIMALS) <= POLICY["daypack_top_above_shoulder"]):
            return False, "backpack_daypack", features
        return None, "backpack_uncertain_mid_size", features
    if height_ratio is None:
        return None, "backpack_uncertain_no_person_height", features
    if round(height_ratio, _DECIMALS) <= POLICY["daypack_height_ratio"]:
        return False, "backpack_daypack_height_only", features
    return None, "backpack_uncertain_no_torso", features


def classify_person_bags(person_box, keypoints, bags, assessable=False):
    """Large-bag state of one person from bags spatially associated with them.

    ``large_bag`` is True when any distinct bag is large, else None when any is
    uncertain, else False for daypacks only. With no usable bag it is False
    only when ``assessable`` is truthy (person visibly assessable), else None.
    Overlapping boxes (IoU >= .5) are one physical bag, counted once. Without
    shoulder/hip keypoints only the height-ratio rules apply.
    """
    reasons = []
    try:
        assessable = bool(assessable)
    except (TypeError, ValueError):
        assessable = False
        reasons.append("assessable_flag_invalid")
    pbox = _box(person_box)
    if pbox is None:
        reasons.append("person_box_invalid")
    person_height = pbox[3] - pbox[1] if pbox else None
    lines = body_lines(keypoints)
    reasons.extend(lines["reasons"])
    shoulder_y, torso = lines["shoulder_y"], lines["torso_px"]
    if torso is not None and person_height and \
            round(torso / person_height, _DECIMALS) < POLICY["min_torso_per_person_height"]:
        torso = None
        reasons.append("torso_too_short_for_rules")
    if torso is None and shoulder_y is None:
        reasons.append("height_ratio_rules_only")
    items = _items(bags)
    if items is None:
        reasons.append("bags_input_invalid_ignored")
        items = []
    per_bag = []
    for index, bag in enumerate(items):
        if not isinstance(bag, dict):
            reasons.append("bag_record_invalid_ignored")
            continue
        label, box = _label(bag), _bag_box(bag)
        conf, mask_area = _round(_num(bag.get("confidence"))), _num(bag.get("mask_area"))
        entry = {"index": index, "label": label, "confidence": conf, "box": box,
                 "mask_area": int(mask_area) if mask_area is not None else None,
                 "status": "ignored", "verdict": None, "rule": None, "duplicate_of": None}
        per_bag.append(entry)
        if label not in LUGGAGE_LABELS and label != "backpack":
            entry["rule"] = "unsupported_label_ignored"
        elif box is None:
            entry["rule"] = "bag_box_invalid_ignored"
        elif conf is None:
            entry["rule"] = "bag_confidence_missing_ignored"
        elif label in LUGGAGE_LABELS:
            long_side = max(box[2] - box[0], box[3] - box[1])
            size_ratio = _round(long_side / torso) if torso else None
            entry["luggage_size_per_torso"] = size_ratio
            if conf >= POLICY["luggage_confidence"] and size_ratio is not None \
                    and size_ratio < POLICY["luggage_min_torso_ratio"]:
                entry.update(status="uncertain", verdict=None, rule=label.replace(" ", "_") + "_small_for_luggage")
            elif conf >= POLICY["luggage_confidence"]:
                entry.update(status="large", verdict=True, rule=label.replace(" ", "_") + "_confident_large")
            else:
                entry["rule"] = label.replace(" ", "_") + "_low_confidence_ignored"
        elif conf < POLICY["backpack_confidence"]:
            entry["rule"] = "backpack_low_confidence_ignored"
        else:
            verdict, rule, features = _backpack_verdict(box, person_height, shoulder_y, torso)
            entry.update(features, verdict=verdict, rule=rule,
                         status={True: "large", False: "daypack", None: "uncertain"}[verdict])
    # One physical bag may arrive from several experts or labels: keep the most
    # decisive reading (large > uncertain > daypack), then higher confidence
    # (box, then input index, break ties so counts never depend on input order).
    kept = []
    for entry in sorted((e for e in per_bag if e["status"] != "ignored"),
                        key=lambda e: (_RANK[e["status"]], -e["confidence"], e["box"], e["index"])):
        twin = next((k for k in kept if _same_bag(k["box"], entry["box"])), None)
        if twin is None:
            kept.append(entry)
        else:
            entry["duplicate_of"] = twin["index"]
    for entry in per_bag:
        reasons.append(entry["rule"] if entry["duplicate_of"] is None else "duplicate_bag_box_merged")
    counts = {status: sum(e["status"] == status for e in kept) for status in _RANK}
    if kept:
        large_bag = True if counts["large"] else None if counts["uncertain"] else False
    else:
        large_bag = False if assessable else None
        reasons.append("no_usable_bag_person_assessable" if assessable
                       else "no_usable_bag_person_not_assessable")
    features = {"rule_version": RULE_VERSION, "person_height_px": _round(person_height),
                "shoulder_y": _round(shoulder_y), "hip_y": _round(lines["hip_y"]),
                "torso_px": _round(torso), "keypoints_used": lines["used"],
                "assessable": assessable, "bags": per_bag,
                "large_bag_count": counts["large"], "uncertain_bag_count": counts["uncertain"],
                "daypack_count": counts["daypack"]}
    return {"large_bag": large_bag, "reasons": list(dict.fromkeys(reasons)), "features": features}


def _count(value):
    """Non-negative integer count (numpy ints accepted, bools refused), else None."""
    if _is_bool(value):
        return None
    try:
        number = operator.index(value)
    except TypeError:
        return None
    return number if number >= 0 else None


def count_large_bags(persons_results, loose_luggage):
    """Image totals: large bags carried plus loose luggage, each bag once.

    Carried bags use each result's ``features`` counts; a bare ``large_bag``
    True counts one large, a bare None nothing (it may be a person with no bag).
    ``large_bags_uncertain`` counts undecided bags, never bagless people. Loose
    luggage: suitcase / duffel detections at conf >= .35, each counted once
    (IoU >= .5 is one bag). As in ``classify_person_bags`` the large reading
    absorbs overlapping ones: loose luggage on a carried large bag (from a
    result's ``features["bags"]``) is that bag, and a counted loose bag settles
    any carried uncertain reading it overlaps.
    """
    large = uncertain = 0
    carried = {"large": [], "uncertain": []}  # boxes of distinct carried bags
    for result in _items(persons_results) or ():
        if not isinstance(result, dict):
            continue
        state = result.get("large_bag")
        if _is_bool(state):
            state = bool(state)
        elif state is not None:
            continue  # not a large-bag verdict
        features = result.get("features")
        features = features if isinstance(features, dict) else {}
        if state is True:
            large += max(1, _count(features.get("large_bag_count")) or 0)
        uncertain += _count(features.get("uncertain_bag_count")) or 0
        for entry in _items(features.get("bags")) or ():
            if isinstance(entry, dict) and entry.get("status") in ("large", "uncertain") \
                    and entry.get("duplicate_of") is None:
                box = _box(entry.get("box"))
                if box is not None:
                    carried[entry["status"]].append(box)
    candidates = []
    for index, item in enumerate(_items(loose_luggage) or ()):
        if not isinstance(item, dict) or _label(item) not in LUGGAGE_LABELS:
            continue
        box, conf = _bag_box(item), _round(_num(item.get("confidence")))
        if box is not None and conf is not None and conf >= POLICY["luggage_confidence"]:
            candidates.append((-conf, box, index))
    kept, loose = list(carried["large"]), []
    for _, box, _ in sorted(candidates):
        if not any(_same_bag(box, other) for other in kept):
            kept.append(box)
            loose.append(box)
    settled = sum(any(_same_bag(box, other) for other in loose) for box in carried["uncertain"])
    return {"large_bags": large + len(loose), "large_bags_uncertain": max(0, uncertain - settled)}
