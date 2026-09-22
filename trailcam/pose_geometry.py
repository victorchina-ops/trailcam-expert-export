"""Conservative, uncalibrated 2-D pose orientation evidence for manual evaluation.

This is a geometry heuristic, not a pretrained orientation classifier. Labels
describe apparent body facing, never motion. Thresholds were chosen before the
full run and must be evaluated against held-out manual labels.
"""
from __future__ import annotations

import math

RULE_VERSION = "pose_geometry_v1"
KEYPOINT_NAMES = ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
                  "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                  "left_wrist", "right_wrist", "left_hip", "right_hip",
                  "left_knee", "right_knee", "left_ankle", "right_ankle"]
THRESHOLDS = {
    "keypoint_confidence": 0.60, "person_confidence": 0.40,
    "minimum_resized_person_height": 72, "minimum_resized_person_width": 12,
    "minimum_resized_torso_height": 20, "minimum_resized_lateral_span": 4,
    "maximum_torso_tilt": 0.50, "maximum_shoulder_slope": 0.35,
    "wide_shoulders_per_torso": 0.55, "wide_hips_per_torso": 0.30,
    "side_shoulders_per_torso": 0.45, "side_hips_per_torso": 0.35,
    "face_eye_separation_per_torso": 0.08,
    "profile_nose_ear_separation_per_torso": 0.12,
}


def orientation_from_keypoints(keypoints, xyxy, image_width, image_height,
                               detection_confidence=1.0, imgsz=960):
    result = {"orientation": "unknown", "facing_direction": "unclear",
              "evidence_score": None, "reason_codes": [], "features": {},
              "rule_version": RULE_VERSION,
              "evidence_score_interpretation": "minimum input keypoint confidence; not a calibrated orientation probability"}

    def unknown(reason):
        result["reason_codes"].append(reason)
        return result

    if len(keypoints) != 17 or any(len(k) != 3 for k in keypoints):
        return unknown("invalid_keypoint_shape")
    if not all(math.isfinite(float(v)) for k in keypoints for v in k):
        return unknown("nonfinite_keypoints")
    x1, y1, x2, y2 = xyxy
    width, height = x2 - x1, y2 - y1
    if width <= 0 or height <= 0 or image_width <= 0 or image_height <= 0:
        return unknown("invalid_box_or_image_dimensions")
    scale = imgsz / max(image_width, image_height)
    result["features"].update(resized_person_height=height * scale,
                              resized_person_width=width * scale)
    if detection_confidence < THRESHOLDS["person_confidence"]:
        return unknown("low_person_confidence")
    if height * scale < 72 or width * scale < 12:
        return unknown("person_too_small_at_model_resolution")

    def good(index):
        k = keypoints[index]
        return k[2] >= .60 and x1 - .05 * width <= k[0] <= x2 + .05 * width and y1 - .05 * height <= k[1] <= y2 + .05 * height

    if not all(good(i) for i in (5, 6, 11, 12)):
        return unknown("insufficient_bilateral_torso_keypoints")
    ls, rs, lh, rh = [keypoints[i] for i in (5, 6, 11, 12)]
    sx, sy = (ls[0]+rs[0])/2, (ls[1]+rs[1])/2
    hx, hy = (lh[0]+rh[0])/2, (lh[1]+rh[1])/2
    torso = hy - sy
    if torso * scale < 20:
        return unknown("short_or_inverted_torso")
    shoulder_delta, hip_delta = ls[0] - rs[0], lh[0] - rh[0]
    shoulder_span, hip_span = abs(shoulder_delta), abs(hip_delta)
    result["features"].update(torso_height=torso, shoulder_delta_x=shoulder_delta,
                              hip_delta_x=hip_delta, shoulder_span_per_torso=shoulder_span/torso,
                              hip_span_per_torso=hip_span/torso,
                              torso_tilt=abs(sx-hx)/torso)
    if abs(sx-hx) > .5 * torso or abs(ls[1]-rs[1]) > .35 * torso:
        return unknown("tilted_or_nonstandard_torso")
    bilateral_face = (all(good(i) for i in (0, 1, 2))
                      and keypoints[1][0] > keypoints[2][0]
                      and keypoints[2][0] <= keypoints[0][0] <= keypoints[1][0]
                      and abs(keypoints[1][0]-keypoints[2][0]) >= max(.08*torso, 2/scale))
    result["features"]["bilateral_front_face_evidence"] = bilateral_face
    wide = (shoulder_span >= max(.55*torso, 4/scale)
            and hip_span >= max(.30*torso, 4/scale))
    ordered = shoulder_delta * hip_delta > 0
    if wide and ordered:
        if shoulder_delta > 0 and bilateral_face:
            result.update(orientation="front", facing_direction="toward",
                          evidence_score=min(keypoints[i][2] for i in (0, 1, 2, 5, 6, 11, 12)),
                          reason_codes=["wide_torso_front_anatomical_order_and_bilateral_face"])
            return result
        if shoulder_delta < 0 and not bilateral_face:
            # Face absence alone is explicitly insufficient: both reliable torso
            # segments must also have compatible anatomical rear ordering.
            result.update(orientation="back", facing_direction="away",
                          evidence_score=min(keypoints[i][2] for i in (5, 6, 11, 12)),
                          reason_codes=["wide_torso_rear_anatomical_order_no_conflicting_front_face"])
            return result
        return unknown("front_back_geometry_and_face_disagree_or_face_missing")
    narrow = shoulder_span <= .45*torso and hip_span <= .35*torso
    if not narrow:
        return unknown("oblique_or_conflicting_torso_geometry")
    if bilateral_face:
        return unknown("narrow_body_but_front_facing_head")
    if not good(0):
        return unknown("profile_without_reliable_nose")
    nose = keypoints[0]
    if abs(nose[0]-sx) > .65*torso or nose[1] >= sy:
        return unknown("head_body_geometry_mismatch")
    candidates = []
    for eye_i, ear_i in ((1, 3), (2, 4)):
        if not good(eye_i) or not good(ear_i):
            continue
        eye, ear = keypoints[eye_i], keypoints[ear_i]
        delta = nose[0]-ear[0]
        nose_eye_delta = nose[0]-eye[0]
        if abs(delta) < max(.12*torso, 3/scale):
            continue
        if delta * nose_eye_delta <= 0 or abs(nose_eye_delta) >= abs(delta):
            continue
        if max(nose[1], eye[1], ear[1]) >= sy or max(nose[1], eye[1], ear[1])-min(nose[1], eye[1], ear[1]) > .30*torso:
            continue
        candidates.append(("right" if delta > 0 else "left", eye_i, ear_i))
    if not candidates:
        return unknown("no_coherent_nose_eye_ear_profile")
    if len({c[0] for c in candidates}) != 1:
        return unknown("conflicting_profile_sides")
    evidence_indices = {0, 5, 6, 11, 12}
    for _, eye_i, ear_i in candidates:
        evidence_indices.update((eye_i, ear_i))
    result.update(orientation="side", facing_direction=candidates[0][0],
                  evidence_score=min(keypoints[i][2] for i in evidence_indices),
                  reason_codes=["narrow_torso_and_coherent_facial_profile"])
    return result


def box_iou(a, b):
    intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    area_a = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    area_b = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
    union = area_a+area_b-intersection
    return intersection/union if union else 0.0


def match_canonical(canonical, poses, min_iou=.5, min_margin=.1):
    """Mutual best, separated matches; deliberately abstain in crowded ambiguity."""
    output = []
    matrix = [[box_iou(c["xyxy"], p["xyxy"]) for p in poses] for c in canonical]
    for ci, c in enumerate(canonical):
        entry = {**c, "pose_detection_index": None, "match_status": "unmatched",
                 "match_iou": None, "match_margin": None,
                 "orientation": "unknown", "facing_direction": "unclear",
                 "evidence_score": None, "reason_codes": ["no_pose_box_match"]}
        row = sorted(enumerate(matrix[ci]), key=lambda t: (-t[1], t[0]))
        if not row or row[0][1] < min_iou:
            output.append(entry)
            continue
        pi, best = row[0]
        margin = best - (row[1][1] if len(row) > 1 else 0)
        column = sorted(((idx, matrix[idx][pi]) for idx in range(len(canonical))), key=lambda t: (-t[1], t[0]))
        column_margin = column[0][1] - (column[1][1] if len(column)>1 else 0)
        entry.update(match_iou=best, match_margin=min(margin, column_margin))
        if column[0][0] != ci or margin < min_margin or column_margin < min_margin:
            entry.update(match_status="ambiguous", reason_codes=["ambiguous_pose_box_association"])
        else:
            entry.update(pose_detection_index=pi, match_status="matched", **poses[pi]["orientation_evidence"])
        output.append(entry)
    return output
