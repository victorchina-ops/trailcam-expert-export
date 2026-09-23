"""Multi-expert candidate people and a post-processing acceptance rule.

Inference keeps every person box down to a low confidence floor. Boxes from
different experts are clustered into candidates (at most one box per expert),
so acceptance thresholds can be tuned later without re-running models. A
candidate is an image appearance, never an identity.
"""
from __future__ import annotations

import math

# Representative-box preference: YOLOE carries masks, pose carries keypoints.
# MegaDetector-compact is excluded from people: on the local labeled images it
# undercounts groups badly; it still corroborates animals and gates frames.
EXPERT_ORDER = ("yoloe", "pose", "yolo26n")
POLICY = {
    "raw_floor": 0.12,          # lowest stored confidence (inference)
    "match_iou": 0.45,          # cross-expert clustering (inference)
    # Acceptance (post-processing). YOLOE's one-to-many head at 1280 px is the
    # best single counter on the local labeled images; the smaller experts
    # need higher confidence alone and mainly corroborate.
    "accept_single": {"yoloe": 0.18, "pose": 0.45, "yolo26n": 0.45},
    "accept_pair": 0.12,        # two or more experts each at >= this confidence
    "pair_experts": ("yoloe", "pose", "yolo26n"),
    "strip_fraction": 0.5,      # box share inside the camera data strip
    "version": "roster_v2",
}


def box_iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def strip_fraction(box, strip, height):
    """Share of the box area inside the top/bottom camera data strip rows."""
    area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    if not area or not strip:
        return 0.0
    covered = 0.0
    top, bottom = int(strip.get("top", 0)), int(strip.get("bottom", 0))
    if top:
        covered += max(0.0, min(box[3], top) - box[1]) * (box[2] - box[0])
    if bottom:
        covered += max(0.0, box[3] - max(box[1], height - bottom)) * (box[2] - box[0])
    return min(1.0, covered / area)


def build_candidates(experts, strip=None, height=0, policy=POLICY):
    """Cluster person boxes from all experts; one member per expert per cluster.

    ``experts`` maps expert name to its detection list (dicts with ``label``,
    ``confidence``, ``xyxy``, ``detection_index``). Returns candidates sorted
    by descending confidence with stable ids.
    """
    boxes = []
    for name in EXPERT_ORDER:
        for index, d in enumerate(experts.get(name, {}).get("detections", [])):
            confidence = d.get("confidence")
            if d.get("label") != "person" or not isinstance(confidence, (int, float)) \
                    or not math.isfinite(confidence) or confidence < policy["raw_floor"]:
                continue
            if not all(math.isfinite(float(v)) for v in d["xyxy"]) or d["xyxy"][2] <= d["xyxy"][0] or d["xyxy"][3] <= d["xyxy"][1]:
                continue
            if strip_fraction(d["xyxy"], strip, height) >= policy["strip_fraction"]:
                continue
            boxes.append((float(d["confidence"]), EXPERT_ORDER.index(name), name, d.get("detection_index", index), list(d["xyxy"])))
    boxes.sort(key=lambda t: (-t[0], t[1], t[3]))
    clusters = []
    for confidence, _, name, index, box in boxes:
        best, best_iou = None, 0.0
        for cluster in clusters:
            if name in cluster["members"]:
                continue
            value = max(box_iou(box, m["xyxy"]) for m in cluster["members"].values())
            if value > best_iou:
                best, best_iou = cluster, value
        member = {"detection_index": index, "confidence": confidence, "xyxy": box}
        if best is not None and best_iou >= policy["match_iou"]:
            best["members"][name] = member
        else:
            clusters.append({"members": {name: member}})
    candidates = []
    for cluster in clusters:
        members = cluster["members"]
        representative = next(members[n] for n in EXPERT_ORDER if n in members)
        candidates.append({"xyxy": list(representative["xyxy"]), "bbox_xyxy": list(representative["xyxy"]),
                           "confidence": max(m["confidence"] for m in members.values()),
                           "members": {n: members[n] for n in EXPERT_ORDER if n in members}})
    candidates.sort(key=lambda c: (-c["confidence"], c["xyxy"][0], c["xyxy"][1]))
    for number, candidate in enumerate(candidates, 1):
        candidate["candidate_id"] = f"cand_{number:04d}"
    return candidates


def accept(candidate, policy=POLICY):
    """Return (accepted, reason) under the transparent acceptance rule."""
    members = candidate.get("members", {})
    single = policy["accept_single"]
    strong = [n for n, m in members.items()
              if m["confidence"] >= (single.get(n, 1.1) if isinstance(single, dict) else single)]
    if strong:
        return True, "single_expert_confident:" + "+".join(strong)
    paired = [n for n, m in members.items() if n in policy["pair_experts"] and m["confidence"] >= policy["accept_pair"]]
    if len(paired) >= 2:
        return True, "multi_expert_agreement:" + "+".join(paired)
    return False, "insufficient_confidence_or_support"
