"""Fake v2 inference results for pipeline tests: no model files, no GPU.

A *scene* describes what the (fake) experts see in one photograph: people and
objects with a box and the raw confidence each expert reports. ``write_image``
renders the scene (coloured boxes on a plain background, optional black camera
data strip, optional EXIF capture time) so the real image, metadata and
appearance code have something to read. ``build_result`` mirrors
``trailcam.vision.VisionEngine.analyze``: expert detections at raw confidence
(YOLOE masks, pose keypoints + orientation evidence), multi-expert roster
candidates from the real ``roster.build_candidates`` (none for a gated frame),
appearance descriptors from the real pixels, capture metadata and data strip from the real helpers,
empty-frame gating, and the transient ``_decoded_image``.

``FakeVisionEngine`` / ``FakeAttributeEngine`` replace the model-backed engines
with the same ``analyze`` signatures.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from trailcam import roster
from trailcam.appearance import descriptor
from trailcam.attributes import MODEL_NAME, NOTES, derive_attributes, unknown_attributes
from trailcam.imageinfo import detect_data_strip, read_capture_metadata
from trailcam.pose_geometry import KEYPOINT_NAMES, orientation_from_keypoints
from trailcam.vision import (GATE_CONFIDENCE, LABELS, MD_LABELS, PROFILES, WEIGHTS,
                             compact_polygon, count_objects)

EXPERTS = ("yolo26n", "megadetector", "yoloe", "pose")  # inference order in VisionEngine.analyze
YOLOE_NAMES = dict(enumerate(LABELS))
YOLO26N_NAMES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck",
                 16: "dog", 24: "backpack", 26: "handbag", 28: "suitcase"}
POSE_NAMES = {0: "person"}
MD_FOR_LABEL = {"person": "person", "dog": "animal"}  # anything else MegaDetector calls "vehicle"
PERSON_EXPERTS = {"yoloe": 0.9, "pose": 0.85, "yolo26n": 0.8, "megadetector": 0.7}
BACKGROUND = (90, 110, 70)
SIZE = (640, 480)
SPEED_MS = {"preprocess": 1.0, "inference": 2.0, "postprocess": 0.5}


# --------------------------------------------------------------------------- scenes

def person(box, experts=None, facing="toward", color=(200, 40, 40), keypoint_confidence=0.9):
    """A person seen by ``experts`` ({name: raw confidence}); pose adds keypoints."""
    return {"kind": "person", "label": "person", "box": [float(v) for v in box],
            "experts": dict(PERSON_EXPERTS if experts is None else experts),
            "facing": facing, "color": tuple(color), "keypoint_confidence": keypoint_confidence}


def thing(label, box, experts, color=(40, 40, 200)):
    """A non-person object; ``label`` uses the YOLOE vocabulary (vision.LABELS)."""
    return {"kind": "object", "label": label, "box": [float(v) for v in box],
            "experts": dict(experts), "color": tuple(color)}


def scene(*items, size=SIZE, background=BACKGROUND, strip_rows=0, decode_scale=1, corrupt_postprocess=False):
    """``strip_rows`` draws a black camera data strip at the bottom. ``decode_scale`` 2
    mimics a JPEG draft decode at half size. ``corrupt_postprocess`` adds a YOLOE
    suitcase without a detection index: it passes the cache check but fails
    post-processing (fusion) whenever the photo also shows an accepted person."""
    return {"items": list(items), "size": tuple(size), "background": tuple(background),
            "strip_rows": int(strip_rows), "decode_scale": decode_scale,
            "corrupt_postprocess": bool(corrupt_postprocess)}


def walker(x, y=100.0, height=200.0, **kwargs):
    """A standing person whose box is ``0.4 * height`` wide, top-left at (x, y)."""
    return person([x, y, x + 0.4 * height, y + height], **kwargs)


def render(spec):
    """PIL RGB image of the scene (people/objects as filled boxes)."""
    image = Image.new("RGB", spec["size"], spec["background"])
    draw = ImageDraw.Draw(image)
    for item in spec["items"]:
        x1, y1, x2, y2 = item["box"]
        draw.rectangle([round(x1), round(y1), round(x2) - 1, round(y2) - 1], fill=item["color"])
    if spec["strip_rows"]:
        width, height = spec["size"]
        draw.rectangle([0, height - spec["strip_rows"], width, height], fill=(0, 0, 0))
    return image


def write_image(path, spec, capture_time=None, make="TestCam", model="TC-1"):
    """Render ``spec`` to ``path`` (JPEG or PNG by suffix) with optional EXIF.

    ``capture_time`` is ``"YYYY:MM:DD HH:MM:SS"`` (EXIF DateTimeOriginal)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exif = Image.Exif()
    if make:
        exif[0x010F] = make
    if model:
        exif[0x0110] = model
    if capture_time:
        exif.get_ifd(0x8769)[0x9003] = capture_time
    image = render(spec)
    if path.suffix.lower() in (".jpg", ".jpeg"):
        image.save(path, "JPEG", quality=95, exif=exif.tobytes())
    else:
        image.save(path, exif=exif.tobytes())
    return path


# ------------------------------------------------------------------- expert outputs

def standing_keypoints(box, facing="toward", confidence=0.9):
    """17 COCO [x, y, conf] triples of an upright, fully visible person.

    ``toward``: confident face, anatomical left on the image right (pose
    geometry reads front/toward). ``away``: mirrored torso, no face (back/away).
    Geometry is complete for age: crown, ankles, torso and extended legs.
    """
    x1, y1, x2, y2 = box
    w, h, cx = x2 - x1, y2 - y1, (x1 + x2) / 2
    face = confidence if facing == "toward" else 0.05
    side = -1.0 if facing == "away" else 1.0

    def at(fx, fy, conf=confidence):
        return [cx + side * fx * w, y1 + fy * h, conf]

    return [at(0, .10, face), at(.05, .08, face), at(-.05, .08, face), at(.10, .09, face), at(-.10, .09, face),
            at(.25, .22), at(-.25, .22), at(.30, .38), at(-.30, .38), at(.30, .50), at(-.30, .50),
            at(.15, .52), at(-.15, .52), at(.12, .74), at(-.12, .74), at(.10, .95), at(-.10, .95)]


def _label_for(expert, label):
    """(class_id, label) as this expert names ``label``; None when it cannot see it."""
    if expert == "yoloe":
        return (LABELS.index(label), label) if label in LABELS else None
    if expert == "yolo26n":
        ids = {name: index for index, name in YOLO26N_NAMES.items()}
        return (ids[label], label) if label in ids else None
    if expert == "megadetector":
        name = MD_FOR_LABEL.get(label, "vehicle")
        return next(k for k, v in MD_LABELS.items() if v == name), name
    return (0, "person") if label == "person" else None


def expert_detections(spec, expert, width, height, pose_size):
    """One expert's raw detections for ``spec``, sorted by confidence like Ultralytics."""
    found = []
    for order, item in enumerate(spec["items"]):
        confidence = item["experts"].get(expert)
        named = _label_for(expert, item["label"])
        if confidence is None or named is None or confidence < roster.POLICY["raw_floor"]:
            continue
        found.append((-float(confidence), order, item, named))
    detections = []
    for index, (negative, _, item, (class_id, label)) in enumerate(sorted(found, key=lambda t: t[:2])):
        box = [float(v) for v in item["box"]]
        detection = {"detection_index": index, "class_id": class_id, "label": label,
                     "confidence": -negative, "xyxy": box, "bbox_xyxy": list(box)}
        if expert == "yoloe":
            x1, y1, x2, y2 = box
            detection["mask_polygon_xy"] = compact_polygon([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], width, height)
        elif expert == "pose":
            keypoints = standing_keypoints(box, item.get("facing") or "toward", item["keypoint_confidence"])
            detection.update(pose_detection_index=index, keypoints=keypoints,
                             orientation_evidence=orientation_from_keypoints(
                                 keypoints, box, width, height, detection["confidence"], pose_size))
        detections.append(detection)
    return detections


def _gate_open(experts, strip, height):
    for key in ("yolo26n", "megadetector"):
        for d in experts[key]["detections"]:
            if d["confidence"] >= GATE_CONFIDENCE and roster.strip_fraction(d["xyxy"], strip, height) < .5:
                return True
    return False


def build_result(spec, rgb_image, metadata, profile="standard", empty_frame_gate=True, device="cpu"):
    """A ``VisionEngine.analyze`` result for ``spec`` (original-image pixels).

    ``rgb_image`` is the decoded image handed on as ``_decoded_image``; when it
    is smaller than the scene, ``analysis_scale`` records the ratio (a draft
    decode) while every coordinate stays in original pixels.
    """
    sizes = dict(PROFILES[profile])
    full_w, full_h = spec["size"]
    scale = full_w / rgb_image.width
    rgb = np.asarray(rgb_image.resize(spec["size"]) if scale != 1 else rgb_image)
    strip = detect_data_strip(rgb)
    experts = {}
    for key in EXPERTS:
        if key in ("yoloe", "pose") and empty_frame_gate and not _gate_open(experts, strip, full_h):
            experts[key] = {"counts": count_objects([], key), "detections": [], "skipped": "empty_frame_gate",
                            "device": device, "inference_seconds": 0.0, "checkpoint": WEIGHTS[key],
                            "imgsz": sizes[key]}
            continue
        detections = expert_detections(spec, key, full_w, full_h, sizes["pose"])
        extra = {}
        if key == "yoloe":
            extra["mask_representation"] = "exterior or merged contour, rounded to 0.1 pixel; holes not retained"
        elif key == "pose":
            extra["keypoint_names"] = KEYPOINT_NAMES
        experts[key] = {"counts": count_objects(detections, key), "detections": detections, "device": device,
                        "inference_seconds": 0.001, "checkpoint": WEIGHTS[key], "imgsz": sizes[key],
                        "predict_speed_ms": dict(SPEED_MS), **extra}
    if spec["corrupt_postprocess"]:
        # Well formed for the (v2, stricter) cache check - label, confidence and
        # box are valid - but without the detection_index fusion needs to tell
        # loose luggage from carried bags.
        experts["yoloe"]["detections"].append({"class_id": LABELS.index("suitcase"), "label": "suitcase",
                                               "confidence": 0.9, "xyxy": [2.0, 2.0, 30.0, 40.0],
                                               "bbox_xyxy": [2.0, 2.0, 30.0, 40.0], "mask_polygon_xy": []})
    gated = any(e.get("skipped") for e in experts.values())
    persons = []
    for candidate in ([] if gated else roster.build_candidates(experts, strip, full_h)):  # as VisionEngine
        members = candidate["members"]
        entry = {"person_id": candidate["candidate_id"], **candidate, "keypoints": None,
                 "pose_confidence": None, "orientation_evidence": None, "mask_polygon_xy": None}
        if "pose" in members:
            pose = experts["pose"]["detections"][members["pose"]["detection_index"]]
            entry.update(keypoints=pose["keypoints"], pose_confidence=pose["confidence"],
                         orientation_evidence=pose["orientation_evidence"])
        if "yoloe" in members:
            entry["mask_polygon_xy"] = experts["yoloe"]["detections"][members["yoloe"]["detection_index"]]["mask_polygon_xy"]
        entry["appearance"] = descriptor(rgb, entry["xyxy"], entry["mask_polygon_xy"], device="cpu")
        persons.append(entry)
    timings = {"decode_seconds": 0.001, **{key + "_seconds": 0.001 for key in EXPERTS
                                           if not experts[key].get("skipped")}, "vision_total_seconds": 0.005}
    return {"width": full_w, "height": full_h, "analysis_scale": scale, "device": device, "profile": profile,
            "devices_used": sorted({e["device"] for e in experts.values()}), "warnings": [],
            "metadata": metadata, "data_strip": strip, "gated_empty": gated,
            "timings": timings, "experts": experts, "persons": persons, "_decoded_image": rgb_image}


def result_for(spec, metadata=None, **kwargs):
    """``build_result`` without a file: renders the scene in memory."""
    info = {"capture_time": None, "time_source": None, "subsec": None, "camera_make": None,
            "camera_model": None, "sequence_number": None, "clock_suspect": False}
    info.update(metadata or {})
    return build_result(spec, render(spec), info, **kwargs)


# ------------------------------------------------------------------------ attributes

def attribute_scores(orientation):
    """Native PP-LCNet scores: adult, no backpack, orientation agreeing with pose."""
    scores = {"handbag": 0.05, "shoulder_bag": 0.05, "backpack": 0.05, "age_under18": 0.05,
              "age_18_60": 0.92, "age_over60": 0.03, "native_female": 0.5,
              "front": 0.3, "side": 0.3, "back": 0.3}
    if orientation in ("front", "side", "back"):
        scores.update(front=0.03, side=0.03, back=0.03)
        scores[orientation] = 0.92
    return scores


def fake_attributes(persons, failed=(), device="cpu"):
    """``AttributeEngine.analyze`` output; person ids (or indices) in ``failed`` error."""
    results = []
    for index, entry in enumerate(persons):
        if entry["person_id"] in failed or index in failed:
            attributes = unknown_attributes("RuntimeError: failed crop")
        else:
            x1, y1, x2, y2 = entry["bbox_xyxy"]
            evidence = entry.get("orientation_evidence") or {}
            attributes = derive_attributes(attribute_scores(evidence.get("orientation")), x2 - x1, y2 - y1)
            attributes.update(status="ok", error=None, crop_xyxy=[round(v, 1) for v in entry["bbox_xyxy"]])
        attributes.update(seconds=0.001, device=device)
        results.append({**entry, "attributes": attributes})
    errors = sum(r["attributes"]["status"] != "ok" for r in results)
    return {"model": MODEL_NAME, "device": device, "requested_device": "auto", "seconds": 0.002,
            "persons": results, "status": "partial_error" if errors else "ok", "people_failed": errors,
            "fallback_reasons": [], "notes": list(NOTES)}


def with_attributes(result, failed=()):
    """Attach fake attributes the way ``__main__.run`` does (drops ``_decoded_image``)."""
    result.pop("_decoded_image", None)
    by_id = {p["person_id"]: p["attributes"] for p in fake_attributes(result["persons"], failed)["persons"]}
    for entry in result["persons"]:
        entry["attributes"] = by_id[entry["person_id"]]
    errors = [p["attributes"].get("error") for p in result["persons"] if p["attributes"]["status"] != "ok"]
    result.update(status="partial_error" if errors else "ok", error="; ".join(errors) if errors else None,
                  attribute_device="cpu", analyzed_at_utc="2026-05-01T00:00:00+00:00")
    return result


# ------------------------------------------------------------------------ engines

class FakeVisionEngine:
    """``VisionEngine`` stand-in: scenes instead of models.

    ``scenes`` maps the image path relative to ``root`` (POSIX; the file name
    when ``root`` is None) to a scene; unknown images show an empty scene.
    ``calls`` records the same keys in call order.
    """

    def __init__(self, scenes=None, root=None, profile="standard", empty_frame_gate=True, device="cpu"):
        self.scenes = {} if scenes is None else scenes
        self.root = None if root is None else Path(root)
        self.profile, self.empty_frame_gate, self.device = profile, empty_frame_gate, device
        self.calls = []

    def key(self, image_path):
        path = Path(image_path)
        return path.relative_to(self.root).as_posix() if self.root else path.name

    def analyze(self, image_path):
        path = Path(image_path)
        key = self.key(path)
        self.calls.append(key)
        with Image.open(path) as source:  # corrupt files raise here, as in the real engine
            metadata = read_capture_metadata(source, path)
            rgb_image = ImageOps.exif_transpose(source).convert("RGB")
        spec = self.scenes.get(key) or scene(size=rgb_image.size)
        if spec["decode_scale"] != 1:
            rgb_image = rgb_image.resize((rgb_image.width // spec["decode_scale"],
                                          rgb_image.height // spec["decode_scale"]))
        return build_result(spec, rgb_image, metadata, self.profile, self.empty_frame_gate, self.device)


class FakeAttributeEngine:
    """``AttributeEngine`` stand-in; ``failed`` person ids/indices report an error."""

    def __init__(self, failed=(), device="cpu"):
        self.failed, self.device = set(failed), device
        self.calls = []

    def analyze(self, image_path, persons, image=None, scale=1.0):
        self.calls.append({"name": Path(image_path).name, "person_ids": [p["person_id"] for p in persons],
                           "image_size": None if image is None else tuple(image.size), "scale": scale})
        return fake_attributes(persons, self.failed, self.device)
