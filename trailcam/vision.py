"""Local object, instance-mask, and conservative pose experts.

Image coordinates are pixels in the EXIF-corrected image. A detection is an
appearance in one photograph, not a unique visitor or evidence of movement.
"""
from __future__ import annotations

from collections import Counter
import math
import os
from pathlib import Path
import time
import warnings

from .pose_geometry import KEYPOINT_NAMES, RULE_VERSION, match_canonical, orientation_from_keypoints

ENGINE_VERSION = "vision_v1"
LABELS = ["person", "bicycle", "motorcycle", "dog", "backpack", "baby stroller",
          "all-terrain vehicle", "utility terrain vehicle", "car", "truck", "bus",
          "golf cart", "tractor", "kick scooter", "suitcase", "duffel bag"]
COCO_CLASSES = [0, 1, 2, 3, 5, 7, 16, 24, 26, 28]
COUNT_FIELDS = ("people_total", "bicycles", "strollers", "motorcycles", "atv_utv",
                "other_vehicles", "dogs", "backpacks", "cars_trucks_buses", "kick_scooters")
WEIGHTS = {"yolo26n": "yolo26n.pt", "yoloe": "yoloe-26s-trail-prompts.pt",
           "megadetector": "MDV6-yolov10-c.pt", "pose": "yolo26n-pose.pt"}
VEHICLE_LABELS = {"car", "truck", "bus", "motorcycle", "all-terrain vehicle",
                  "utility terrain vehicle", "golf cart", "tractor"}


def area(box):
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def intersection(a, b):
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def iou(a, b):
    overlap = intersection(a, b)
    union = area(a) + area(b) - overlap
    return overlap / union if union else 0.0


def semantic_vehicle_dedup(detections):
    """Suppress only near-identical boxes competing for motor-vehicle labels."""
    accepted, suppressed = [], []
    candidates = [i for i, d in enumerate(detections) if d["label"] in VEHICLE_LABELS]
    for index in sorted(candidates, key=lambda i: (-detections[i]["confidence"], i)):
        strongest = max(((iou(detections[index]["xyxy"], detections[j]["xyxy"]), j)
                         for j in accepted), default=(0.0, -1))
        if strongest[0] >= .8:
            suppressed.append({**detections[index], "raw_detection_index": index,
                               "kept_raw_detection_index": strongest[1],
                               "iou_with_kept": strongest[0],
                               "suppression_reason": "overlapping_motor_vehicle_labels"})
        else:
            accepted.append(index)
    removed = {d["raw_detection_index"] for d in suppressed}
    return [d for i, d in enumerate(detections) if i not in removed], suppressed


def count_objects(detections, expert):
    """None denotes unsupported; zero denotes no detection for a supported class."""
    counts = Counter(d["label"] for d in detections)
    result = dict.fromkeys(COUNT_FIELDS)
    result["people_total"] = counts["person"]
    if expert == "megadetector":
        result.update(animals_total=counts["animal"], vehicles_total=counts["vehicle"])
        return result
    if expert == "pose":
        return result
    result.update(bicycles=counts["bicycle"], motorcycles=counts["motorcycle"],
                  dogs=counts["dog"], backpacks=counts["backpack"],
                  cars_trucks_buses=sum(counts[k] for k in ("car", "truck", "bus")))
    if expert == "yoloe":
        result.update(strollers=counts["baby stroller"],
                      atv_utv=sum(counts[k] for k in ("all-terrain vehicle", "utility terrain vehicle", "golf cart")),
                      other_vehicles=sum(counts[k] for k in ("car", "truck", "bus", "tractor")),
                      kick_scooters=counts["kick scooter"])
    return result


def compact_polygon(points, width, height):
    polygon = []
    for x, y in points:
        if not math.isfinite(float(x)) or not math.isfinite(float(y)):
            raise ValueError("Nonfinite segmentation coordinate")
        if x < -.1 or y < -.1 or x > width + .1 or y > height + .1:
            raise ValueError("Mask polygon outside image")
        pair = [round(min(max(float(x), 0.0), width), 1),
                round(min(max(float(y), 0.0), height), 1)]
        if not polygon or pair != polygon[-1]:
            polygon.append(pair)
    return polygon


def assign_bag(bag_box, people):
    """Spatial support only: this cannot establish that a person carries a bag."""
    candidates = []
    for person in people:
        box = person["xyxy"]
        fraction = intersection(bag_box, box) / area(bag_box) if area(bag_box) else 0.0
        diagonal = math.hypot(box[2] - box[0], box[3] - box[1])
        distance = math.hypot((bag_box[0] + bag_box[2] - box[0] - box[2]) / 2,
                              (bag_box[1] + bag_box[3] - box[1] - box[3]) / 2)
        normalized = distance / diagonal if diagonal else math.inf
        if fraction >= .10 and normalized <= 1.0:
            candidates.append({"person_id": person["person_id"],
                               "source_detection_index": person["source_detection_index"],
                               "backpack_box_fraction_inside_person_box": fraction,
                               "center_distance_divided_by_person_diagonal": normalized,
                               "spatial_score": .7 * fraction + .3 * max(0.0, 1.0 - normalized)})
    candidates.sort(key=lambda c: (-c["spatial_score"], c["source_detection_index"]))
    if not candidates:
        return {"status": "unassigned", "person_id": None, "candidates": []}
    best = candidates[0]
    margin = best["spatial_score"] - (candidates[1]["spatial_score"] if len(candidates) > 1 else 0.0)
    supported = (best["backpack_box_fraction_inside_person_box"] >= .5
                 and best["center_distance_divided_by_person_diagonal"] <= .75 and margin >= .15)
    return {"status": "spatial_support" if supported else "ambiguous",
            "person_id": best["person_id"] if supported else None,
            "best_score_margin": margin, "candidates": candidates}


def select_device(torch, requested="auto"):
    """Check an actual CUDA kernel, since driver discovery alone is insufficient."""
    requested = str(requested).lower()
    if requested == "cpu":
        return "cpu", None
    if requested in ("auto", "gpu", "cuda"):
        index = 0
    elif requested.isdigit():
        index = int(requested)
    elif requested.startswith("cuda:") and requested[5:].isdigit():
        index = int(requested[5:])
    else:
        raise ValueError("Device must be auto, cpu, gpu, or a CUDA index such as 0")
    try:
        if not torch.cuda.is_available():
            return "cpu", "CUDA is unavailable; running detector and pose experts on CPU."
        device = "cuda:" + str(index)
        probe = torch.ones((8, 8), device=device)
        (probe @ probe).sum().item()
        torch.cuda.synchronize(index)
        del probe
        return device, None
    except (RuntimeError, AssertionError, OSError) as exc:
        return "cpu", "CUDA kernel check failed; using CPU: " + str(exc)


def is_cuda_failure(exc):
    message = str(exc).lower()
    return any(token in message for token in ("cuda", "cudnn", "cublas", "out of memory",
                                              "no kernel image", "device-side assert"))


class VisionEngine:
    """Loads four small local model checkpoints once and processes individual images."""

    def __init__(self, models_dir: Path, device: str = "auto", threads: int = 8):
        started = time.perf_counter()
        if threads < 1:
            raise ValueError("threads must be at least 1")
        self.models_dir = Path(models_dir).resolve()
        missing = [name for name in WEIGHTS.values() if not (self.models_dir / name).is_file()]
        if missing:
            raise FileNotFoundError("Missing model files; run setup first: " + ", ".join(missing))
        os.environ.setdefault("OMP_NUM_THREADS", str(threads))
        os.environ.setdefault("MKL_NUM_THREADS", str(threads))
        import torch
        import ultralytics
        from ultralytics import YOLO, YOLOE
        self.torch = torch
        self._constructors = {"yolo26n": YOLO, "yoloe": YOLOE, "megadetector": YOLO, "pose": YOLO}
        torch.set_num_threads(threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass  # Another engine may already have started Torch parallel work.
        self.device, warning = select_device(torch, device)
        self.warnings = [warning] if warning else []
        if warning:
            warnings.warn(warning, RuntimeWarning, stacklevel=2)
        self.models = {}
        try:
            self._load_models()
        except RuntimeError as exc:
            if self.device == "cpu" or not is_cuda_failure(exc):
                raise
            self._fallback_cpu(exc)
        self.config = {"engine_version": ENGINE_VERSION, "torch_version": torch.__version__,
                       "ultralytics_version": ultralytics.__version__, "threads": threads,
                       "imgsz": 960, "megadetector_imgsz": 1280, "conf": .25, "iou": .5,
                       "half": False, "max_det": 300, "pose_rule_version": RULE_VERSION,
                       "labels": LABELS, "requested_device": str(device),
                       "coordinates": "EXIF-corrected original-image pixel coordinates",
                       "direction_semantics": "apparent body facing, never verified motion",
                       "mask_role": "same YOLOE expert; no independent mask vote"}
        self.startup_seconds = time.perf_counter() - started

    def _load_models(self):
        self.models = {key: constructor(str(self.models_dir / WEIGHTS[key])).to(self.device)
                       for key, constructor in self._constructors.items()}
        names = {int(k): str(v) for k, v in self.models["megadetector"].names.items()}
        if names != {0: "animal", 1: "person", 2: "vehicle"}:
            raise ValueError("Unexpected MegaDetector labels: " + repr(names))
        names = {int(k): str(v) for k, v in self.models["yoloe"].names.items()}
        if names != dict(enumerate(LABELS)):
            raise ValueError("YOLOE checkpoint prompts do not match the 16 trail categories; rerun setup")

    def _fallback_cpu(self, exc):
        self.device = "cpu"
        warning = "CUDA inference failed; remaining detector and pose work uses CPU: " + str(exc)
        self.warnings.append(warning)
        warnings.warn(warning, RuntimeWarning, stacklevel=2)
        self.models.clear()
        try:
            self.torch.cuda.empty_cache()
        except RuntimeError:
            pass
        self._load_models()

    def _predict(self, key, image):
        options = {"imgsz": 1280 if key == "megadetector" else 960,
                   "conf": .25, "iou": .5, "half": False, "max_det": 300,
                   "agnostic_nms": key == "yoloe", "verbose": False, "save": False}
        if key == "yolo26n":
            options["classes"] = COCO_CLASSES
        if key == "yoloe":
            options["retina_masks"] = False
        try:
            return self.models[key].predict(image, device=self.device, **options)[0]
        except RuntimeError as exc:
            if self.device == "cpu" or not is_cuda_failure(exc):
                raise
            self._fallback_cpu(exc)
            return self.models[key].predict(image, device="cpu", **options)[0]

    def _extract_detections(self, result):
        boxes = result.boxes.cpu()
        names = result.names
        return [{"detection_index": index, "class_id": int(class_id),
                 "label": str(names[int(class_id)]), "confidence": float(confidence),
                 "xyxy": [float(v) for v in box], "bbox_xyxy": [float(v) for v in box]}
                for index, (box, class_id, confidence) in enumerate(
                    zip(boxes.xyxy.tolist(), boxes.cls.tolist(), boxes.conf.tolist()))]

    def analyze(self, image_path: Path) -> dict:
        import numpy as np
        from PIL import Image, ImageOps
        started = time.perf_counter()
        with Image.open(image_path) as source:
            rgb = ImageOps.exif_transpose(source).convert("RGB")
            image = np.asarray(rgb)[:, :, ::-1].copy()
        height, width = image.shape[:2]
        timings = {"decode_seconds": time.perf_counter() - started}
        experts, persons = {}, []
        for key in ("yolo26n", "yoloe", "megadetector", "pose"):
            model_started = time.perf_counter()
            result = self._predict(key, image)
            detections = self._extract_detections(result)
            extra = {}
            if key == "yoloe":
                if detections and result.masks is None:
                    raise ValueError("YOLOE returned detections without instance masks")
                polygons = result.masks.xy if result.masks is not None else []
                masks = result.masks.data.cpu().numpy().astype(bool) if result.masks is not None else np.empty((0, 0, 0), dtype=bool)
                if len(polygons) != len(detections) or len(masks) != len(detections):
                    raise ValueError("YOLOE mask and detection instance identity mismatch")
                mask_areas = [int(mask.sum()) for mask in masks]
                for index, detection in enumerate(detections):
                    detection["mask_polygon_xy"] = compact_polygon(polygons[index], width, height)
                    detection["mask_area_input_pixels"] = mask_areas[index]
                detections, extra["suppressed_detections"] = semantic_vehicle_dedup(detections)
                for index, detection in enumerate(detections):
                    if detection["label"] == "person":
                        persons.append({"person_id": f"person_{len(persons) + 1:04d}",
                                        "source_detection_index": index,
                                        "xyxy": detection["xyxy"], "bbox_xyxy": detection["xyxy"],
                                        "confidence": detection["confidence"], "mask_backpack": None,
                                        "mask_backpack_status": "no_detected_association", "bag_associations": []})
                by_id = {p["person_id"]: p for p in persons}
                associations = []
                for index, bag in enumerate(detections):
                    if bag["label"] != "backpack":
                        continue
                    association = assign_bag(bag["xyxy"], persons)
                    association.update(backpack_detection_index=index, backpack_xyxy=bag["xyxy"],
                                       backpack_confidence=bag["confidence"],
                                       evidence_scope="spatial proximity and mask overlap; not proof of carrying")
                    for candidate in association["candidates"]:
                        person = by_id[candidate["person_id"]]
                        person_detection = detections[person["source_detection_index"]]
                        bag_index, person_index = bag["detection_index"], person_detection["detection_index"]
                        bag_area, person_area = mask_areas[bag_index], mask_areas[person_index]
                        candidate.update(backpack_mask_fraction_overlapping_person_mask=(
                            float(np.count_nonzero(masks[bag_index] & masks[person_index]) / bag_area) if bag_area else None),
                            backpack_to_person_mask_area_ratio=(float(bag_area / person_area) if person_area else None))
                        if association["person_id"] == person["person_id"]:
                            person.update(mask_backpack=True, mask_backpack_status="spatial_support")
                            person["bag_associations"].append({"backpack_detection_index": index, "status": "spatial_support"})
                        elif association["status"] == "ambiguous":
                            if person["mask_backpack"] is not True:
                                person["mask_backpack_status"] = "ambiguous"
                            person["bag_associations"].append({"backpack_detection_index": index, "status": "ambiguous"})
                    associations.append(association)
                extra.update(backpack_associations=associations, mask_tensor_shape=list(masks.shape),
                             mask_representation="exterior or merged contour, rounded to 0.1 pixel; holes not retained")
            elif key == "yolo26n":
                detections, extra["suppressed_detections"] = semantic_vehicle_dedup(detections)
            elif key == "pose":
                points = result.keypoints.data.cpu().tolist() if result.keypoints is not None else []
                if len(points) != len(detections):
                    raise ValueError("Pose keypoint and detection identity mismatch")
                for index, (detection, keypoints) in enumerate(zip(detections, points)):
                    detection.update(pose_detection_index=index, keypoints=keypoints,
                                     orientation_evidence=orientation_from_keypoints(
                                         keypoints, detection["xyxy"], width, height, detection["confidence"], 960))
                matched = match_canonical(persons, detections)
                for person, match in zip(persons, matched):
                    person["pose"] = {k: v for k, v in match.items() if k not in person}
                    # Explicit names keep this independent of future canonical fields.
                    person["pose"].update(orientation=match["orientation"],
                                          facing_direction=match["facing_direction"],
                                          direction=match["facing_direction"],
                                          confidence=match["evidence_score"])
                orientation = Counter(p["pose"]["orientation"] for p in persons)
                direction = Counter(p["pose"]["facing_direction"] for p in persons)
                extra.update(orientation_counts={k: orientation[k] for k in ("front", "back", "side", "unknown")},
                             direction_counts={k: direction[k] for k in ("left", "right", "toward", "away", "unclear")},
                             canonical_people_total=len(persons), keypoint_names=KEYPOINT_NAMES,
                             orientation_method="experimental conservative 2-D geometry rules; uncalibrated")
            elapsed = time.perf_counter() - model_started
            timings[key + "_seconds"] = elapsed
            experts[key] = {"counts": count_objects(detections, key), "detections": detections,
                            "device": self.device, "inference_seconds": elapsed,
                            "checkpoint": WEIGHTS[key],
                            "predict_speed_ms": {k: float(v) for k, v in result.speed.items()}, **extra}
        timings["vision_total_seconds"] = time.perf_counter() - started
        used_devices = sorted({expert["device"] for expert in experts.values()})
        return {"width": width, "height": height, "device": self.device,
                "devices_used": used_devices, "warnings": list(self.warnings),
                "timings": timings, "experts": experts, "persons": persons}
