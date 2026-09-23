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

from . import roster
from .pose_geometry import KEYPOINT_NAMES, RULE_VERSION, orientation_from_keypoints

ENGINE_VERSION = "vision_v2"
LABELS = ["person", "bicycle", "motorcycle", "dog", "backpack", "baby stroller",
          "all-terrain vehicle", "utility terrain vehicle", "car", "truck", "bus",
          "golf cart", "tractor", "kick scooter", "suitcase", "duffel bag"]
COCO_CLASSES = [0, 1, 2, 3, 5, 7, 16, 24, 26, 28]
COUNT_FIELDS = ("people_total", "bicycles", "strollers", "motorcycles", "atv_utv",
                "other_vehicles", "dogs", "backpacks", "cars_trucks_buses", "kick_scooters")
WEIGHTS = {"yolo26n": "yolo26n.pt", "yoloe": "yoloe-26s-trail-prompts.pt",
           "megadetector": "MDV6-yolov10-c.pt", "pose": "yolo26n-pose.pt"}
# Per-expert inference sizes. Identical on CPU and GPU so results do not
# depend on the machine; "fast" trades small/occluded people for speed.
PROFILES = {
    "standard": {"yolo26n": 960, "megadetector": 960, "yoloe": 1280, "pose": 1280},
    "fast": {"yolo26n": 640, "megadetector": 640, "yoloe": 960, "pose": 960},
}
# One-to-many heads + IoU NMS keep overlapping people in groups that the
# end-to-end (NMS-free) heads merge; measured on the local labeled images.
ONE_TO_MANY = ("yoloe", "yolo26n", "pose")
NMS_IOU = 0.7
# Large JPEGs are decoded at reduced scale (PIL draft) when the long side is
# at least this; detector inputs stay >= their inference size.
DRAFT_MIN_LONG_SIDE = 3000
RAW_CONFIDENCE = roster.POLICY["raw_floor"]
GATE_CONFIDENCE = 0.20
MD_LABELS = {0: "animal", 1: "person", 2: "vehicle"}
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


def detections_by_id(expert):
    """Stable raw detection IDs remain valid after invalid boxes are removed."""
    return {d.get("detection_index", index): d
            for index, d in enumerate(expert.get("detections", []))}


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
    """Loads the detector/pose checkpoints once and processes individual images.

    Every expert keeps raw detections down to ``RAW_CONFIDENCE``; acceptance
    thresholds live in post-processing so they can change without re-inference.
    """

    def __init__(self, models_dir: Path, device: str = "auto", threads: int = 8,
                 profile: str = "standard", empty_frame_gate: bool = False):
        started = time.perf_counter()
        if threads < 1:
            raise ValueError("threads must be at least 1")
        if profile not in PROFILES:
            raise ValueError("profile must be one of: " + ", ".join(PROFILES))
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
        self.threads = int(threads)
        self.profile, self.sizes = profile, dict(PROFILES[profile])
        self.empty_frame_gate = bool(empty_frame_gate)
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
                       "profile": profile, "imgsz": self.sizes, "raw_confidence": RAW_CONFIDENCE,
                       "empty_frame_gate": self.empty_frame_gate, "gate_confidence": GATE_CONFIDENCE,
                       "iou": NMS_IOU, "one_to_many_heads": list(ONE_TO_MANY), "half": False, "max_det": 300,
                       "pose_rule_version": RULE_VERSION,
                       "roster_version": roster.POLICY["version"], "labels": LABELS,
                       "requested_device": str(device),
                       "coordinates": "EXIF-corrected original-image pixel coordinates",
                       "direction_semantics": "facing evidence per image; motion comes from events",
                       "mask_role": "same YOLOE expert; no independent mask vote"}
        self.startup_seconds = time.perf_counter() - started

    def _load_models(self):
        self.models = {key: constructor(str(self.models_dir / WEIGHTS[key])).to(self.device)
                       for key, constructor in self._constructors.items()}
        for key in ONE_TO_MANY:
            # Must happen before the first predict: Ultralytics removes the
            # one-to-many branch when it fuses an end-to-end model.
            self.models[key].model.end2end = False
            if getattr(self.models[key].model, "end2end", False):
                raise RuntimeError(f"Could not select the one-to-many head for {key}")
        import numpy as np
        blank = np.zeros((64, 64, 3), dtype=np.uint8)
        for key in list(self.models):  # _predict may rebuild self.models on CPU fallback
            self._predict(key, blank)
        self.torch.set_num_threads(self.threads)
        names = {int(k): str(v) for k, v in self.models["megadetector"].names.items()}
        if names != MD_LABELS:
            raise ValueError("Unexpected MegaDetector labels: " + repr(names))
        names = {int(k): str(v) for k, v in self.models["yoloe"].names.items()}
        if names != dict(enumerate(LABELS)):
            raise ValueError("YOLOE checkpoint prompts do not match the trail categories; rerun setup")

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
        options = {"imgsz": self.sizes[key], "conf": RAW_CONFIDENCE, "iou": NMS_IOU, "half": False,
                   "max_det": 300, "agnostic_nms": False, "verbose": False, "save": False}
        if key == "yolo26n":
            options["classes"] = COCO_CLASSES
        if key == "yoloe":
            options["retina_masks"] = False
        failure = None
        try:
            return self.models[key].predict(image, device=self.device, **options)[0]
        except RuntimeError as exc:
            if self.device == "cpu" or not is_cuda_failure(exc):
                raise
            failure = exc if "out of memory" not in str(exc).lower() else None
            message = str(exc)
        if failure is None:
            # Transient memory pressure (one very crowded frame): the failed
            # pass's tensors are released once the except block has exited.
            import gc
            gc.collect()
            self.torch.cuda.empty_cache()
            try:
                return self.models[key].predict(image, device=self.device, **options)[0]
            except RuntimeError as again:
                if not is_cuda_failure(again):
                    raise
                failure = again if str(again) else RuntimeError(message)
        self._fallback_cpu(failure)
        return self.models[key].predict(image, device="cpu", **options)[0]

    def _extract_detections(self, result):
        boxes = result.boxes.cpu()
        names = result.names
        return [{"detection_index": index, "class_id": int(class_id),
                 "label": str(names[int(class_id)]), "confidence": float(confidence),
                 "xyxy": [float(v) for v in box], "bbox_xyxy": [float(v) for v in box]}
                for index, (box, class_id, confidence) in enumerate(
                    zip(boxes.xyxy.tolist(), boxes.cls.tolist(), boxes.conf.tolist()))
                if all(math.isfinite(v) for v in box) and box[2] > box[0] and box[3] > box[1]]

    def _gate_open(self, experts, strip, height):
        """Cheap experts must see something before the heavier ones run."""
        for key in ("yolo26n", "megadetector"):
            for d in experts[key]["detections"]:
                if d["confidence"] >= GATE_CONFIDENCE and roster.strip_fraction(d["xyxy"], strip, height) < .5:
                    return True
        return False

    def analyze(self, image_path: Path) -> dict:
        import numpy as np
        from PIL import Image, ImageOps
        from .appearance import descriptor
        from .imageinfo import detect_data_strip, read_capture_metadata
        started = time.perf_counter()
        # PaddlePaddle (attribute model) resets the shared OpenMP pool to one
        # thread on every run, which silently made detection ~2x slower on CPU.
        if self.torch.get_num_threads() != self.threads:
            self.torch.set_num_threads(self.threads)
        with Image.open(image_path) as source:
            metadata = read_capture_metadata(source, Path(image_path))
            full_w, full_h = _oriented_size(source)
            if source.format in ("JPEG", "MPO") and max(source.size) >= DRAFT_MIN_LONG_SIDE:
                source.draft("RGB", (source.size[0] // 2, source.size[1] // 2))
            rgb_image = to_rgb8(ImageOps.exif_transpose(source))
        scale = full_w / rgb_image.width
        rgb = np.asarray(rgb_image)
        image = rgb[:, :, ::-1].copy()
        height, width = image.shape[:2]
        strip = detect_data_strip(rgb)
        timings = {"decode_seconds": time.perf_counter() - started}
        experts = {}
        for key in ("yolo26n", "megadetector", "yoloe", "pose"):
            if key in ("yoloe", "pose") and self.empty_frame_gate and not self._gate_open(experts, strip, height):
                experts[key] = {"counts": count_objects([], key), "detections": [], "skipped": "empty_frame_gate",
                                "device": self.device, "inference_seconds": 0.0, "checkpoint": WEIGHTS[key],
                                "imgsz": self.sizes[key]}
                continue
            model_started = time.perf_counter()
            result = self._predict(key, image)
            detections = self._extract_detections(result)
            raw_count = len(result.boxes.cpu().conf.tolist())  # before degenerate boxes are dropped
            extra = {}
            if key == "yoloe":
                if detections and result.masks is None:
                    raise ValueError("YOLOE returned detections without instance masks")
                polygons = result.masks.xy if result.masks is not None else []
                if len(polygons) != raw_count:
                    raise ValueError("YOLOE mask and detection instance identity mismatch")
                for detection in detections:
                    detection["mask_polygon_xy"] = compact_polygon(polygons[detection["detection_index"]], width, height)
                extra["mask_representation"] = "exterior or merged contour, rounded to 0.1 pixel; holes not retained"
            elif key == "pose":
                points = result.keypoints.data.cpu().tolist() if result.keypoints is not None else []
                if len(points) != raw_count:
                    raise ValueError("Pose keypoint and detection identity mismatch")
                for detection in detections:
                    index = detection["detection_index"]
                    keypoints = points[index]
                    detection.update(pose_detection_index=index, keypoints=keypoints,
                                     orientation_evidence=orientation_from_keypoints(
                                         keypoints, detection["xyxy"], width, height,
                                         detection["confidence"], self.sizes["pose"]))
                extra["keypoint_names"] = KEYPOINT_NAMES
            elapsed = time.perf_counter() - model_started
            timings[key + "_seconds"] = elapsed
            experts[key] = {"counts": count_objects(detections, key), "detections": detections,
                            "device": self.device, "inference_seconds": elapsed,
                            "checkpoint": WEIGHTS[key], "imgsz": self.sizes[key],
                            "predict_speed_ms": {k: float(v) for k, v in result.speed.items()}, **extra}
        gated = any(e.get("skipped") for e in experts.values())
        persons = []
        detections = {key: detections_by_id(expert) for key, expert in experts.items()}
        for candidate in ([] if gated else roster.build_candidates(experts, strip, height)):
            members = candidate["members"]
            person = {"person_id": candidate["candidate_id"], **candidate, "keypoints": None,
                      "pose_confidence": None, "orientation_evidence": None, "mask_polygon_xy": None}
            if "pose" in members:
                pose = detections["pose"][members["pose"]["detection_index"]]
                person.update(keypoints=pose["keypoints"], pose_confidence=pose["confidence"],
                              orientation_evidence=pose["orientation_evidence"])
            if "yoloe" in members:
                person["mask_polygon_xy"] = detections["yoloe"][members["yoloe"]["detection_index"]]["mask_polygon_xy"]
            person["appearance"] = descriptor(rgb, person["xyxy"], person["mask_polygon_xy"], device=self.device)
            persons.append(person)
        timings["vision_total_seconds"] = time.perf_counter() - started
        used_devices = sorted({expert["device"] for expert in experts.values()})
        if scale != 1.0:
            _rescale(experts, persons, strip, scale)
        return {"width": full_w, "height": full_h, "analysis_scale": scale, "device": self.device,
                "profile": self.profile, "devices_used": used_devices, "warnings": list(self.warnings),
                "metadata": metadata, "data_strip": strip, "gated_empty": gated, "timings": timings,
                "experts": experts, "persons": persons,
                # Transient: reused by the attribute engine, never cached.
                "_decoded_image": rgb_image}


def to_rgb8(image):
    """8-bit RGB; 16-bit and float single-channel images are rescaled, not clipped."""
    from PIL import Image
    if image.mode in ("I;16", "I;16B", "I;16L", "I;16N", "I", "F"):
        import numpy as np
        array = np.asarray(image, dtype=np.float64)
        finite = array[np.isfinite(array)]
        low, high = (np.percentile(finite, (0.5, 99.5)) if finite.size else (0.0, 1.0))
        if high <= low:
            high = low + 1.0
        array = np.clip((np.nan_to_num(array, nan=low) - low) / (high - low), 0, 1) * 255
        return Image.fromarray(array.astype(np.uint8), "L").convert("RGB")
    return image.convert("RGB")


def _oriented_size(image):
    """Image size after EXIF orientation, without decoding pixels."""
    try:
        orientation = image.getexif().get(0x0112, 1)
    except Exception:
        orientation = 1
    width, height = image.size
    return (height, width) if orientation in (5, 6, 7, 8) else (width, height)


def _rescale(experts, persons, strip, scale):
    """Convert decoded-image pixel coordinates to original-image pixels."""
    def box(values):
        return [round(v * scale, 2) for v in values]

    def points(values):
        return [[round(x * scale, 1), round(y * scale, 1)] for x, y in values]

    for expert in experts.values():
        for d in expert.get("detections", []):
            d["xyxy"] = box(d["xyxy"])
            d["bbox_xyxy"] = list(d["xyxy"])
            if d.get("mask_polygon_xy"):
                d["mask_polygon_xy"] = points(d["mask_polygon_xy"])
            if d.get("keypoints"):
                d["keypoints"] = [[x * scale, y * scale, c] for x, y, c in d["keypoints"]]
    detections = {key: detections_by_id(expert) for key, expert in experts.items()}
    for person in persons:
        person["xyxy"] = box(person["xyxy"])
        person["bbox_xyxy"] = list(person["xyxy"])
        for member in person["members"].values():
            member["xyxy"] = box(member["xyxy"])
        # Re-link to the rescaled expert keypoints and mask polygons.
        members = person["members"]
        if "pose" in members:
            person["keypoints"] = detections["pose"][members["pose"]["detection_index"]]["keypoints"]
        if "yoloe" in members:
            person["mask_polygon_xy"] = detections["yoloe"][members["yoloe"]["detection_index"]]["mask_polygon_xy"]
    for key in ("top", "bottom"):
        if strip.get(key):
            strip[key] = int(round(strip[key] * scale))
