"""One PP-LCNet pass per detected person; no PaddleX or network is required.

Preprocessing follows the official model's ``inference.yml``: RGB, OpenCV
bilinear resize to 192 x 256, ImageNet channel normalization, then NCHW.
The model emits independent sigmoid scores, not calibrated probabilities.
"""
from __future__ import annotations

import math
import os
import time
from pathlib import Path

MODEL_NAME = "PP-LCNet_x1_0_pedestrian_attribute"
INDEX_NAMES = {
    15: "handbag", 16: "shoulder_bag", 17: "backpack", 19: "age_under18",
    20: "age_18_60", 21: "age_over60", 22: "native_female", 23: "front",
    24: "side", 25: "back",
}
POLICY = {
    "score_threshold": 0.8, "margin": 0.2, "min_width": 80,
    "min_height": 160, "backpack_high": 0.8, "backpack_low": 0.2,
    "presentation_high": 0.8, "presentation_low": 0.2,
    "version": 1,
}
NOTES = [
    "Native model scores and thresholds are uncalibrated; unknown is preserved.",
    "Native age bins are appearance estimates. Under18 includes teenagers; child detection failed prior spot checks.",
    "The native Female score is an unvalidated binary training label, not a person's gender identity or sex.",
    "Presentation proxy is experimental and must not be treated as an established personal attribute.",
    "Backpack presence describes a person carrying one; this differs from the detector's backpack object count.",
    "Front/back/side describe apparent body orientation, not measured movement.",
]


def _classify_group(scores, names, detail_ok):
    ordered = sorted(((float(scores[key]), label) for label, key in names.items()), reverse=True)
    reasons = []
    if not detail_ok:
        reasons.append("crop_below_configured_detail_gate")
    if ordered[0][0] < POLICY["score_threshold"]:
        reasons.append("top_score_below_threshold")
    if ordered[0][0] - ordered[1][0] < POLICY["margin"]:
        reasons.append("top_vs_runner_up_margin_below_threshold")
    return ("unknown" if reasons else ordered[0][1]), reasons


def derive_attributes(scores, width, height):
    """Apply conservative gates while retaining every relevant native score."""
    if any(key not in scores for key in INDEX_NAMES.values()):
        raise ValueError("Requested native attribute score is missing")
    if any(not math.isfinite(float(value)) or not 0 <= float(value) <= 1 for value in scores.values()):
        raise ValueError("Native attribute score is outside [0, 1]")
    detail_ok = width >= POLICY["min_width"] and height >= POLICY["min_height"]
    age, age_reasons = _classify_group(scores, {
        "under18": "age_under18", "18_60": "age_18_60", "over60": "age_over60",
    }, detail_ok)
    orientation, orientation_reasons = _classify_group(scores, {
        "front": "front", "side": "side", "back": "back",
    }, detail_ok)
    bag_reasons = [] if detail_ok else ["crop_below_configured_detail_gate"]
    if POLICY["backpack_low"] < scores["backpack"] < POLICY["backpack_high"]:
        bag_reasons.append("backpack_score_between_low_and_high_thresholds")
    backpack = None if bag_reasons else scores["backpack"] >= POLICY["backpack_high"]
    proxy_reasons = [] if detail_ok else ["crop_below_configured_detail_gate"]
    female_score = scores["native_female"]
    if POLICY["presentation_low"] < female_score < POLICY["presentation_high"]:
        proxy_reasons.append("native_binary_score_between_low_and_high_thresholds")
    proxy = "unclear" if proxy_reasons else (
        "feminine_presentation_proxy" if female_score >= POLICY["presentation_high"]
        else "masculine_presentation_proxy"
    )
    return {
        "scores": dict(scores), "native_age_label": age,
        "native_orientation_label": orientation, "backpack_presence": backpack,
        "native_female_score": female_score, "presentation_proxy": proxy,
        "detail_gate_passed": detail_ok, "crop_width": width, "crop_height": height,
        "gate_reasons": {"age": age_reasons, "orientation": orientation_reasons,
                         "backpack": bag_reasons, "presentation_proxy": proxy_reasons},
    }


def crop_box(box, width, height):
    """Round outwards and clamp a finite detector box to EXIF-corrected pixels."""
    if len(box) != 4 or not all(math.isfinite(float(value)) for value in box):
        raise ValueError("Person bounding box must contain four finite coordinates")
    x1, y1, x2, y2 = box
    result = [max(0, math.floor(x1)), max(0, math.floor(y1)),
              min(width, math.ceil(x2)), min(height, math.ceil(y2))]
    if result[2] <= result[0] or result[3] <= result[1]:
        raise ValueError("Person bounding box has no image area")
    return result


def unknown_attributes(error):
    return {
        "status": "error", "error": str(error), "scores": None,
        "native_age_label": "unknown", "native_orientation_label": "unknown",
        "backpack_presence": None, "native_female_score": None,
        "presentation_proxy": "unclear", "detail_gate_passed": False,
        "crop_width": None, "crop_height": None, "seconds": 0.0,
        "gate_reasons": {key: ["person_prediction_failed"] for key in
                         ("age", "orientation", "backpack", "presentation_proxy")},
    }


class AttributeEngine:
    """Run local PP-LCNet with usable Paddle CUDA, otherwise CPU.

    The Windows installer uses the supported CPU Paddle wheel. Object detectors
    and pose can still use CUDA independently. A compatible user-installed
    Paddle CUDA build is recognized and tried, with inference-tested fallback.
    """

    def __init__(self, models_dir: Path, device: str = "auto", threads: int = 8):
        if int(threads) < 1:
            raise ValueError("threads must be at least 1")
        os.environ.setdefault("GLOG_minloglevel", "2")
        import cv2
        import numpy as np
        import paddle
        from paddle import inference

        self._cv2, self._np, self._paddle, self._inference = cv2, np, paddle, inference
        self.models_dir = Path(models_dir)
        self.model_dir = self.models_dir / (MODEL_NAME + "_infer")
        if (self.models_dir / "inference.json").is_file():
            self.model_dir = self.models_dir
        for filename in ("inference.json", "inference.pdiparams"):
            if not (self.model_dir / filename).is_file():
                raise FileNotFoundError(f"Missing attribute model: {self.model_dir / filename}")
        self.threads = int(threads)
        self.requested_device = str(device)
        self.device = "cpu"
        self.fallback_reasons = []
        self.predictor = None
        requested = str(device).lower()
        allowed = requested in ("auto", "cpu", "cuda", "gpu") or requested.isdigit() or (
            requested.startswith(("cuda:", "gpu:")) and requested.split(":", 1)[1].isdigit()
        )
        if not allowed:
            raise ValueError(f"Unsupported attribute device: {device}")
        cuda_available = False
        if requested != "cpu":
            try:
                cuda_available = bool(paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0)
            except Exception as exc:
                self.fallback_reasons.append(f"Paddle CUDA probe failed: {type(exc).__name__}: {exc}")
        if cuda_available:
            gpu_id = int(requested.split(":", 1)[1]) if ":" in requested else (
                int(requested) if requested.isdigit() else 0
            )
            try:
                self.predictor = self._create_predictor(gpu_id=gpu_id)
                self.device = f"cuda:{gpu_id}"
                self._run_tensor(self._np.zeros((1, 3, 256, 192), dtype=self._np.float32))
            except Exception as exc:
                self.fallback_reasons.append(f"Paddle CUDA initialization failed; using CPU: {type(exc).__name__}: {exc}")
                self.predictor = None
        elif requested != "cpu":
            self.fallback_reasons.append("Paddle CUDA runtime unavailable; attribute classifier uses CPU")
        if self.predictor is None:
            self._init_cpu()

    def _create_predictor(self, gpu_id=None, mkldnn=True):
        config = self._inference.Config(str(self.model_dir / "inference.json"),
                                        str(self.model_dir / "inference.pdiparams"))
        if gpu_id is None:
            config.disable_gpu()
            config.set_cpu_math_library_num_threads(self.threads)
            if mkldnn:
                config.enable_mkldnn()
                config.set_mkldnn_cache_capacity(10)
            else:
                config.disable_mkldnn()
            config.set_optimization_level(3)
        else:
            config.enable_use_gpu(100, gpu_id)
            config.disable_mkldnn()
        config.enable_new_ir()
        config.enable_new_executor()
        config.enable_memory_optim()
        config.disable_glog_info()
        return self._inference.create_predictor(config)

    def _init_cpu(self):
        self.device = "cpu"
        try:
            self.predictor = self._create_predictor(mkldnn=True)
            self._run_tensor(self._np.zeros((1, 3, 256, 192), dtype=self._np.float32))
        except Exception as exc:
            self.fallback_reasons.append(f"CPU oneDNN unavailable; using plain CPU: {type(exc).__name__}: {exc}")
            self.predictor = self._create_predictor(mkldnn=False)
            self._run_tensor(self._np.zeros((1, 3, 256, 192), dtype=self._np.float32))

    def _run_tensor(self, tensor):
        input_handle = self.predictor.get_input_handle(self.predictor.get_input_names()[0])
        input_handle.reshape(tensor.shape)
        input_handle.copy_from_cpu(tensor)
        self.predictor.run()
        return self.predictor.get_output_handle(self.predictor.get_output_names()[0]).copy_to_cpu()[0]

    def _predict_crop(self, crop):
        # Multiplication then addition reproduces the official preprocessing's
        # float32 rounding, including the order of channel operations.
        array = self._cv2.resize(self._np.asarray(crop), (192, 256),
                                 interpolation=self._cv2.INTER_LINEAR).astype(self._np.float32)
        for channel, (mean, std) in enumerate(zip((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))):
            array[:, :, channel] *= (1.0 / 255.0) / std
            array[:, :, channel] += -mean / std
        tensor = self._np.ascontiguousarray(array.transpose(2, 0, 1)[None])
        try:
            scores = self._run_tensor(tensor)
        except Exception as exc:
            if not self.device.startswith("cuda"):
                raise
            self.fallback_reasons.append(f"Paddle CUDA inference failed; using CPU: {type(exc).__name__}: {exc}")
            self.predictor = None
            self._init_cpu()
            scores = self._run_tensor(tensor)
        if len(scores) != 26:
            raise ValueError(f"Expected 26 native PP-LCNet scores; got {len(scores)}")
        # Keep precision identical to PaddleX's native postprocessor.
        return {name: round(float(scores[index]), 5) for index, name in INDEX_NAMES.items()}

    def analyze(self, image_path: Path, persons: list) -> dict:
        """Return one attribute record for every canonical person, including failures."""
        from PIL import Image, ImageOps

        started = time.perf_counter()
        results = []
        if persons:
            with Image.open(image_path) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
            for person in persons:
                person_started = time.perf_counter()
                try:
                    box = crop_box(person["bbox_xyxy"], image.width, image.height)
                    crop = image.crop(box)
                    attributes = derive_attributes(self._predict_crop(crop), crop.width, crop.height)
                    attributes.update({"status": "ok", "error": None, "crop_xyxy": box})
                except Exception as exc:
                    attributes = unknown_attributes(f"{type(exc).__name__}: {exc}")
                attributes["seconds"] = time.perf_counter() - person_started
                attributes["device"] = self.device
                results.append({**person, "attributes": attributes})
        errors = sum(person["attributes"]["status"] != "ok" for person in results)
        return {
            "model": MODEL_NAME, "device": self.device, "requested_device": self.requested_device,
            "seconds": time.perf_counter() - started, "persons": results,
            "status": "partial_error" if errors else "ok", "people_failed": errors,
            "fallback_reasons": list(self.fallback_reasons), "notes": list(NOTES),
        }
