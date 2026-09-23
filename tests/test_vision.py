"""Portable detector invariants; these do not estimate real-image accuracy.

Nothing here loads model files, torch or ultralytics: engine pieces run on a
``VisionEngine`` made with ``object.__new__`` and fake expert results.
"""
import copy
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import warnings

import numpy as np
from PIL import Image, ImageOps

from trailcam import roster
from trailcam.vision import (COCO_CLASSES, DRAFT_MIN_LONG_SIDE, ENGINE_VERSION, GATE_CONFIDENCE, LABELS,
                             MD_LABELS, NMS_IOU, ONE_TO_MANY, PROFILES, RAW_CONFIDENCE, WEIGHTS,
                             VisionEngine, _oriented_size, _rescale, assign_bag, compact_polygon,
                             count_objects, select_device, semantic_vehicle_dedup, to_rgb8)


def detection(label, box=(0, 0, 100, 100), confidence=.8):
    return {"label": label, "xyxy": list(box), "confidence": confidence}


def person(name, index, box):
    return {"person_id": name, "source_detection_index": index, "xyxy": box}


class VisionTests(unittest.TestCase):
    def test_unsupported_category_is_not_reported_as_zero(self):
        nano = count_objects([], "yolo26n")
        self.assertEqual(nano["backpacks"], 0)
        self.assertIsNone(nano["strollers"])
        self.assertEqual(count_objects([], "yoloe")["strollers"], 0)
        md = count_objects([detection("animal")], "megadetector")
        self.assertEqual(md["animals_total"], 1)
        self.assertIsNone(md["dogs"])

    def test_vehicle_categories_do_not_count_backpacks_as_transport(self):
        objects = [detection(name) for name in ("utility terrain vehicle", "golf cart", "car", "tractor", "backpack")]
        counts = count_objects(objects, "yoloe")
        self.assertEqual(counts["atv_utv"], 2)
        self.assertEqual(counts["other_vehicles"], 2)
        self.assertEqual(counts["cars_trucks_buses"], 1)
        self.assertEqual(counts["backpacks"], 1)

    def test_vehicle_dedup_keeps_other_object_classes(self):
        objects = [detection("car", confidence=.9), detection("utility terrain vehicle"),
                   detection("person"), detection("bicycle"), detection("backpack")]
        kept, suppressed = semantic_vehicle_dedup(objects)
        self.assertEqual([d["label"] for d in kept], ["car", "person", "bicycle", "backpack"])
        self.assertEqual(suppressed[0]["label"], "utility terrain vehicle")
        self.assertEqual(suppressed[0]["kept_raw_detection_index"], 0)

    def test_spatial_bag_association_abstains_between_two_people(self):
        people = [person("p1", 0, [0, 0, 100, 200]), person("p2", 1, [0, 0, 100, 200])]
        result = assign_bag([20, 40, 80, 100], people)
        self.assertEqual(result["status"], "ambiguous")
        self.assertIsNone(result["person_id"])
        self.assertEqual(len(result["candidates"]), 2)

    def test_bag_association_requires_containment_and_proximity(self):
        people = [person("p1", 0, [0, 0, 100, 200])]
        self.assertEqual(assign_bag([20, 40, 80, 100], people)["person_id"], "p1")
        self.assertEqual(assign_bag([300, 300, 400, 400], people)["status"], "unassigned")

    def test_polygon_rejects_corrupt_coordinates(self):
        for points in ([[float("nan"), 0]], [[float("inf"), 0]], [[150, 1]]):
            with self.assertRaises(ValueError):
                compact_polygon(points, 100, 100)
        self.assertEqual(compact_polygon([[-.05, 5], [0, 5], [99.99, 100.05]], 100, 100), [[0, 5], [100.0, 100]])

    def test_explicit_cpu_does_not_probe_cuda(self):
        self.assertEqual(select_device(SimpleNamespace(), "cpu"), ("cpu", None))

    def test_available_cuda_requires_working_kernel(self):
        def broken_kernel(*args, **kwargs):
            raise RuntimeError("no kernel image is available")
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True), ones=broken_kernel)
        device, warning = select_device(torch, "auto")
        self.assertEqual(device, "cpu")
        self.assertIn("kernel check failed", warning)

    def test_gpu_absence_falls_back_without_importing_gpu_tools(self):
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        self.assertEqual(select_device(torch, "auto")[0], "cpu")

    def test_device_validation_is_explicit(self):
        with self.assertRaises(ValueError):
            select_device(SimpleNamespace(), "not-a-device")


# ---------------------------------------------------------------------------
# v2 engine pieces, exercised without model files.

COCO_NAMES = {0: "person", 1: "bicycle", 16: "dog", 24: "backpack"}
YOLOE_NAMES = dict(enumerate(LABELS))
POSE_NAMES = {0: "person"}
NO_STRIP = {"top": 0, "bottom": 0}


class FakeTorch:
    """The torch calls the engine makes: the thread pool and ``cuda.empty_cache``."""

    def __init__(self, threads, log=None):
        self.threads, self.set_calls, self.empty_cache_calls = threads, [], 0
        self.log = log if log is not None else []
        self.cuda = SimpleNamespace(empty_cache=self._empty_cache)

    def get_num_threads(self):
        return self.threads

    def set_num_threads(self, threads):
        self.set_calls.append(threads)
        self.log.append(("set_num_threads", threads))
        self.threads = threads

    def _empty_cache(self):
        self.empty_cache_calls += 1
        self.log.append(("empty_cache",))


def bare_engine(**attributes):
    """A VisionEngine that skips __init__: no model files, torch or ultralytics."""
    engine = object.__new__(VisionEngine)
    state = {"threads": 4, "profile": "standard", "sizes": dict(PROFILES["standard"]),
             "empty_frame_gate": True, "device": "cpu", "warnings": [], "models": {}}
    state.update(attributes)
    state.setdefault("torch", FakeTorch(state["threads"]))
    engine.__dict__.update(state)
    return engine


class FakeArray:
    """The slice of a torch tensor analyze() touches: .cpu() and .tolist()."""

    def __init__(self, values):
        self.values = values

    def cpu(self):
        return self

    def tolist(self):
        return copy.deepcopy(self.values)


def fake_result(names, detections=(), polygons=None, keypoints=None):
    """Ultralytics-like result for ``[(class_id, [x1, y1, x2, y2], confidence), ...]``."""
    boxes = SimpleNamespace(xyxy=FakeArray([list(map(float, d[1])) for d in detections]),
                            cls=FakeArray([float(d[0]) for d in detections]),
                            conf=FakeArray([float(d[2]) for d in detections]))
    return SimpleNamespace(boxes=SimpleNamespace(cpu=lambda: boxes), names=names,
                           speed={"preprocess": 1.0, "inference": 2.0, "postprocess": .5},
                           masks=None if polygons is None else SimpleNamespace(xy=polygons),
                           keypoints=None if keypoints is None else SimpleNamespace(data=FakeArray(keypoints)))


def empty_results():
    return {"yolo26n": fake_result(COCO_NAMES), "megadetector": fake_result(MD_LABELS),
            "yoloe": fake_result(YOLOE_NAMES), "pose": fake_result(POSE_NAMES)}


class ProfileAndConstantTests(unittest.TestCase):
    def test_profiles_size_every_expert(self):
        self.assertEqual(set(PROFILES), {"standard", "fast"})
        for name, sizes in PROFILES.items():
            with self.subTest(profile=name):
                self.assertEqual(set(sizes), set(WEIGHTS))
                for size in sizes.values():
                    self.assertIs(type(size), int)
                    self.assertEqual(size % 32, 0)  # detector stride

    def test_profile_sizes(self):
        self.assertEqual(PROFILES["standard"], {"yolo26n": 960, "megadetector": 960, "yoloe": 1280, "pose": 1280})
        self.assertEqual(PROFILES["fast"], {"yolo26n": 640, "megadetector": 640, "yoloe": 960, "pose": 960})
        for key in WEIGHTS:
            self.assertLessEqual(PROFILES["fast"][key], PROFILES["standard"][key])
        for sizes in PROFILES.values():
            # The people experts never see a smaller image than the gate experts.
            self.assertGreaterEqual(min(sizes["yoloe"], sizes["pose"]), max(sizes["yolo26n"], sizes["megadetector"]))

    def test_draft_decoding_keeps_detector_inputs_above_inference_size(self):
        # A draft decode halves the long side; the result must still cover every profile size.
        largest = max(max(sizes.values()) for sizes in PROFILES.values())
        self.assertGreaterEqual(DRAFT_MIN_LONG_SIDE // 2, largest)

    def test_one_to_many_and_nms_constants(self):
        self.assertEqual(ONE_TO_MANY, ("yoloe", "yolo26n", "pose"))
        self.assertLessEqual(set(ONE_TO_MANY), set(WEIGHTS))
        self.assertNotIn("megadetector", ONE_TO_MANY)  # MegaDetector keeps its own head
        self.assertEqual(NMS_IOU, .7)
        self.assertEqual(ENGINE_VERSION, "vision_v2")
        self.assertEqual(MD_LABELS, {0: "animal", 1: "person", 2: "vehicle"})

    def test_raw_floor_stays_below_every_acceptance_threshold(self):
        # Raw detections are kept low so acceptance can be tuned without re-inference.
        self.assertEqual(RAW_CONFIDENCE, roster.POLICY["raw_floor"])
        self.assertLessEqual(RAW_CONFIDENCE, roster.POLICY["accept_pair"])
        self.assertLessEqual(RAW_CONFIDENCE, min(roster.POLICY["accept_single"].values()))
        self.assertEqual(GATE_CONFIDENCE, .20)
        self.assertGreater(GATE_CONFIDENCE, RAW_CONFIDENCE)


class FakeCheckpoint:
    """Ultralytics-like checkpoint; ``predict`` records the warm-up calls ``_load_models`` makes.

    ``fail_on_cuda`` makes every CUDA predict raise that message (a GPU that is
    visible but cannot run the model); CPU predicts always succeed.
    """

    def __init__(self, path, names, log=None, fail_on_cuda=None):
        self.path, self.names, self.device = path, names, None
        self.model = SimpleNamespace(end2end=True)  # Ultralytics' NMS-free default
        self.predicts, self.log, self.fail_on_cuda = [], log if log is not None else [], fail_on_cuda

    def to(self, device):
        self.device = device
        return self

    def predict(self, image, device=None, **options):
        self.predicts.append({"image": image, "device": device, **options})
        self.log.append(("predict", Path(self.path).name, device))
        if self.fail_on_cuda and str(device).startswith("cuda"):
            raise RuntimeError(self.fail_on_cuda)
        return [SimpleNamespace(warmup=True)]


def constructors(names=None, log=None, fail_on_cuda=None):
    names = {"yolo26n": dict(COCO_NAMES), "yoloe": dict(YOLOE_NAMES), "megadetector": dict(MD_LABELS),
             "pose": dict(POSE_NAMES), **(names or {})}
    return {key: (lambda path, key=key: FakeCheckpoint(path, names[key], log, fail_on_cuda)) for key in WEIGHTS}


class EngineSetupTests(unittest.TestCase):
    def test_arguments_are_validated_before_heavy_imports(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.dict("sys.modules", {"torch": None, "ultralytics": None}):
            with self.assertRaisesRegex(ValueError, "profile must be one of: standard, fast"):
                VisionEngine(Path(tmp), device="cpu", profile="turbo")
            with self.assertRaisesRegex(ValueError, "threads"):
                VisionEngine(Path(tmp), threads=0)
            with self.assertRaisesRegex(FileNotFoundError, "Missing model files"):
                VisionEngine(Path(tmp), profile="fast")

    def test_load_models_selects_one_to_many_heads(self):
        engine = bare_engine(models_dir=Path("models"), _constructors=constructors(), device="cuda:0")
        engine._load_models()
        self.assertEqual(set(engine.models), set(WEIGHTS))
        for key, model in engine.models.items():
            with self.subTest(expert=key):
                self.assertEqual(Path(model.path).name, WEIGHTS[key])
                self.assertEqual(model.device, "cuda:0")
                self.assertIs(model.model.end2end, key not in ONE_TO_MANY)

    def test_load_models_warms_every_expert_then_restores_threads(self):
        # v2: each expert runs one 64x64 blank predict at load time (so CUDA
        # problems surface before the first photo), with the normal predict
        # options; the torch pool is reset to ``threads`` afterwards.
        log = []
        engine = bare_engine(models_dir=Path("models"), _constructors=constructors(log=log), device="cuda:0",
                             threads=3, torch=FakeTorch(1, log))
        engine._load_models()
        for key, model in engine.models.items():
            with self.subTest(expert=key):
                warmup, = model.predicts
                self.assertEqual((warmup["image"].shape, warmup["image"].dtype.name, int(warmup["image"].max())),
                                 ((64, 64, 3), "uint8", 0))
                self.assertEqual((warmup["device"], warmup["imgsz"], warmup["conf"], warmup["iou"]),
                                 ("cuda:0", PROFILES["standard"][key], RAW_CONFIDENCE, NMS_IOU))
        self.assertEqual([entry[1] for entry in log if entry[0] == "predict"], [WEIGHTS[k] for k in engine.models])
        self.assertEqual(log[-1], ("set_num_threads", 3))  # after the last warm-up predict
        self.assertEqual(engine.torch.threads, 3)

    # REAL REGRESSION (reported, not papered over): the warm-up loop iterates
    # ``self.models`` while a CUDA failure inside ``_predict`` makes
    # ``_fallback_cpu`` clear and rebuild that dict, so construction dies with
    # "RuntimeError: dictionary changed size during iteration" instead of
    # continuing on CPU (fixed: warm-up iterates over a snapshot of the model keys).
    def test_cuda_failure_during_warmup_falls_back_to_cpu(self):
        engine = bare_engine(models_dir=Path("models"), device="cuda:0",
                             _constructors=constructors(fail_on_cuda="CUDA error: no kernel image is available"))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            try:
                engine._load_models()
            except RuntimeError as exc:  # the same recovery VisionEngine.__init__ applies
                if engine.device == "cpu" or "cuda" not in str(exc).lower():
                    raise
                engine._fallback_cpu(exc)
        self.assertEqual(engine.device, "cpu")
        self.assertEqual({model.device for model in engine.models.values()}, {"cpu"})
        self.assertTrue(any("CUDA inference failed" in w for w in engine.warnings))

    def test_load_models_refuses_a_head_that_stays_end_to_end(self):
        class Stuck:
            end2end = property(lambda self: True, lambda self, value: None)

        def stuck(path):
            model = FakeCheckpoint(path, dict(POSE_NAMES))
            model.model = Stuck()
            return model

        makers = constructors()
        makers["pose"] = stuck
        with self.assertRaisesRegex(RuntimeError, "one-to-many head for pose"):
            bare_engine(models_dir=Path("models"), _constructors=makers)._load_models()

    def test_load_models_checks_label_maps(self):
        string_keys = {"megadetector": {str(k): v for k, v in MD_LABELS.items()}}
        bare_engine(models_dir=Path("models"), _constructors=constructors(string_keys))._load_models()
        for names, message in (({"megadetector": {0: "animal", 1: "person"}}, "MegaDetector"),
                               ({"yoloe": dict(enumerate(LABELS[:-1]))}, "YOLOE")):
            with self.subTest(message=message):
                engine = bare_engine(models_dir=Path("models"), _constructors=constructors(names))
                with self.assertRaisesRegex(ValueError, message):
                    engine._load_models()


class PredictOptionTests(unittest.TestCase):
    def engine(self, profile, device="cuda:0"):
        models = {key: mock.Mock(name=key) for key in WEIGHTS}
        for key, model in models.items():
            model.predict.return_value = [key + "_result"]
        return bare_engine(profile=profile, sizes=dict(PROFILES[profile]), models=models, device=device)

    def test_every_expert_uses_profile_size_raw_floor_and_nms(self):
        for profile in PROFILES:
            engine = self.engine(profile)
            for key in WEIGHTS:
                with self.subTest(profile=profile, expert=key):
                    self.assertEqual(engine._predict(key, "image"), key + "_result")
                    args, options = engine.models[key].predict.call_args
                    self.assertEqual(args, ("image",))
                    self.assertEqual(options["imgsz"], PROFILES[profile][key])
                    self.assertEqual((options["conf"], options["iou"], options["agnostic_nms"], options["half"],
                                      options["max_det"], options["device"]),
                                     (RAW_CONFIDENCE, NMS_IOU, False, False, 300, "cuda:0"))
                    self.assertEqual(options.get("classes"), COCO_CLASSES if key == "yolo26n" else None)
                    self.assertEqual(options.get("retina_masks", "absent"), False if key == "yoloe" else "absent")

    def fallback(self):
        return mock.patch.object(VisionEngine, "_fallback_cpu", autospec=True,
                                 side_effect=lambda self, exc: setattr(self, "device", "cpu"))

    def test_cuda_failure_retries_the_expert_on_cpu(self):
        # v2: an out-of-memory error is first retried once on the GPU after
        # ``torch.cuda.empty_cache()``; only a second CUDA failure moves to CPU.
        engine = self.engine("standard")
        model = engine.models["yoloe"]
        model.predict.side_effect = [RuntimeError("CUDA error: out of memory"),
                                     RuntimeError("CUDA error: out of memory"), ["cpu_result"]]
        with self.fallback() as fallback:
            self.assertEqual(engine._predict("yoloe", "image"), "cpu_result")
        fallback.assert_called_once()
        self.assertEqual(engine.torch.empty_cache_calls, 1)
        self.assertEqual([c.kwargs["device"] for c in model.predict.call_args_list], ["cuda:0", "cuda:0", "cpu"])
        model.predict.side_effect = RuntimeError("CUDA error: out of memory")
        with self.assertRaises(RuntimeError):  # already on CPU: nothing left to fall back to
            engine._predict("yoloe", "image")
        self.assertEqual(engine.torch.empty_cache_calls, 1)  # no GPU retry once on CPU

    def test_transient_out_of_memory_is_retried_on_the_gpu(self):
        engine = self.engine("standard")
        model = engine.models["pose"]
        model.predict.side_effect = [RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"), ["gpu_result"]]
        with self.fallback() as fallback:
            self.assertEqual(engine._predict("pose", "image"), "gpu_result")
        fallback.assert_not_called()
        self.assertEqual(engine.device, "cuda:0")
        self.assertEqual(engine.torch.empty_cache_calls, 1)
        self.assertEqual([c.kwargs["device"] for c in model.predict.call_args_list], ["cuda:0", "cuda:0"])

    def test_other_cuda_errors_fall_back_without_a_gpu_retry(self):
        engine = self.engine("standard")
        model = engine.models["yolo26n"]
        model.predict.side_effect = [RuntimeError("CUDA error: no kernel image is available"), ["cpu_result"]]
        with self.fallback() as fallback:
            self.assertEqual(engine._predict("yolo26n", "image"), "cpu_result")
        fallback.assert_called_once()
        self.assertEqual(engine.torch.empty_cache_calls, 0)
        self.assertEqual([c.kwargs["device"] for c in model.predict.call_args_list], ["cuda:0", "cpu"])

    def test_non_cuda_error_during_the_gpu_retry_is_raised(self):
        engine = self.engine("standard")
        engine.models["yoloe"].predict.side_effect = [RuntimeError("CUDA error: out of memory"),
                                                      RuntimeError("shape mismatch")]
        with self.fallback() as fallback, self.assertRaisesRegex(RuntimeError, "shape mismatch"):
            engine._predict("yoloe", "image")
        fallback.assert_not_called()

    def test_non_cuda_errors_are_not_retried(self):
        engine = self.engine("standard")
        engine.models["pose"].predict.side_effect = RuntimeError("shape mismatch")
        with mock.patch.object(VisionEngine, "_fallback_cpu") as fallback, \
                self.assertRaisesRegex(RuntimeError, "shape mismatch"):
            engine._predict("pose", "image")
        fallback.assert_not_called()


def jpeg(size=(40, 20), orientation=None):
    """An opened in-memory JPEG, optionally carrying an EXIF orientation tag."""
    buffer = io.BytesIO()
    options = {}
    if orientation is not None:
        exif = Image.Exif()
        exif[0x0112] = orientation
        options["exif"] = exif
    Image.new("RGB", size, (90, 120, 150)).save(buffer, "JPEG", **options)
    buffer.seek(0)
    return Image.open(buffer)


class OrientedSizeTests(unittest.TestCase):
    def test_orientations_6_and_8_swap_width_and_height(self):
        for orientation in (6, 8):
            with self.subTest(orientation=orientation), jpeg((40, 20), orientation) as image:
                self.assertEqual(image.size, (40, 20))
                self.assertEqual(_oriented_size(image), (20, 40))

    def test_matches_exif_transpose_for_every_orientation(self):
        for orientation in range(1, 9):
            with self.subTest(orientation=orientation), jpeg((40, 20), orientation) as image:
                expected = (20, 40) if orientation in (5, 6, 7, 8) else (40, 20)
                self.assertEqual(_oriented_size(image), expected)
                self.assertEqual(ImageOps.exif_transpose(image).size, expected)

    def test_missing_unknown_or_unreadable_orientation_keeps_size(self):
        for orientation in (None, 0, 9):
            with self.subTest(orientation=orientation), jpeg((40, 20), orientation) as image:
                self.assertEqual(_oriented_size(image), (40, 20))
        broken = SimpleNamespace(size=(40, 20), getexif=mock.Mock(side_effect=SyntaxError("corrupt EXIF")))
        self.assertEqual(_oriented_size(broken), (40, 20))

    def test_size_is_read_without_decoding_pixels(self):
        with jpeg((40, 20), 6) as image, mock.patch.object(image, "load", side_effect=AssertionError("decoded")):
            self.assertEqual(_oriented_size(image), (20, 40))


def with_index(record, index, **extra):
    return {**record, "detection_index": index, "bbox_xyxy": list(record["xyxy"]), **extra}


def rescale_experts():
    """Decoded-image detections: one person seen by three experts, one YOLOE-only, one pose-only."""
    return {
        "yolo26n": {"detections": [with_index(detection("person", (10, 20, 30, 60), .9), 0)]},
        "megadetector": {"detections": []},
        "yoloe": {"detections": [
            with_index(detection("person", (11, 21, 31, 61), .8), 0,
                       mask_polygon_xy=[[11.0, 21.0], [31.0, 21.0], [31.0, 61.0]]),
            with_index(detection("person", (200, 20, 220, 60), .7), 1, mask_polygon_xy=[[200.0, 20.0], [220.0, 60.0]]),
            with_index(detection("backpack", (15, 25, 25, 40), .5), 2, mask_polygon_xy=[])]},
        "pose": {"detections": [
            with_index(detection("person", (10.2, 20.2, 30.2, 60.2), .7), 0,
                       keypoints=[[12.5 + i, 30.25, .9] for i in range(17)]),
            with_index(detection("person", (400, 20, 420, 60), .6), 1,
                       keypoints=[[410.0, 30.0 + i, .5] for i in range(17)])]},
    }


def link_persons(experts, strip, height):
    """Person records as analyze() builds them from roster candidates (pre-rescale)."""
    persons = []
    for candidate in roster.build_candidates(experts, strip, height):
        members = candidate["members"]
        record = {"person_id": candidate["candidate_id"], **candidate, "keypoints": None, "mask_polygon_xy": None}
        if "pose" in members:
            record["keypoints"] = experts["pose"]["detections"][members["pose"]["detection_index"]]["keypoints"]
        if "yoloe" in members:
            record["mask_polygon_xy"] = experts["yoloe"]["detections"][members["yoloe"]["detection_index"]]["mask_polygon_xy"]
        persons.append(record)
    return persons


class RescaleTests(unittest.TestCase):
    def setUp(self):
        self.experts = rescale_experts()
        self.persons = link_persons(self.experts, NO_STRIP, 100)
        self.assertEqual([sorted(p["members"]) for p in self.persons],
                         [["pose", "yolo26n", "yoloe"], ["yoloe"], ["pose"]])
        self.strip = {"top": 0, "bottom": 12}
        _rescale(self.experts, self.persons, self.strip, 2.0)

    def test_detection_boxes_masks_and_keypoints(self):
        yolo = self.experts["yolo26n"]["detections"][0]
        self.assertEqual(yolo["xyxy"], [20.0, 40.0, 60.0, 120.0])
        self.assertEqual(yolo["bbox_xyxy"], yolo["xyxy"])
        self.assertIsNot(yolo["bbox_xyxy"], yolo["xyxy"])
        yoloe = self.experts["yoloe"]["detections"]
        self.assertEqual(yoloe[0]["mask_polygon_xy"], [[22.0, 42.0], [62.0, 42.0], [62.0, 122.0]])
        self.assertEqual(yoloe[2]["mask_polygon_xy"], [])  # empty mask stays empty
        self.assertEqual(yoloe[2]["xyxy"], [30.0, 50.0, 50.0, 80.0])
        pose = self.experts["pose"]["detections"][0]
        self.assertEqual(pose["xyxy"], [20.4, 40.4, 60.4, 120.4])
        self.assertEqual(pose["keypoints"], [[25.0 + 2 * i, 60.5, .9] for i in range(17)])  # confidences unscaled

    def test_person_boxes_and_members(self):
        first = self.persons[0]
        self.assertEqual(first["xyxy"], [22.0, 42.0, 62.0, 122.0])  # YOLOE is the representative box
        self.assertEqual(first["bbox_xyxy"], first["xyxy"])
        self.assertIsNot(first["bbox_xyxy"], first["xyxy"])
        self.assertEqual({name: member["xyxy"] for name, member in first["members"].items()},
                         {"yoloe": [22.0, 42.0, 62.0, 122.0], "pose": [20.4, 40.4, 60.4, 120.4],
                          "yolo26n": [20.0, 40.0, 60.0, 120.0]})
        for record in self.persons:
            for name, member in record["members"].items():
                source = self.experts[name]["detections"][member["detection_index"]]
                self.assertEqual(member["xyxy"], source["xyxy"])

    def test_persons_are_relinked_to_rescaled_keypoints_and_polygons(self):
        pose, yoloe = self.experts["pose"]["detections"], self.experts["yoloe"]["detections"]
        first, yoloe_only, pose_only = self.persons
        self.assertIs(first["keypoints"], pose[0]["keypoints"])
        self.assertIs(first["mask_polygon_xy"], yoloe[0]["mask_polygon_xy"])
        self.assertEqual(first["keypoints"][0], [25.0, 60.5, .9])
        self.assertIsNone(yoloe_only["keypoints"])
        self.assertIs(yoloe_only["mask_polygon_xy"], yoloe[1]["mask_polygon_xy"])
        self.assertEqual(yoloe_only["mask_polygon_xy"], [[400.0, 40.0], [440.0, 120.0]])
        self.assertIsNone(pose_only["mask_polygon_xy"])
        self.assertIs(pose_only["keypoints"], pose[1]["keypoints"])
        self.assertEqual(pose_only["keypoints"][3], [820.0, 66.0, .5])

    def test_strip_rows_are_rescaled(self):
        self.assertEqual(self.strip, {"top": 0, "bottom": 24})
        self.assertIs(type(self.strip["bottom"]), int)

    def test_fractional_scale_rounds_boxes_and_polygons_only(self):
        experts = {"yoloe": {"detections": [with_index(detection("person", (10, 7, 20, 13)), 0,
                                                       mask_polygon_xy=[[10.0, 7.0]])]},
                   "pose": {"detections": [with_index(detection("person", (10, 7, 20, 13)), 0,
                                                      keypoints=[[10.0, 7.0, .5]] * 17)]}}
        strip = {"top": 7, "bottom": 10}
        _rescale(experts, [], strip, 4 / 3)
        self.assertEqual(experts["yoloe"]["detections"][0]["xyxy"], [13.33, 9.33, 26.67, 17.33])
        self.assertEqual(experts["yoloe"]["detections"][0]["mask_polygon_xy"], [[13.3, 9.3]])
        self.assertAlmostEqual(experts["pose"]["detections"][0]["keypoints"][0][0], 40 / 3)
        self.assertEqual(strip, {"top": 9, "bottom": 13})

    def test_shared_lists_are_scaled_once_and_inputs_are_not_mutated(self):
        box, polygon, points = [10.0, 20.0, 30.0, 60.0], [[10.0, 20.0], [30.0, 60.0]], [[15.0, 25.0, .9]] * 17
        experts = {"yoloe": {"detections": [{"label": "person", "confidence": .9, "detection_index": 0,
                                             "xyxy": box, "bbox_xyxy": box, "mask_polygon_xy": polygon}]},
                   "pose": {"detections": [{"label": "person", "confidence": .9, "detection_index": 0,
                                            "xyxy": box, "bbox_xyxy": box, "keypoints": points}]}}
        persons = [{"xyxy": box, "bbox_xyxy": box, "keypoints": points, "mask_polygon_xy": polygon,
                    "members": {"yoloe": {"detection_index": 0, "xyxy": box}, "pose": {"detection_index": 0, "xyxy": box}}}]
        _rescale(experts, persons, dict(NO_STRIP), 2.0)
        boxes = [experts[k]["detections"][0][f] for k in ("yoloe", "pose") for f in ("xyxy", "bbox_xyxy")]
        boxes += [persons[0]["xyxy"], persons[0]["bbox_xyxy"]] + [m["xyxy"] for m in persons[0]["members"].values()]
        self.assertTrue(all(value == [20.0, 40.0, 60.0, 120.0] for value in boxes), boxes)
        self.assertEqual(persons[0]["keypoints"], [[30.0, 50.0, .9]] * 17)
        self.assertEqual(persons[0]["mask_polygon_xy"], [[20.0, 40.0], [60.0, 120.0]])
        self.assertEqual((box, polygon, points[0]), ([10.0, 20.0, 30.0, 60.0], [[10.0, 20.0], [30.0, 60.0]], [15.0, 25.0, .9]))

    def test_skipped_experts_and_missing_detection_lists_are_tolerated(self):
        experts = {"yolo26n": {"detections": []}, "megadetector": {"detections": []},
                   "yoloe": {"detections": [], "skipped": "empty_frame_gate"}, "pose": {"skipped": "empty_frame_gate"}}
        strip = {"top": 5, "bottom": 0}
        _rescale(experts, [], strip, 2.0)
        self.assertEqual(strip, {"top": 10, "bottom": 0})
        self.assertEqual(experts["yoloe"]["detections"], [])


class GateTests(unittest.TestCase):
    HEIGHT = 1000

    def gate(self, yolo=(), md=(), strip=NO_STRIP, **heavy):
        experts = {"yolo26n": {"detections": list(yolo)}, "megadetector": {"detections": list(md)}}
        experts.update({key: {"detections": list(value)} for key, value in heavy.items()})
        return bare_engine()._gate_open(experts, strip, self.HEIGHT)

    def test_nothing_from_the_cheap_experts_keeps_the_gate_closed(self):
        self.assertIs(self.gate(), False)
        # Only YOLO26n and MegaDetector are consulted, never the gated experts.
        self.assertIs(self.gate(yoloe=[detection("person", confidence=.99)], pose=[detection("person", confidence=.99)]),
                      False)

    def test_gate_confidence_is_inclusive(self):
        self.assertIs(self.gate([detection("person", confidence=GATE_CONFIDENCE)]), True)
        self.assertIs(self.gate([detection("person", confidence=GATE_CONFIDENCE - 1e-6)]), False)
        self.assertIs(self.gate([detection("person", confidence=RAW_CONFIDENCE)]), False)
        self.assertIs(self.gate(md=[detection("animal", confidence=.1999)]), False)

    def test_either_cheap_expert_and_any_label_opens_the_gate(self):
        for label in ("person", "dog", "bicycle", "backpack", "car"):
            with self.subTest(label=label):
                self.assertIs(self.gate([detection(label, confidence=.3)]), True)
        for label in MD_LABELS.values():
            with self.subTest(megadetector=label):
                self.assertIs(self.gate(md=[detection(label, confidence=.3)]), True)
        self.assertIs(self.gate([detection("person", confidence=.1)], [detection("animal", confidence=.5)]), True)

    def test_detections_in_the_data_strip_do_not_open_the_gate(self):
        bottom, top = {"top": 0, "bottom": 100}, {"top": 50, "bottom": 0}  # rows 900-1000 / 0-50
        cases = [(bottom, (0, 900, 100, 1000), False),   # entirely in the strip
                 (bottom, (0, 800, 100, 1000), False),   # exactly half inside
                 (bottom, (0, 799, 100, 1000), True),    # 49.8% inside
                 (top, (0, 0, 100, 50), False),
                 (top, (0, 0, 100, 101), True)]          # 49.5% inside
        for strip, box, expected in cases:
            with self.subTest(strip=strip, box=box):
                self.assertIs(self.gate([detection("person", box, .9)], strip=strip), expected)
                self.assertIs(self.gate(md=[detection("vehicle", box, .9)], strip=strip), expected)
        in_strip = detection("person", (0, 900, 100, 1000), .95)
        self.assertIs(self.gate([in_strip, detection("dog", (0, 0, 50, 50), .19)], strip=bottom), False)
        self.assertIs(self.gate([in_strip], [detection("animal", (0, 0, 50, 50), .2)], strip=bottom), True)

    def test_missing_strip_means_no_strip(self):
        for strip in (None, {}, NO_STRIP):
            with self.subTest(strip=strip):
                self.assertIs(self.gate([detection("person", (0, 900, 100, 1000), .9)], strip=strip), True)


class ToRgb8Tests(unittest.TestCase):
    def test_sixteen_bit_single_channel_is_rescaled(self):
        values = np.array([[0, 1000], [30000, 65535]], dtype=np.uint16)
        for image in (Image.fromarray(values), Image.fromarray(values.astype(np.int32), "I")):
            with self.subTest(mode=image.mode):
                rgb = to_rgb8(image)
                self.assertEqual(rgb.mode, "RGB")
                array = np.asarray(rgb)
                self.assertTrue(np.array_equal(array[..., 0], array[..., 2]))  # grey
                self.assertLess(int(array.min()), int(array.max()))            # not flat
                self.assertEqual(int(array[1, 1, 0]), 255)
                self.assertLessEqual(int(array[0, 0, 0]), 5)

    def test_float_images_constant_images_and_nan_do_not_crash(self):
        floats = Image.fromarray(np.array([[0.0, 0.5], [2.0, float("nan")]], dtype=np.float32), "F")
        array = np.asarray(to_rgb8(floats))
        self.assertEqual(array.shape, (2, 2, 3))
        self.assertLess(int(array[0, 0, 0]), int(array[1, 0, 0]))
        flat = np.asarray(to_rgb8(Image.fromarray(np.full((4, 4), 5000, np.uint16))))
        self.assertEqual(flat.shape, (4, 4, 3))  # no division by zero on a constant frame

    def test_eight_bit_images_are_only_converted(self):
        source = Image.new("L", (3, 2), 77)
        self.assertEqual(np.asarray(to_rgb8(source)).tolist(), np.asarray(source.convert("RGB")).tolist())
        rgba = Image.new("RGBA", (3, 2), (10, 20, 30, 40))
        self.assertEqual(to_rgb8(rgba).getpixel((0, 0)), (10, 20, 30))


class AnalyzeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)

    def write_image(self, name, size, orientation=None, strip_rows=0, image_format="JPEG"):
        array = np.full((size[1], size[0], 3), (120, 130, 110), np.uint8)
        if strip_rows:
            array[-strip_rows:] = 0  # black camera info bar at the bottom
        options = {"quality": 90} if image_format == "JPEG" else {}
        if orientation is not None:
            exif = Image.Exif()
            exif[0x0112] = orientation
            options["exif"] = exif
        path = self.folder / name
        Image.fromarray(array).save(path, image_format, **options)
        return path

    def analyze(self, path, results, **attributes):
        engine = bare_engine(**attributes)
        calls = []

        def predict(key, image):
            calls.append({"expert": key, "shape": image.shape, "threads": engine.torch.threads})
            return results[key]

        engine._predict = predict
        return engine, engine.analyze(path), calls

    def test_analyze_restores_torch_threads_before_decoding(self):
        torch = FakeTorch(1)  # e.g. PaddlePaddle reset the shared pool to one thread
        engine = bare_engine(torch=torch, threads=6)
        with self.assertRaises(FileNotFoundError):
            engine.analyze(self.folder / "missing.jpg")
        self.assertEqual(torch.set_calls, [6])

    def test_analyze_restores_threads_on_every_image_before_inference(self):
        path = self.write_image("small.jpg", (64, 48))
        torch = FakeTorch(1)
        engine, _, calls = self.analyze(path, empty_results(), torch=torch, threads=6)
        self.assertEqual({call["threads"] for call in calls}, {6})
        torch.threads = 1  # reset again between images
        engine.analyze(path)
        self.assertEqual(torch.set_calls, [6, 6])

    def test_analyze_leaves_a_matching_thread_count_alone(self):
        path = self.write_image("small.jpg", (64, 48))
        engine, _, _ = self.analyze(path, empty_results(), threads=4)
        self.assertEqual(engine.torch.set_calls, [])

    def test_empty_frame_gate_skips_heavy_experts(self):
        path = self.write_image("empty.jpg", (64, 48))
        results = empty_results()
        results["yolo26n"] = fake_result(COCO_NAMES, [(16, [5, 5, 20, 20], .19)])  # dog below the gate
        _, result, calls = self.analyze(path, results)
        self.assertEqual([call["expert"] for call in calls], ["yolo26n", "megadetector"])
        self.assertIs(result["gated_empty"], True)
        self.assertEqual(result["persons"], [])
        for key in ("yoloe", "pose"):
            with self.subTest(expert=key):
                expert = result["experts"][key]
                self.assertEqual(expert["skipped"], "empty_frame_gate")
                self.assertEqual((expert["detections"], expert["inference_seconds"], expert["checkpoint"], expert["imgsz"]),
                                 ([], 0.0, WEIGHTS[key], PROFILES["standard"][key]))
                self.assertEqual(expert["counts"], count_objects([], key))
        self.assertEqual((result["width"], result["height"], result["analysis_scale"]), (64, 48, 1.0))

    def test_disabled_gate_runs_every_expert(self):
        path = self.write_image("empty.jpg", (64, 48))
        _, result, calls = self.analyze(path, empty_results(), empty_frame_gate=False, profile="fast",
                                        sizes=dict(PROFILES["fast"]))
        self.assertEqual([call["expert"] for call in calls], ["yolo26n", "megadetector", "yoloe", "pose"])
        self.assertIs(result["gated_empty"], False)
        self.assertEqual({key: expert["imgsz"] for key, expert in result["experts"].items()}, PROFILES["fast"])
        self.assertEqual(result["profile"], "fast")

    def test_gated_frame_builds_no_candidates(self):
        # v2: a closed gate means "nothing here"; a faint cheap-expert person
        # (above the raw floor, below the gate) is no longer a candidate.
        path = self.write_image("faint.jpg", (64, 48))
        results = empty_results()
        results["yolo26n"] = fake_result(COCO_NAMES, [(0, [5, 5, 30, 45], .15)])
        self.assertGreaterEqual(.15, RAW_CONFIDENCE)
        self.assertEqual(len(roster.build_candidates({"yolo26n": {"detections": [
            {"label": "person", "confidence": .15, "xyxy": [5, 5, 30, 45]}]}})), 1)  # would be one if built
        _, result, calls = self.analyze(path, results)
        self.assertEqual([call["expert"] for call in calls], ["yolo26n", "megadetector"])
        self.assertIs(result["gated_empty"], True)
        self.assertEqual(result["experts"]["yolo26n"]["detections"][0]["confidence"], .15)  # raw detection kept
        self.assertEqual(result["persons"], [])

    def test_sixteen_bit_png_is_rescaled_not_clipped(self):
        # A plain convert("RGB") clips 16-bit values above 255 to white; to_rgb8 rescales.
        ramp = np.tile(np.linspace(1000, 60000, 64, dtype=np.uint16), (48, 1))
        path = self.folder / "thermal.png"
        Image.fromarray(ramp).save(path)
        with Image.open(path) as source:
            self.assertIn(source.mode, ("I;16", "I;16B", "I;16L", "I"))
            self.assertEqual(int(np.asarray(source.convert("RGB")).min()), 255)  # the old, flat decode
        _, result, calls = self.analyze(path, empty_results())
        decoded = np.asarray(result["_decoded_image"])
        self.assertEqual((result["_decoded_image"].mode, decoded.shape), ("RGB", (48, 64, 3)))
        self.assertLess(int(decoded.min()), 10)
        self.assertGreater(int(decoded.max()), 245)
        self.assertTrue(np.all(np.diff(decoded[0, :, 0].astype(int)) >= 0))  # the ramp survives
        self.assertEqual(calls[0]["shape"], (48, 64, 3))

    def test_detection_in_the_camera_data_strip_keeps_the_gate_closed(self):
        path = self.write_image("strip.jpg", (640, 480), strip_rows=30)
        results = empty_results()
        results["yolo26n"] = fake_result(COCO_NAMES, [(0, [300, 455, 340, 478], .95)])  # camera text read as a person
        _, result, calls = self.analyze(path, results)
        self.assertEqual(result["data_strip"], {"top": 0, "bottom": 30})
        self.assertEqual([call["expert"] for call in calls], ["yolo26n", "megadetector"])
        self.assertIs(result["gated_empty"], True)
        self.assertEqual(result["persons"], [])  # the roster drops strip boxes too

    def test_large_jpeg_is_draft_decoded_and_rescaled_to_original_pixels(self):
        path = self.write_image("big.jpg", (3200, 2400), strip_rows=120)
        keypoints = [[150.0 + i, 200.0 + 5 * i, .9] for i in range(17)]
        results = empty_results()
        results["yolo26n"] = fake_result(COCO_NAMES, [(0, [100, 100, 200, 400], .9),
                                                      (0, [700, 1150, 760, 1195], .95)])  # inside the strip
        results["yoloe"] = fake_result(YOLOE_NAMES, [(0, [102, 98, 204, 402], .8)],
                                       polygons=[np.array([[102, 98], [204, 98], [204, 402], [102, 402]], np.float32)])
        results["pose"] = fake_result(POSE_NAMES, [(0, [100, 101, 201, 400], .7)], keypoints=[keypoints])
        _, result, calls = self.analyze(path, results)
        self.assertEqual({call["shape"] for call in calls}, {(1200, 1600, 3)})  # half-scale decode
        self.assertEqual([call["expert"] for call in calls], ["yolo26n", "megadetector", "yoloe", "pose"])
        self.assertEqual((result["width"], result["height"], result["analysis_scale"]), (3200, 2400, 2.0))
        self.assertEqual(result["_decoded_image"].size, (1600, 1200))
        self.assertEqual(result["data_strip"], {"top": 0, "bottom": 120})
        self.assertIs(result["gated_empty"], False)
        experts = result["experts"]
        self.assertEqual(experts["yolo26n"]["detections"][1]["xyxy"], [1400.0, 2300.0, 1520.0, 2390.0])
        self.assertEqual(len(result["persons"]), 1)  # the strip "person" is not a candidate
        found = result["persons"][0]
        self.assertEqual(found["xyxy"], [204.0, 196.0, 408.0, 804.0])
        self.assertEqual({name: member["xyxy"] for name, member in found["members"].items()},
                         {"yoloe": [204.0, 196.0, 408.0, 804.0], "pose": [200.0, 202.0, 402.0, 800.0],
                          "yolo26n": [200.0, 200.0, 400.0, 800.0]})
        self.assertIs(found["keypoints"], experts["pose"]["detections"][0]["keypoints"])
        self.assertEqual(found["keypoints"], [[300.0 + 2 * i, 400.0 + 10 * i, .9] for i in range(17)])
        self.assertIs(found["mask_polygon_xy"], experts["yoloe"]["detections"][0]["mask_polygon_xy"])
        self.assertEqual(found["mask_polygon_xy"], [[204.0, 196.0], [408.0, 196.0], [408.0, 804.0], [204.0, 804.0]])
        self.assertEqual(found["pose_confidence"], .7)
        self.assertIsInstance(found["orientation_evidence"], dict)
        self.assertIs(found["appearance"]["valid"], True)

    def test_filtered_boxes_preserve_person_mask_and_pose_identity(self):
        # A clipped/degenerate raw box can be removed before a valid person.
        # Raw IDs 1 and 2 must not be used as positions in the two-item list.
        for size, scale in (((300, 220), 1), ((3200, 2400), 2)):
            with self.subTest(scale=scale):
                path = self.write_image(f"filtered_{scale}.jpg", size)
                boxes = [[0, 0, 0, 0], [20, 20, 80, 190], [160, 20, 220, 190]]
                polygons = [[], [[20, 20], [80, 20], [80, 190], [20, 190]],
                            [[160, 20], [220, 20], [220, 190], [160, 190]]]
                points = [[[30 + index * 100, 50 + point, .9] for point in range(17)]
                          for index in range(3)]
                results = empty_results()
                detections = [(0, box, .9 - index * .1) for index, box in enumerate(boxes)]
                results["yoloe"] = fake_result(YOLOE_NAMES, detections, polygons=polygons)
                results["pose"] = fake_result(POSE_NAMES, detections, keypoints=points)
                _, result, _ = self.analyze(path, results, empty_frame_gate=False)
                self.assertEqual(len(result["persons"]), 2)
                for person, raw_id in zip(result["persons"], (1, 2)):
                    self.assertEqual(person["members"]["pose"]["detection_index"], raw_id)
                    self.assertEqual(person["members"]["yoloe"]["detection_index"], raw_id)
                    self.assertEqual(person["keypoints"],
                                     [[x * scale, y * scale, confidence] for x, y, confidence in points[raw_id]])
                    self.assertEqual(person["mask_polygon_xy"],
                                     [[x * scale, y * scale] for x, y in polygons[raw_id]])
                    self.assertIs(person["keypoints"], result["experts"]["pose"]["detections"][raw_id - 1]["keypoints"])
                    self.assertIs(person["mask_polygon_xy"], result["experts"]["yoloe"]["detections"][raw_id - 1]["mask_polygon_xy"])

    def test_rotated_large_jpeg_reports_the_oriented_full_size(self):
        path = self.write_image("rotated.jpg", (3200, 2400), orientation=6)
        _, result, calls = self.analyze(path, empty_results())
        self.assertEqual((result["width"], result["height"], result["analysis_scale"]), (2400, 3200, 2.0))
        self.assertEqual(calls[0]["shape"], (1600, 1200, 3))

    def test_draft_decoding_starts_at_the_long_side_threshold_for_jpeg_only(self):
        cases = [("at.jpg", (DRAFT_MIN_LONG_SIDE, 64), "JPEG", 2.0),
                 ("below.jpg", (DRAFT_MIN_LONG_SIDE - 1, 64), "JPEG", 1.0),
                 ("tall.jpg", (64, DRAFT_MIN_LONG_SIDE), "JPEG", 2.0),
                 ("lossless.png", (DRAFT_MIN_LONG_SIDE, 64), "PNG", 1.0)]
        for name, size, image_format, scale in cases:
            with self.subTest(name=name):
                path = self.write_image(name, size, image_format=image_format)
                _, result, calls = self.analyze(path, empty_results())
                self.assertEqual((result["width"], result["height"]), size)
                self.assertEqual(result["analysis_scale"], scale)
                self.assertEqual(calls[0]["shape"][:2], (round(size[1] / scale), round(size[0] / scale)))


if __name__ == "__main__":
    unittest.main()
