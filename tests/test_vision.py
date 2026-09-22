"""Portable detector invariants; these do not estimate real-image accuracy."""
from types import SimpleNamespace
import unittest

from trailcam.vision import (assign_bag, compact_polygon, count_objects,
                             select_device, semantic_vehicle_dedup)


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


if __name__ == "__main__":
    unittest.main()
