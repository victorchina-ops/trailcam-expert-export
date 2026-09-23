"""Appearance descriptor invariants on synthetic images only (no photos, no models)."""
import colorsys
from collections import Counter
import importlib.util
import json
from types import SimpleNamespace
import unittest
from unittest import mock
import warnings

import numpy as np

from trailcam import appearance
from trailcam.appearance import BIN_COUNT, POLICY, VERSION, descriptor, distance, resolve_device

BOX = (100, 40, 160, 220)
RED, BLUE, TROUSERS = (190, 40, 35), (40, 70, 180), (45, 55, 110)


def scene(box=BOX, shirt=RED, trousers=TROUSERS, background=(90, 125, 60), size=(240, 320),
          seed=0, noise=10.0, margin=0.0):
    """Noisy person: skin head, shirt to 50 % of box height, trousers below.

    ``margin`` narrows the body to the middle of the box, leaving background at the sides.
    """
    rng = np.random.default_rng(seed)
    image = np.empty(size + (3,))
    image[:] = background
    x1, y1, x2, y2 = box
    left, right = int(round(x1 + margin * (x2 - x1))), int(round(x2 - margin * (x2 - x1)))
    for top, bottom, colour in ((0.0, 0.14, (205, 160, 130)), (0.14, 0.5, shirt), (0.5, 1.0, trousers)):
        image[int(round(y1 + top * (y2 - y1))):int(round(y1 + bottom * (y2 - y1))), left:right] = colour
    image += rng.normal(0, noise, image.shape)
    return np.clip(np.rint(image), 0, 255).astype(np.uint8)


def uniform(colour, size=(100, 100)):
    image = np.empty(size + (3,), dtype=np.uint8)
    image[:] = colour
    return image


def cpu(*args, **kwargs):
    return descriptor(*args, device="cpu", **kwargs)


def plain(value):
    """True when a value tree holds only JSON-native Python types."""
    if isinstance(value, dict):
        return all(type(k) is str and plain(v) for k, v in value.items())
    if isinstance(value, list):
        return all(plain(v) for v in value)
    return value is None or type(value) in (bool, int, float, str)


def tie(x):
    return abs(x - round(x)) < 1e-9


class DescriptorTests(unittest.TestCase):
    def test_output_is_json_native_rounded_and_normalised(self):
        d = cpu(scene(), BOX)
        self.assertEqual(list(d), ["version", "upper", "lower", "valid", "pixels"])
        self.assertEqual(d["version"], VERSION)
        self.assertIs(d["valid"], True)
        self.assertTrue(plain(d))
        json.dumps(d)
        for part in ("upper", "lower"):
            self.assertEqual(len(d[part]), BIN_COUNT)
            self.assertTrue(all(v == round(v, 5) and 0 <= v <= 1 for v in d[part]))
            self.assertAlmostEqual(sum(d[part]), 1.0, delta=BIN_COUNT * 5e-6)
            self.assertIs(type(d["pixels"][part]), int)

    def test_identical_crops_have_zero_distance_and_are_deterministic(self):
        image = scene()
        a, b = cpu(image, BOX), cpu(image.copy(), list(BOX))
        self.assertEqual(a, b)
        self.assertEqual(distance(a, b), 0.0)

    def test_red_and_blue_shirts_are_far_apart(self):
        red, blue = cpu(scene(shirt=RED), BOX), cpu(scene(shirt=BLUE, seed=2), BOX)
        self.assertGreater(distance(red, blue), 0.45)
        top = [[100, 40], [160, 40], [160, 130], [100, 130]]  # mask ends above the lower region
        red_top, blue_top = cpu(scene(shirt=RED), BOX, mask=top), cpu(scene(shirt=BLUE, seed=2), BOX, mask=top)
        self.assertIsNone(red_top["lower"])
        self.assertGreater(distance(red_top, blue_top), 0.9)

    def test_same_person_shifted_and_scaled_is_close(self):
        moved = (140, 30, 206, 228)  # shifted and ~10 % larger, different noise
        a, b = cpu(scene(), BOX), cpu(scene(moved, seed=1), moved)
        other = cpu(scene(shirt=BLUE, seed=2), BOX)
        self.assertLess(distance(a, b), 0.1)
        self.assertLess(distance(a, b) * 5, distance(a, other))

    def test_mask_restriction_changes_result_and_removes_background(self):
        green = scene(margin=0.25, background=(90, 125, 60), seed=3)
        brown = scene(margin=0.25, background=(140, 110, 80), seed=4)
        raster = np.zeros(green.shape[:2], dtype=bool)
        raster[40:220, 115:145] = True
        polygon = [[115, 40], [145, 40], [145, 220], [115, 220]]
        self.assertNotEqual(cpu(green, BOX), cpu(green, BOX, mask=raster))
        self.assertGreater(distance(cpu(green, BOX), cpu(brown, BOX)), 0.4)
        self.assertLess(distance(cpu(green, BOX, mask=raster), cpu(brown, BOX, mask=raster)), 0.1)
        self.assertEqual(cpu(green, BOX, mask=polygon), cpu(green, BOX, mask=raster))
        self.assertEqual(cpu(green, BOX, mask=raster.astype(np.uint8) * 255), cpu(green, BOX, mask=raster))

    def test_polygon_masks_are_filled_at_pixel_centres(self):
        image, box = uniform((100, 150, 200)), [0, 0, 100, 100]  # upper rows 15-49, lower 50-89
        self.assertEqual(cpu(image, box)["pixels"], {"upper": 3500, "lower": 4000})
        square = [[10, 20], [30, 20], [30, 40], [10, 40]]
        d = cpu(image, box, mask=square)
        self.assertEqual(d["pixels"], {"upper": 400, "lower": 0})
        self.assertIsNone(d["lower"])
        self.assertIs(d["valid"], True)
        straddle = [[0, 40], [10, 40], [10, 60], [0, 60]]
        self.assertEqual(cpu(image, box, mask=straddle)["pixels"], {"upper": 100, "lower": 100})
        overlapping = [square, [[20, 20], [40, 20], [40, 40], [20, 40]]]
        self.assertEqual(cpu(image, box, mask=overlapping)["pixels"]["upper"], 600)  # union, not XOR
        as_arrays = [np.array(square, dtype=np.float32)]
        self.assertEqual(cpu(image, box, mask=as_arrays), d)
        triangle = cpu(image, box, mask=[(0, 50), (60, 50), (0, 90)])["pixels"]["lower"]
        self.assertAlmostEqual(triangle, 1200, delta=40)
        outside = [[-50, -50], [150, -50], [150, 150], [-50, 150]]
        self.assertEqual(cpu(image, box, mask=outside), cpu(image, box))
        for empty in ([], [[1, 1], [5, 5]], [[]]):
            self.assertIs(cpu(image, box, mask=empty)["valid"], False)

    def test_malformed_masks_raise(self):
        image = uniform((100, 150, 200))
        for mask in (np.ones((50, 50), dtype=bool), np.ones((1, 100, 100), dtype=bool),
                     [[0, 0], [10, 0], [float("nan"), 10]], [[0, 0, 0], [1, 1, 1], [2, 2, 2]], [1.0, 2.0]):
            with self.assertRaises(ValueError):
                cpu(image, [0, 0, 100, 100], mask=mask)

    def test_tiny_box_is_invalid(self):
        d = cpu(scene(), [120, 60, 124, 66])
        self.assertIs(d["valid"], False)
        self.assertIsNone(d["upper"])
        self.assertIsNone(d["lower"])
        self.assertLess(d["pixels"]["upper"], POLICY["min_pixels"])
        self.assertIsNone(distance(d, cpu(scene(), BOX)))

    def test_box_partly_outside_image_is_clamped(self):
        image = scene(size=(100, 100), box=(0, 0, 60, 100))
        self.assertEqual(cpu(image, [-50, -20, 60, 120]), cpu(image, [0, 0, 60, 100]))
        self.assertEqual(cpu(image, [40.0, -30.0, 180.0, 100.0]), cpu(image, [40, 0, 100, 100]))

    def test_region_rows_follow_box_height(self):
        d = cpu(uniform((100, 150, 200)), [10, 20, 30, 70])  # rows 28-44 and 45-64, cols 10-29
        self.assertEqual(d["pixels"], {"upper": 17 * 20, "lower": 20 * 20})

    def test_degenerate_boxes_are_invalid(self):
        image = scene()
        for box in (None, [10, 10, 10, 50], [10, 10, 50, 10], [50, 10, 10, 60], [float("nan"), 0, 50, 50],
                    [0, 0, float("inf"), 50], [400, 300, 500, 400], [-80, -80, -10, -10], [10.2, 10, 10.4, 200]):
            d = cpu(image, box)
            self.assertIs(d["valid"], False, box)
            self.assertEqual(d["pixels"], {"upper": 0, "lower": 0}, box)

    def test_malformed_boxes_raise(self):
        for box in ([0, 0, 10], [0, 0, 10, 10, 5]):
            with self.assertRaises(ValueError):
                cpu(scene(), box)

    def test_dark_pixels_are_ignored_at_threshold(self):
        box = [0, 0, 100, 100]
        self.assertIs(cpu(uniform((19, 19, 19)), box)["valid"], False)
        self.assertIs(cpu(uniform((19, 5, 5)), box)["valid"], False)
        self.assertEqual(cpu(uniform((20, 20, 20)), box)["pixels"], {"upper": 3500, "lower": 4000})
        image = uniform((200, 60, 60))
        image[15:30] = 0  # 15 dark rows of the 35 upper rows
        self.assertEqual(cpu(image, box)["pixels"]["upper"], 2000)

    def test_minimum_usable_pixels_boundary(self):
        image, box = uniform((100, 150, 200)), [0, 0, 100, 100]
        for count, valid in ((POLICY["min_pixels"], True), (POLICY["min_pixels"] - 1, False)):
            mask = np.zeros((100, 100), dtype=bool)
            mask[20, :count] = True
            d = cpu(image, box, mask=mask)
            self.assertEqual(d["pixels"], {"upper": count, "lower": 0})
            self.assertIs(d["valid"], valid)
            self.assertEqual(d["upper"] is not None, valid)
            self.assertIsNone(d["lower"])

    def test_parts_are_assessed_independently(self):
        dark_legs = uniform((200, 40, 40))
        dark_legs[50:] = 0
        a = cpu(dark_legs, [0, 0, 100, 100])
        self.assertIsNotNone(a["upper"])
        self.assertIsNone(a["lower"])
        self.assertIs(a["valid"], True)
        blue_legs = uniform((200, 40, 40))
        blue_legs[50:] = (40, 40, 200)
        self.assertEqual(distance(a, cpu(blue_legs, [0, 0, 100, 100])), 0.0)

    def test_known_colour_bins(self):
        # index = (h * 4 + s) * 4 + v with 16 hue bins of 22.5 degrees, S and V quartiles.
        cases = {(255, 0, 0): 15, (255, 0, 10): 255, (0, 255, 0): 95, (0, 0, 255): 175,
                 (255, 255, 0): 47, (128, 128, 128): 2, (255, 255, 255): 3, (63, 63, 63): 0,
                 (64, 64, 64): 1, (200, 100, 100): 11, (200, 150, 150): 7, (200, 151, 151): 3}
        for colour, index in cases.items():
            upper = cpu(uniform(colour), [0, 0, 100, 100])["upper"]
            self.assertEqual(upper[index], 1.0, colour)

    def test_binning_matches_float_hsv_reference(self):
        rng = np.random.default_rng(7)
        pixels, expected = [], Counter()
        for r, g, b in rng.integers(0, 256, (8000, 3)).tolist():
            h, s, _ = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
            if max(r, g, b) < POLICY["min_value"] or tie(h * 16) or tie(s * 4):
                continue  # float rounding at exact bin edges is not a fair reference
            pixels.append((r, g, b))
            expected[(int(h * 16) * 4 + min(int(s * 4), 3)) * 4 + max(r, g, b) // 64] += 1
            if len(pixels) == 3500:
                break
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[15:50] = np.array(pixels, dtype=np.uint8).reshape(35, 100, 3)
        d = cpu(image, [0, 0, 100, 100])
        self.assertEqual(d["pixels"]["upper"], 3500)
        self.assertEqual(d["upper"], [round(expected[i] / 3500, 5) for i in range(BIN_COUNT)])

    def test_greyscale_rgba_and_float_inputs(self):
        rgb = scene()
        grey = rgb.mean(axis=2).round().astype(np.uint8)
        self.assertEqual(cpu(grey, BOX), cpu(np.repeat(grey[..., None], 3, axis=2), BOX))
        rgba = np.concatenate([rgb, np.full(rgb.shape[:2] + (1,), 7, dtype=np.uint8)], axis=2)
        self.assertEqual(cpu(rgba, BOX), cpu(rgb, BOX))
        self.assertEqual(cpu(rgb.astype(np.float64), BOX), cpu(rgb, BOX))
        night_a, night_b = cpu(grey, BOX), cpu(np.clip(grey.astype(int) + 60, 0, 255).astype(np.uint8), BOX)
        self.assertGreater(distance(night_a, night_b), 0.3)

    def test_invalid_image_shapes_raise(self):
        for image in (np.zeros(10, dtype=np.uint8), np.zeros((10, 10, 2), dtype=np.uint8),
                      np.zeros((2, 10, 10, 3), dtype=np.uint8)):
            with self.assertRaises(ValueError):
                cpu(image, [0, 0, 5, 5])


class DistanceTests(unittest.TestCase):
    def setUp(self):
        self.red = cpu(uniform((200, 40, 40)), [0, 0, 100, 100])
        self.blue = cpu(uniform((40, 40, 200)), [0, 0, 100, 100])

    def test_missing_or_invalid_descriptors_give_none(self):
        invalid = cpu(uniform((0, 0, 0)), [0, 0, 100, 100])
        for a, b in ((None, self.red), (self.red, None), (None, None), ({}, self.red),
                     (invalid, self.red), (self.red, invalid), ("hsv_v1", self.red)):
            self.assertIsNone(distance(a, b))

    def test_version_mismatch_and_malformed_histograms(self):
        self.assertIsNone(distance({**self.red, "version": "hsv_v0"}, self.red))
        self.assertIsNone(distance({**self.red, "valid": 1}, self.red))
        broken = {**self.red, "upper": [1.0] * 10, "lower": None}
        self.assertIsNone(distance(broken, self.red))
        for bad in ([float("nan")] * BIN_COUNT, [-1.0] + [0.0] * (BIN_COUNT - 1), [0.0] * BIN_COUNT, ["x"] * BIN_COUNT):
            only_lower = {**self.red, "upper": bad}
            self.assertEqual(distance(only_lower, self.blue), 1.0)  # falls back to the lower part

    def test_range_symmetry_and_extremes(self):
        self.assertEqual(distance(self.red, self.blue), 1.0)
        self.assertEqual(distance(self.red, self.red), 0.0)
        people = [cpu(scene(shirt=tuple(c), seed=i), BOX)
                  for i, c in enumerate(np.random.default_rng(3).integers(0, 256, (6, 3)).tolist())]
        for a in people:
            for b in people:
                value = distance(a, b)
                self.assertIs(type(value), float)
                self.assertTrue(0.0 <= value <= 1.0)
                self.assertEqual(value, distance(b, a))

    def test_only_parts_valid_in_both_are_compared(self):
        top_only = {**self.red, "lower": None}
        bottom_only = {**self.blue, "upper": None}
        self.assertEqual(distance(top_only, self.red), 0.0)
        self.assertIsNone(distance(top_only, bottom_only))
        mixed = {**self.red, "lower": self.blue["lower"]}
        self.assertEqual(distance(mixed, self.red), 0.5)

    def test_normalisation_absorbs_rounding_and_scale(self):
        scaled = {**self.red, "upper": [2 * v for v in self.red["upper"]]}
        self.assertEqual(distance(scaled, self.red), 0.0)


def strict(function, *args, **kwargs):
    """Call with every warning turned into an error (NumPy casts, overflow, deprecations)."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return function(*args, **kwargs)


def scaled(d, factor):
    return {**d, **{part: [v * factor for v in d[part]] for part in ("upper", "lower") if d[part] is not None}}


class AdversarialDescriptorTests(unittest.TestCase):
    """Malformed, extreme and degenerate inputs: never crash on data, never return NumPy types."""

    def test_non_finite_float_pixels_are_ignored_without_warnings(self):
        box = [0, 0, 100, 100]
        base = uniform((200, 50, 50)).astype(np.float64)
        rows_masked = np.ones((100, 100), dtype=bool)
        rows_masked[20:30] = False
        expected = cpu(uniform((200, 50, 50)), box, mask=rows_masked)
        self.assertEqual(expected["pixels"], {"upper": 2500, "lower": 4000})
        for value, channels in ((np.nan, slice(None)), (np.nan, 1), (np.inf, 0), (-np.inf, 2)):
            image = base.copy()
            image[20:30, :, channels] = value
            self.assertEqual(strict(cpu, image, box), expected, (value, channels))
        grey = np.full((100, 100), 120.0)
        grey[20:30] = np.nan
        self.assertEqual(strict(cpu, grey, box)["pixels"], {"upper": 2500, "lower": 4000})
        self.assertIs(strict(cpu, np.full((100, 100, 3), np.nan), box)["valid"], False)

    def test_non_numeric_images_raise(self):
        for image in (np.full((20, 20, 3), 200, dtype=object), np.full((20, 20, 3), "x"),
                      np.full((20, 20, 3), 1 + 2j), None, "image"):
            with self.assertRaises(ValueError):
                cpu(image, [0, 0, 20, 20])

    def test_bool_and_wide_integer_images_are_clipped(self):
        box = [0, 0, 100, 100]
        self.assertIs(cpu(np.ones((100, 100, 3), dtype=bool), box)["valid"], False)  # 0/1 values are dark
        bright = uniform((255, 255, 255))
        self.assertEqual(cpu(np.full((100, 100, 3), 60000, dtype=np.uint16), box), cpu(bright, box))
        self.assertEqual(cpu(np.full((100, 100, 3), -5, dtype=np.int64), box)["pixels"], {"upper": 0, "lower": 0})
        self.assertEqual(cpu(uniform((200, 50, 50)).astype(np.float32) + 0.4, box), cpu(uniform((200, 50, 50)), box))

    def test_none_and_overflowing_coordinates_give_invalid_descriptors(self):
        image = scene()
        for box in ([None, 40, 160, 220], [100, 40, 160, None], [None] * 4, [0, 0, 10 ** 400, 220],
                    [-(10 ** 400), 0, 50, 50], np.array([np.nan, 0, 50, 50])):
            d = strict(cpu, image, box)
            self.assertEqual((d["valid"], d["pixels"]), (False, {"upper": 0, "lower": 0}), box)

    def test_box_container_types_are_equivalent(self):
        image, expected = scene(), cpu(scene(), BOX)
        for box in (tuple(BOX), np.array(BOX, dtype=np.float32), np.array(BOX, dtype=np.int64),
                    (v for v in BOX), [np.float64(v) for v in BOX], [float(v) for v in BOX]):
            self.assertEqual(strict(cpu, image, box), expected)
        huge = [-1e308, -1e308, 1e308, 1e308]
        self.assertEqual(cpu(image, huge), cpu(image, [0, 0, 320, 240]))
        self.assertEqual(cpu(image, [-0.0, -0.0, 320, 240]), cpu(image, [0, 0, 320, 240]))

    def test_malformed_box_types_raise_value_error(self):
        image = scene()
        for box in ("1234", b"1234", 5, 5.0, {"a": 1, "b": 2, "c": 3, "d": 4}, [[0, 0, 10, 10]],
                    np.array([[0, 0, 10, 10]]), np.zeros((4, 1)), [0, 0, "10", 10], [0, 0, [1], 10],
                    [0, 0, object(), 10], [], [0, 0, 10, 10, None]):
            with self.assertRaises(ValueError, msg=repr(box)):
                strict(cpu, image, box)

    def test_fractional_box_edges_use_pixel_centres(self):
        image = uniform((100, 150, 200))
        image[:, 0] = 0  # dark first column is ignored when included
        self.assertEqual(cpu(image, [0.5, 0, 10.5, 100])["pixels"]["upper"], 10 * 35)  # columns 1-10
        self.assertEqual(cpu(image, [0.49, 0, 10.49, 100])["pixels"]["upper"], 9 * 35)  # columns 0-9
        self.assertEqual(cpu(image, [1.5, 0, 1.51, 100])["pixels"], {"upper": 0, "lower": 0})

    def test_thin_and_single_pixel_images(self):
        for size, box in (((1, 1), [0, 0, 1, 1]), ((1, 500), [0, 0, 500, 1]), ((500, 1), [0, 0, 1, 500]),
                          ((0, 10), [0, 0, 10, 10]), ((10, 0), [0, 0, 10, 10])):
            d = strict(cpu, uniform((100, 150, 200), size), box)
            self.assertTrue(plain(d))
            self.assertIs(d["valid"], size in ((1, 500), (500, 1)), size)
        self.assertEqual(cpu(uniform((100, 150, 200), (500, 1)), [0, 0, 1, 500])["pixels"], {"upper": 175, "lower": 200})
        self.assertEqual(cpu(uniform((100, 150, 200), (1, 500)), [0, 0, 500, 1])["pixels"], {"upper": 500, "lower": 0})

    def test_extreme_polygon_coordinates_do_not_crash(self):
        image, box = uniform((100, 150, 200)), [0, 0, 100, 100]
        everything = [[-1e308, -1e308], [1e308, -1e308], [1e308, 1e308], [-1e308, 1e308]]
        self.assertEqual(strict(cpu, image, box, mask=everything), cpu(image, box))
        full = cpu(image, box)["pixels"]
        for polygon in ([[-1.7e308, 50.5], [1.7e308, 80], [1.7e308, 20]],  # a vertex on a pixel-centre row
                        [[-1.7e308, 15.5], [1.7e308, 89.5], [0, 89.5]], [[0, -1.7e308], [50, 1.7e308], [100, 50.5]]):
            first, second = strict(cpu, image, box, mask=polygon), strict(cpu, image, box, mask=polygon)
            self.assertEqual(first, second)
            self.assertTrue(all(0 <= first["pixels"][p] <= full[p] for p in full))

    def test_polygon_order_orientation_and_closure_do_not_matter(self):
        image, box = scene(margin=0.2, seed=6), BOX
        body = [[112, 45], [150, 44], [156, 150], [140, 219], [118, 218], [104, 140]]
        extra = [[100, 180], [130, 170], [160, 215], [105, 219]]
        expected = cpu(image, box, mask=[body, extra])
        for mask in ([extra, body], [body[::-1], extra], [body[3:] + body[:3], extra[::-1]],
                     [body + [body[0]], extra + [extra[0]]], [np.array(body), tuple(map(tuple, extra))]):
            self.assertEqual(cpu(image, box, mask=mask), expected)

    def test_polygon_fill_does_not_depend_on_vertex_order_at_pixel_centre_ties(self):
        # 0.1-px triangles (like mask_polygon_xy) whose edges pass exactly through pixel centres;
        # computing each crossing from the edge's own start vertex used to give 733 vs 732 pixels.
        for triangle in ([[20.4, 0.8], [55.9, 19.3], [50.6, 57.7]], [[39.3, 30.7], [15.2, 59.6], [53.3, 46.9]],
                         [[20.1, 40.4], [21.5, 6.8], [1.5, 52.8]]):
            points = np.array(triangle)
            reference = appearance._fill_polygons([points], 0, 60, 0, 60)
            self.assertTrue(np.array_equal(reference, appearance._fill_polygons([points[::-1].copy()], 0, 60, 0, 60)))
        rng = np.random.default_rng(0)
        for _ in range(300):
            points = np.round(rng.uniform(0, 60, (int(rng.integers(3, 7)), 2)) * 10) / 10
            reference = appearance._fill_polygons([points], 0, 60, 0, 60)
            for variant in (points[::-1].copy(), np.roll(points, int(rng.integers(1, len(points))), axis=0)):
                self.assertTrue(np.array_equal(reference, appearance._fill_polygons([variant], 0, 60, 0, 60)))

    def test_inputs_are_not_mutated(self):
        image, box = scene(), list(BOX)
        raster = np.zeros(image.shape[:2], dtype=np.uint8)
        raster[50:200, 110:150] = 1
        polygon = [[110, 50], [150, 50], [150, 200], [110, 200]]
        copies = (image.copy(), list(box), raster.copy(), [list(p) for p in polygon])
        read_only = image.copy()
        read_only.setflags(write=False)
        cpu(read_only, box, mask=raster)
        cpu(image, box, mask=polygon)
        cpu(image.astype(np.float64), box)
        np.testing.assert_array_equal(image, copies[0])
        self.assertEqual((box, polygon), (copies[1], copies[3]))
        np.testing.assert_array_equal(raster, copies[2])

    def test_device_and_mask_are_validated_even_when_the_box_is_empty(self):
        with self.assertRaises(ValueError):
            descriptor(scene(), None, device="tpu")
        with self.assertRaises(ValueError):
            descriptor(scene(), [0, 0, 0, 0], device="cuda:x")
        with self.assertRaises(ValueError):
            cpu(scene(), None, mask=np.ones((3, 3), dtype=bool))

    def test_greyscale_crops_only_use_value_bins(self):
        rng = np.random.default_rng(12)
        grey = rng.integers(0, 256, (100, 100), dtype=np.uint8)
        d = cpu(grey, [0, 0, 100, 100])
        for part, (a, b) in (("upper", (15, 50)), ("lower", (50, 90))):
            region = grey[a:b].ravel()
            region = region[region >= POLICY["min_value"]]
            expected = [0] * BIN_COUNT
            for v in region.tolist():
                expected[v // 64] += 1
            self.assertEqual(d["pixels"][part], len(region))
            self.assertEqual(d[part], [round(c / len(region), 5) for c in expected])

    def test_hue_bin_boundaries(self):
        # (100, 70, 52): chroma 48, hue exactly 22.5 degrees -> bin 1; one step lower -> bin 0.
        cases = {(100, 70, 52): (1 * 4 + 1) * 4 + 1, (100, 69, 52): (0 * 4 + 1) * 4 + 1,
                 (255, 0, 1): (15 * 4 + 3) * 4 + 3, (1, 0, 255): (10 * 4 + 3) * 4 + 3,
                 (0, 255, 255): (8 * 4 + 3) * 4 + 3, (255, 0, 255): (13 * 4 + 3) * 4 + 3}
        for colour, index in cases.items():
            self.assertEqual(cpu(uniform(colour), [0, 0, 100, 100])["upper"][index], 1.0, colour)

    def test_part_validity_matches_pixel_counts_on_random_inputs(self):
        rng = np.random.default_rng(21)
        image = rng.integers(0, 256, (120, 160, 3), dtype=np.uint8)
        image[rng.random((120, 160)) < 0.5] = 0
        for _ in range(60):
            x1, y1 = rng.uniform(-30, 170), rng.uniform(-30, 130)
            box = [x1, y1, x1 + rng.uniform(-5, 60), y1 + rng.uniform(-5, 90)]
            d = cpu(image, box)
            self.assertTrue(plain(d))
            json.dumps(d, allow_nan=False)
            for part in ("upper", "lower"):
                self.assertEqual(d[part] is not None, d["pixels"][part] >= POLICY["min_pixels"], box)
            self.assertEqual(d["valid"], d["upper"] is not None or d["lower"] is not None)

    def test_large_frame_is_fast_and_deterministic(self):
        import time
        rng = np.random.default_rng(4)
        image = rng.integers(0, 256, (2000, 3000, 3), dtype=np.uint8)
        polygon = [[900 + 300 * np.cos(a), 1000 + 950 * np.sin(a)] for a in np.linspace(0, 2 * np.pi, 400)]
        for mask in (None, polygon):
            started = time.perf_counter()
            first = cpu(image, [0, 0, 3000, 2000], mask=mask)
            self.assertLess(time.perf_counter() - started, 3.0)
            self.assertEqual(first, cpu(image, [0, 0, 3000, 2000], mask=mask))
            self.assertIs(first["valid"], True)


class AdversarialDistanceTests(unittest.TestCase):
    def setUp(self):
        self.red = cpu(uniform((200, 40, 40)), [0, 0, 100, 100])
        self.blue = cpu(uniform((40, 40, 200)), [0, 0, 100, 100])
        mixed = uniform((200, 40, 40))
        mixed[:, 50:] = (40, 40, 200)
        self.mixed = cpu(mixed, [0, 0, 100, 100])

    def test_extreme_scales_neither_crash_nor_change_the_result(self):
        reference = distance(self.red, self.mixed)
        self.assertTrue(0.5 < reference < 0.6)
        for factor in (1e-320, 1e-200, 1e-100, 1e100, 1e200, 1e300):
            self.assertEqual(strict(distance, scaled(self.red, factor), self.red), 0.0, factor)
            self.assertEqual(strict(distance, scaled(self.red, factor), scaled(self.red, factor)), 0.0, factor)
            self.assertAlmostEqual(strict(distance, scaled(self.red, factor), scaled(self.mixed, factor)),
                                   reference, delta=2e-6, msg=factor)
            self.assertAlmostEqual(strict(distance, scaled(self.red, factor), scaled(self.mixed, 1 / factor)
                                          if 1 / factor != float("inf") else self.mixed), reference, delta=2e-6)
            self.assertEqual(strict(distance, scaled(self.red, factor), self.blue), 1.0)

    def test_non_numeric_bins_invalidate_only_that_part(self):
        lower_only = distance({**self.red, "upper": None}, self.blue)
        for bad in (["0.5"] * BIN_COUNT, [True] * BIN_COUNT, [np.bool_(True)] * BIN_COUNT,
                    [None] + [0.0] * (BIN_COUNT - 1), [[0.1]] + [0.0] * (BIN_COUNT - 1),
                    [10 ** 400] + [0.0] * (BIN_COUNT - 1), [float("inf")] + [0.0] * (BIN_COUNT - 1),
                    [0.0] * (BIN_COUNT + 1), [0.0] * (BIN_COUNT - 1), "x" * BIN_COUNT, {}, 5):
            self.assertEqual(strict(distance, {**self.red, "upper": bad}, self.blue), lower_only)
            self.assertEqual(strict(distance, self.blue, {**self.red, "upper": bad}), lower_only)

    def test_numpy_scalar_bins_and_tuples_are_accepted(self):
        as_numpy = {**self.red, "upper": [np.float32(v) for v in self.red["upper"]],
                    "lower": tuple(np.int64(round(v * 1000)) for v in self.red["lower"])}
        self.assertEqual(distance(as_numpy, self.red), 0.0)
        self.assertEqual(distance(as_numpy, self.blue), 1.0)

    def test_structural_problems_give_none(self):
        spec_shaped = {k: self.red[k] for k in ("version", "upper", "lower", "valid")}
        self.assertEqual(distance(spec_shaped, self.red), 0.0)  # "pixels" is optional
        for broken in ({k: v for k, v in self.red.items() if k != "version"},
                       {k: v for k, v in self.red.items() if k != "valid"},
                       {**self.red, "valid": "true"}, {**self.red, "valid": np.bool_(True)},
                       {**self.red, "upper": None, "lower": None}, {**self.red, "version": None},
                       [self.red], "hsv_v1", 1.0):
            self.assertIsNone(strict(distance, broken, self.red))
            self.assertIsNone(strict(distance, self.red, broken))

    def test_json_round_trip_preserves_distance(self):
        people = [cpu(scene(shirt=tuple(c), seed=i), BOX)
                  for i, c in enumerate(np.random.default_rng(8).integers(0, 256, (4, 3)).tolist())]
        people.append(cpu(scene(), BOX, mask=[[100, 40], [160, 40], [160, 130], [100, 130]]))  # upper only
        loaded = [json.loads(json.dumps(p, allow_nan=False)) for p in people]
        for i, a in enumerate(people):
            for j, b in enumerate(people):
                self.assertEqual(distance(loaded[i], loaded[j]), distance(a, b))

    def test_triangle_inequality_and_identity(self):
        rng = np.random.default_rng(9)
        people = [cpu(scene(shirt=tuple(c), trousers=tuple(t), seed=i), BOX)
                  for i, (c, t) in enumerate(zip(rng.integers(0, 256, (7, 3)).tolist(),
                                                 rng.integers(0, 256, (7, 3)).tolist()))]
        for a in people:
            self.assertEqual(distance(a, a), 0.0)
            for b in people:
                for c in people:
                    self.assertLessEqual(distance(a, c), distance(a, b) + distance(b, c) + 2e-6)


class DeviceTests(unittest.TestCase):
    """GPU when a working CUDA device exists; every CUDA problem falls back to identical CPU output."""

    def setUp(self):
        self.saved = dict(appearance._BACKENDS)
        appearance._BACKENDS.clear()

    def tearDown(self):
        appearance._BACKENDS.clear()
        appearance._BACKENDS.update(self.saved)

    def quiet(self, function, *args, **kwargs):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            value = function(*args, **kwargs)
        self.assertEqual([str(w.message) for w in caught], [])
        return value

    def test_cpu_never_imports_torch(self):
        with mock.patch.object(appearance, "_import_torch", side_effect=AssertionError("imported")):
            self.assertEqual(resolve_device("cpu"), "cpu")
            self.assertEqual(resolve_device(None), "cpu")
            self.assertIs(descriptor(scene(), BOX, device="CPU")["valid"], True)

    def test_invalid_device_is_rejected(self):
        for device in ("tpu", "cuda:x", "mps", ""):
            with self.assertRaises(ValueError):
                resolve_device(device)

    def test_auto_without_cuda_or_torch_is_a_silent_cpu_fallback(self):
        no_cuda = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        with mock.patch.object(appearance, "_import_torch", return_value=no_cuda):
            self.assertEqual(self.quiet(resolve_device, "auto"), "cpu")
        appearance._BACKENDS.clear()
        with mock.patch.object(appearance, "_import_torch", side_effect=ImportError("No module named torch")):
            self.assertEqual(self.quiet(descriptor, scene(), BOX), cpu(scene(), BOX))

    def test_explicit_cuda_without_gpu_warns_and_uses_cpu(self):
        no_cuda = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        with mock.patch.object(appearance, "_import_torch", return_value=no_cuda):
            with self.assertWarns(RuntimeWarning):
                self.assertEqual(resolve_device("cuda"), "cpu")
            self.assertEqual(self.quiet(resolve_device, "cuda"), "cpu")  # resolved once

    def test_broken_cuda_kernel_falls_back_to_cpu(self):
        def broken_kernel(*args, **kwargs):
            raise RuntimeError("no kernel image is available")
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True), ones=broken_kernel)
        with mock.patch.object(appearance, "_import_torch", return_value=torch):
            with self.assertWarns(RuntimeWarning) as caught:
                self.assertEqual(resolve_device("auto"), "cpu")
        self.assertIn("kernel check failed", str(caught.warning))

    def test_cuda_failure_during_counting_falls_back_to_identical_cpu_result(self):
        def out_of_memory(*args, **kwargs):
            raise RuntimeError("CUDA error: out of memory")
        appearance._BACKENDS["cuda"] = ("cuda:0", SimpleNamespace(from_numpy=out_of_memory))
        image, expected = scene(), cpu(scene(), BOX)
        with mock.patch.object(appearance, "GPU_MIN_PIXELS", 0):
            with self.assertWarns(RuntimeWarning):
                self.assertEqual(descriptor(image, BOX, device="cuda"), expected)
            self.assertEqual(appearance._BACKENDS["cuda"], ("cpu", None))
            self.assertEqual(self.quiet(descriptor, image, BOX, device="cuda"), expected)

    def test_any_gpu_counting_error_falls_back_to_identical_cpu_result(self):
        image, expected = scene(), cpu(scene(), BOX)
        for error in (TypeError("where(): argument 'other' must be Tensor"), ValueError("bad"), MemoryError()):
            appearance._BACKENDS["auto"] = ("cuda:0", SimpleNamespace(from_numpy=mock.Mock(side_effect=error)))
            with mock.patch.object(appearance, "GPU_MIN_PIXELS", 0):
                with self.assertWarns(RuntimeWarning) as caught:
                    self.assertEqual(descriptor(image, BOX), expected)
            self.assertIn(type(error).__name__, str(caught.warning))
            self.assertEqual(appearance._BACKENDS["auto"], ("cpu", None))

    def test_broken_torch_or_probe_never_raises(self):
        with mock.patch.object(appearance, "_import_torch", side_effect=ValueError("numpy ABI mismatch")):
            self.assertEqual(self.quiet(resolve_device, "auto"), "cpu")
            with self.assertWarns(RuntimeWarning):
                self.assertEqual(resolve_device("cuda:1"), "cpu")
        appearance._BACKENDS.clear()
        cuda_ok = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
        with mock.patch.object(appearance, "_import_torch", return_value=cuda_ok):
            with mock.patch.dict("sys.modules", {"trailcam.vision": None}):  # vision import fails
                with self.assertWarns(RuntimeWarning):
                    self.assertEqual(resolve_device("auto"), "cpu")
            appearance._BACKENDS.clear()
            with mock.patch("trailcam.vision.select_device", side_effect=KeyError("driver")):
                with self.assertWarns(RuntimeWarning) as caught:
                    self.assertEqual(descriptor(scene(), BOX, device="gpu"), cpu(scene(), BOX))
        self.assertIn("KeyError", str(caught.warning))

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch not installed")
    def test_torch_counts_match_numpy_on_random_crops(self):
        import torch
        rng = np.random.default_rng(13)
        for trial in range(25):
            h, w = rng.integers(1, 70, 2).tolist()
            crop = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
            if trial % 3 == 0:
                crop[:] = crop[..., :1]  # greyscale
            if trial % 4 == 0:
                crop[..., rng.integers(0, 3)] = rng.choice([0, 255])  # saturated channels, ties
            keep = None if trial % 2 else rng.random((h, w)) > rng.random()
            cuts = sorted(rng.integers(0, h + 1, 3).tolist())
            parts = [(cuts[0], cuts[1]), (cuts[1], cuts[2])]
            self.assertEqual(appearance._count_torch(torch, "cpu", crop, keep, parts),
                             appearance._count_numpy(crop, keep, parts), trial)

    def test_real_cuda_matches_cpu_on_random_inputs(self):
        if not importlib.util.find_spec("torch"):
            self.skipTest("PyTorch not installed")
        import torch
        if not torch.cuda.is_available() or not resolve_device("cuda").startswith("cuda"):
            self.skipTest("CUDA not available")
        rng = np.random.default_rng(17)
        cases = []
        for trial in range(12):
            image = rng.integers(0, 256, (rng.integers(40, 300), rng.integers(40, 300), 3), dtype=np.uint8)
            if trial % 4 == 1:
                image = image[..., 0]
            elif trial % 4 == 2:
                image = image.astype(np.float32)
                image[rng.random(image.shape[:2]) < 0.1] = np.nan
            height, width = image.shape[:2]
            x1, y1 = rng.uniform(-40, width), rng.uniform(-40, height)
            box = [x1, y1, x1 + rng.uniform(10, width), y1 + rng.uniform(10, height)]
            if trial % 3 == 0:
                mask = rng.random((height, width)) > 0.4
            elif trial % 3 == 1:
                mask = rng.uniform(-20, max(width, height) + 20, (12, 2)).tolist()
            else:
                mask = None
            cases.append((image, box, mask, cpu(image, box, mask=mask)))
        with mock.patch.object(appearance, "GPU_MIN_PIXELS", 0), \
                mock.patch.object(appearance, "_count_numpy", side_effect=AssertionError("CPU path used")):
            for image, box, mask, expected in cases:
                if expected["pixels"] != {"upper": 0, "lower": 0}:
                    self.assertEqual(self.quiet(descriptor, image, box, mask=mask, device="cuda"), expected, box)
        self.assertTrue(appearance._BACKENDS["cuda"][0].startswith("cuda"))

    def test_small_crops_stay_on_numpy(self):
        appearance._BACKENDS["auto"] = ("cuda:0", SimpleNamespace(from_numpy=None))  # would fail if used
        self.assertEqual(self.quiet(descriptor, scene(), BOX), cpu(scene(), BOX))
        self.assertEqual(appearance._BACKENDS["auto"][0], "cuda:0")

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch not installed")
    def test_torch_counts_match_numpy_exactly(self):
        import torch
        rng = np.random.default_rng(11)
        crop = rng.integers(0, 256, (120, 90, 3), dtype=np.uint8)
        crop[:10] = rng.integers(0, 256, (10, 90, 1), dtype=np.uint8)  # achromatic rows
        crop[10:20, :, 0] = 255  # saturated / hue-wrap rows
        keep = rng.random((120, 90)) > 0.3
        for mask in (None, keep):
            for parts in ([(0, 50), (50, 120)], [(0, 0), (0, 120)]):
                self.assertEqual(appearance._count_torch(torch, "cpu", crop, mask, parts),
                                 appearance._count_numpy(crop, mask, parts))

    def test_real_cuda_matches_cpu_when_available(self):
        if not importlib.util.find_spec("torch"):
            self.skipTest("PyTorch not installed")
        import torch
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        device = resolve_device("cuda")
        self.assertTrue(device.startswith("cuda"))
        image = scene(size=(1200, 900), box=(200, 100, 600, 1100), seed=5, margin=0.2)
        box, polygon = (200, 100, 600, 1100), [[280, 100], [520, 100], [520, 1100], [280, 1100]]
        masked, unmasked = cpu(image, box, mask=polygon), cpu(image, box)
        with mock.patch.object(appearance, "_count_numpy", side_effect=AssertionError("CPU path used")):
            self.assertEqual(self.quiet(descriptor, image, box, mask=polygon, device="cuda"), masked)
            self.assertEqual(descriptor(image, box, device=device), unmasked)


if __name__ == "__main__":
    unittest.main()
