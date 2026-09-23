"""Capture metadata, data-strip, and camera-id invariants on synthetic inputs only."""
from __future__ import annotations

from datetime import datetime
import io
import json
import os
from pathlib import Path
import re
import tempfile
import time
import unittest
import warnings

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps, features
from PIL.TiffImagePlugin import IFDRational

from trailcam.imageinfo import (POLICY, camera_id_for, clock_suspect, detect_data_strip, in_data_strip,
                                read_capture_metadata, sequence_number)

MAKE, MODEL, DATETIME = 0x010F, 0x0110, 0x0132
ORIGINAL, DIGITIZED = 0x9003, 0x9004
SUBSEC, SUBSEC_ORIGINAL, SUBSEC_DIGITIZED = 0x9290, 0x9291, 0x9292
KEYS = ["capture_time", "time_source", "subsec", "camera_make", "camera_model",
        "sequence_number", "clock_suspect"]


def make_exif(ifd0=None, exif_ifd=None):
    exif = Image.Exif()
    for tag, value in (ifd0 or {}).items():
        exif[tag] = value
    if exif_ifd:
        sub = exif.get_ifd(0x8769)
        for tag, value in exif_ifd.items():
            sub[tag] = value
    return exif


def stamp(text):
    return datetime.fromisoformat(text).timestamp()


class FakeImage:
    """Stands in for a PIL image whose EXIF holds values Pillow would not write."""

    def __init__(self, exif=None, error=None):
        self.exif, self.error = exif, error

    def getexif(self):
        if self.error:
            raise self.error
        return self.exif


class CaptureMetadataTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def jpeg(self, name, ifd0=None, exif_ifd=None, mtime="2024-06-01T12:00:00", raw_exif=None):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        options = {}
        if raw_exif is not None:
            options["exif"] = raw_exif
        elif ifd0 or exif_ifd:
            options["exif"] = make_exif(ifd0, exif_ifd)
        Image.new("RGB", (16, 12), (90, 120, 60)).save(path, "JPEG", quality=90, **options)
        if mtime:
            os.utime(path, (stamp(mtime), stamp(mtime)))
        return path

    def read(self, path):
        with Image.open(path) as image:
            return read_capture_metadata(image, path)

    def test_original_time_with_subseconds_make_model_and_sequence(self):
        path = self.jpeg("IMAG0571.JPG", {MAKE: "Browning", MODEL: "BTC-8E", DATETIME: "2024:05:02 00:00:00"},
                         {ORIGINAL: "2024:05:01 10:20:30", SUBSEC_ORIGINAL: "42",
                          DIGITIZED: "2024:05:01 11:00:00"})
        result = self.read(path)
        self.assertEqual(list(result), KEYS)
        self.assertEqual(result, {"capture_time": "2024-05-01T10:20:30.420000",
                                  "time_source": "exif_original", "subsec": "42",
                                  "camera_make": "Browning", "camera_model": "BTC-8E",
                                  "sequence_number": 571, "clock_suspect": False})
        self.assertEqual(json.loads(json.dumps(result)), result)

    def test_exif_survives_transpose_and_rgb_conversion(self):
        path = self.jpeg("IMG_0042.jpg", {MAKE: "Bushnell"}, {ORIGINAL: "2023:07:08 09:10:11"})
        with Image.open(path) as source:
            converted = ImageOps.exif_transpose(source).convert("RGB")
        result = read_capture_metadata(converted, path)
        self.assertEqual((result["capture_time"], result["time_source"]), ("2023-07-08T09:10:11", "exif_original"))
        self.assertEqual(result["sequence_number"], 42)
        self.assertIsNone(result["subsec"])

    def test_digitized_then_datetime_fallbacks(self):
        digitized = self.read(self.jpeg("a1.jpg", {DATETIME: "2022:01:05 08:00:00"},
                                        {DIGITIZED: "2022:01:05 07:59:58", SUBSEC_DIGITIZED: "5"}))
        self.assertEqual((digitized["capture_time"], digitized["time_source"], digitized["subsec"]),
                         ("2022-01-05T07:59:58.500000", "exif_digitized", "5"))
        plain = self.read(self.jpeg("a2.jpg", {DATETIME: "2022:01:05 08:00:00"}, {SUBSEC: "123"}))
        self.assertEqual((plain["capture_time"], plain["time_source"], plain["subsec"]),
                         ("2022-01-05T08:00:00.123000", "exif_datetime", "123"))

    def test_subsecond_tag_must_match_the_chosen_time(self):
        result = self.read(self.jpeg("b.jpg", exif_ifd={ORIGINAL: "2022:03:04 05:06:07", SUBSEC_DIGITIZED: "9"}))
        self.assertEqual((result["capture_time"], result["subsec"]), ("2022-03-04T05:06:07", None))

    def test_no_exif_uses_local_file_mtime(self):
        result = self.read(self.jpeg("cam/DSCF0003.JPG", mtime="2021-09-10T13:14:15"))
        self.assertEqual(result, {"capture_time": "2021-09-10T13:14:15", "time_source": "file_mtime",
                                  "subsec": None, "camera_make": None, "camera_model": None,
                                  "sequence_number": 3, "clock_suspect": False})

    def test_malformed_times_fall_through_to_the_next_source(self):
        bad = ("0000:00:00 00:00:00", "    :  :     :  :  ", "2024:13:45 10:00:00", "garbage", "")
        for index, value in enumerate(bad):
            with self.subTest(value=value):
                result = self.read(self.jpeg(f"m{index}.jpg", {DATETIME: "2020:02:03 04:05:06"},
                                             {ORIGINAL: value, DIGITIZED: value}))
                self.assertEqual((result["capture_time"], result["time_source"]),
                                 ("2020-02-03T04:05:06", "exif_datetime"))
        result = self.read(self.jpeg("m9.jpg", {DATETIME: "2020:02:30 04:05:06"}, mtime="2024-06-01T12:00:00"))
        self.assertEqual((result["capture_time"], result["time_source"]), ("2024-06-01T12:00:00", "file_mtime"))

    def test_corrupt_exif_block_never_raises(self):
        path = self.jpeg("IMG_0100.jpg", raw_exif=b"Exif\x00\x00II*\x00\x08\x00\x00\x00\xff\xff\x01")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Pillow warns about the corrupt block at open time
            result = self.read(path)
        self.assertEqual((result["time_source"], result["sequence_number"]), ("file_mtime", 100))
        for action in ("ignore", "error"):  # lazily parsed EXIF: even -W error must not escape
            with self.subTest(warnings=action), warnings.catch_warnings():
                warnings.simplefilter(action)
                memory = Image.new("RGB", (4, 4))
                memory.info["exif"] = b"Exif\x00\x00II*\x00\x08\x00\x00\x00\xff\xff\x01"
                self.assertEqual(read_capture_metadata(memory, path)["time_source"], "file_mtime")
        broken = FakeImage(error=SyntaxError("not a TIFF file"))
        self.assertEqual(read_capture_metadata(broken, path)["time_source"], "file_mtime")
        self.assertEqual(read_capture_metadata(object(), path)["time_source"], "file_mtime")

    def test_odd_value_types_and_padding_are_cleaned(self):
        exif = make_exif({MAKE: b"Reconyx\x00\x00  ", MODEL: "   ", DATETIME: (1, 2)},
                         {ORIGINAL: b"2019:04:05 06:07:08\x00", SUBSEC_ORIGINAL: " 07 "})
        result = read_capture_metadata(FakeImage(exif), None)
        self.assertEqual(result["camera_make"], "Reconyx")
        self.assertIsNone(result["camera_model"])
        self.assertEqual((result["capture_time"], result["subsec"]), ("2019-04-05T06:07:08.070000", "07"))
        exif = make_exif({MAKE: 12}, {ORIGINAL: "2019:04:05 06:07:08", SUBSEC_ORIGINAL: "4a"})
        result = read_capture_metadata(FakeImage(exif), None)
        self.assertIsNone(result["camera_make"])
        self.assertEqual((result["capture_time"], result["subsec"]), ("2019-04-05T06:07:08", None))

    def test_long_subseconds_truncate_to_microseconds_and_inline_fraction_is_used(self):
        result = read_capture_metadata(FakeImage(make_exif(exif_ifd={
            ORIGINAL: "2019:04:05 06:07:08", SUBSEC_ORIGINAL: "123456789"})), None)
        self.assertEqual((result["capture_time"], result["subsec"]), ("2019-04-05T06:07:08.123456", "123456789"))
        result = read_capture_metadata(FakeImage(make_exif(exif_ifd={ORIGINAL: "2019-04-05T06:07:08.25+02:00"})), None)
        self.assertEqual((result["capture_time"], result["subsec"]), ("2019-04-05T06:07:08.250000", "25"))
        zero = read_capture_metadata(FakeImage(make_exif(exif_ifd={
            ORIGINAL: "2019:04:05 06:07:08", SUBSEC_ORIGINAL: "000"})), None)
        self.assertEqual((zero["capture_time"], zero["subsec"]), ("2019-04-05T06:07:08", "000"))

    def test_original_time_written_in_ifd0_is_still_used(self):
        result = read_capture_metadata(FakeImage(make_exif({ORIGINAL: "2018:08:09 10:11:12"})), None)
        self.assertEqual((result["capture_time"], result["time_source"]), ("2018-08-09T10:11:12", "exif_original"))

    def test_nothing_available(self):
        expected = {"capture_time": None, "time_source": None, "subsec": None, "camera_make": None,
                    "camera_model": None, "sequence_number": None, "clock_suspect": False}
        self.assertEqual(read_capture_metadata(None, None), expected)
        missing = read_capture_metadata(None, self.root / "gone" / "IMG_0009.JPG")
        self.assertEqual((missing["capture_time"], missing["time_source"], missing["sequence_number"]),
                         (None, None, 9))
        self.assertEqual(read_capture_metadata(FakeImage(Image.Exif()), None), expected)

    def test_clock_suspect_rules(self):
        cases = [("2014:12:31 23:59:59", True), ("2015:01:02 00:00:00", False),
                 ("2024:01:01 00:00:03", True), ("2024:01:01 00:09:59", True),
                 ("2024:01:01 00:10:00", False), ("2024:01:02 00:00:00", False),
                 ("2024:06:02 12:00:00", False), ("2024:06:02 12:00:01", True)]
        for index, (value, suspect) in enumerate(cases):
            with self.subTest(value=value):
                path = self.jpeg(f"c{index}.jpg", exif_ifd={ORIGINAL: value}, mtime="2024-06-01T12:00:00")
                self.assertIs(self.read(path)["clock_suspect"], suspect)
        old_file = self.jpeg("c_old.jpg", mtime="2012-05-05T12:00:00")
        self.assertIs(self.read(old_file)["clock_suspect"], True)
        no_file = read_capture_metadata(FakeImage(make_exif(exif_ifd={ORIGINAL: "2030:01:01 12:00:00"})), None)
        self.assertIs(no_file["clock_suspect"], False)

    def test_sequence_number_is_last_digit_run_of_the_stem(self):
        cases = {"IMAG0571.JPG": 571, "IMG_0042.jpg": 42, "cam3\\sub\\IMG_0007.JPG": 7,
                 "cam3/2024-05-01_0015.jpg": 15, "DSCF0001 (2).JPG": 2, "trail.v2/IMG_0010": 10,
                 "PICT.JPG": None, ".jpg": None, "": None, "0000.jpg": 0}
        for name, expected in cases.items():
            with self.subTest(name=name):
                self.assertEqual(sequence_number(name), expected)
        self.assertEqual(sequence_number(Path("x") / "RCNX0123.JPG"), 123)
        self.assertIsNone(sequence_number(None))


def smooth_field(rng, height, width, channels, low, high, cell=24):
    coarse = rng.uniform(low, high, size=(height // cell + 2, width // cell + 2, channels)).astype(np.uint8)
    field = Image.fromarray(coarse if channels == 3 else coarse[..., 0]).resize((width, height), Image.BILINEAR)
    field = np.asarray(field, dtype=np.float64)
    return field if channels == 3 else field[..., None]


def natural_scene(height=600, width=800, seed=1, grey=False, low=60, high=200, noise=14.0):
    """Textured daylight (or night-IR grey) scene with dark tree trunks reaching the bottom."""
    rng = np.random.default_rng(seed)
    scene = smooth_field(rng, height, width, 1 if grey else 3, low, high)
    scene = scene + rng.normal(0, noise, size=scene.shape)
    for x in rng.integers(0, width - 40, size=3):  # dark trunks cover ~10% of columns
        scene[:, x:x + 25] *= 0.15
    scene = np.clip(scene, 0, 255).astype(np.uint8)
    return np.repeat(scene, 3, axis=2) if grey else scene


def add_strip(image, rows, top=False, seed=2, coverage=0.35):
    """Near-black camera bar with white text-like glyph bars in its middle rows."""
    rng = np.random.default_rng(seed)
    height, width = image.shape[:2]
    band = slice(0, rows) if top else slice(height - rows, height)
    image[band] = rng.integers(0, 14, size=image[band].shape, dtype=np.uint8)
    start = (0 if top else height - rows) + max(1, rows // 4)
    stop = (rows if top else height) - max(1, rows // 4)
    x = 10
    while x < width - 10:
        glyph = int(rng.integers(2, 7))
        if rng.random() < coverage * 1.4:
            image[start:stop, x:x + glyph] = 245
        x += glyph + int(rng.integers(1, 4))
    return image


def soil_scene(height=600, width=800, seed=3, start=2 / 3, darkest=0.12):
    """Scene whose ground darkens smoothly toward the bottom: no sharp edge."""
    image = natural_scene(height, width, seed, low=100, high=160, noise=0).astype(np.float64)
    ramp = np.ones(height)
    first = int(start * height)
    ramp[first:] = np.linspace(1.0, darkest, height - first)
    rng = np.random.default_rng(seed + 10)
    image = image * ramp[:, None, None] + rng.normal(0, 8, size=image.shape)
    pebbles = rng.random((height, width)) < 0.03
    image[pebbles] += 45
    return np.clip(image, 0, 255).astype(np.uint8)


class DataStripTests(unittest.TestCase):
    def assertStrip(self, image, top, bottom, tolerance=0):
        result = detect_data_strip(image)
        self.assertEqual(set(result), {"top", "bottom"})
        self.assertIs(type(result["top"]), int)
        self.assertIs(type(result["bottom"]), int)
        self.assertLessEqual(abs(result["top"] - top), tolerance, result)
        self.assertLessEqual(abs(result["bottom"] - bottom), tolerance, result)
        return result

    def test_daylight_bottom_bar_with_text_is_detected_exactly(self):
        self.assertStrip(add_strip(natural_scene(), 30), 0, 30)

    def test_top_bar_and_both_bars(self):
        self.assertStrip(add_strip(natural_scene(seed=4), 24, top=True), 24, 0)
        both = add_strip(add_strip(natural_scene(seed=5), 40), 20, top=True, seed=6)
        self.assertStrip(both, 20, 40)

    def test_night_ir_grey_scene(self):
        image = add_strip(natural_scene(seed=7, grey=True, low=45, high=150, noise=10), 36)
        self.assertTrue(np.array_equal(image[:-36, :, 0], image[:-36, :, 2]))  # grey scene
        self.assertStrip(image, 0, 36)

    def test_dense_text_rows_stay_inside_the_band(self):
        image = add_strip(natural_scene(seed=8), 40, coverage=0.55)
        text_rows = (image[-30:-10].mean(axis=2) < 30).mean(axis=1)
        self.assertLess(text_rows.min(), POLICY["strip_row_dark_share"])  # lighter than a band row
        self.assertStrip(image, 0, 40)

    def test_boundary_is_refined_at_full_resolution(self):
        for rows in (75, 76, 77):
            with self.subTest(rows=rows):  # 1500 rows -> working step 2
                self.assertStrip(add_strip(natural_scene(1500, 2000, seed=rows), rows), 0, rows)
                self.assertStrip(add_strip(natural_scene(1500, 2000, seed=rows), rows, top=True), rows, 0)

    def test_jpeg_round_trip_keeps_the_boundary(self):
        buffer = io.BytesIO()
        Image.fromarray(add_strip(natural_scene(seed=9), 30)).save(buffer, "JPEG", quality=85)
        decoded = np.asarray(Image.open(io.BytesIO(buffer.getvalue())).convert("RGB"))
        self.assertStrip(decoded, 0, 30, tolerance=2)

    def test_partly_shadowed_scene_above_the_bar(self):
        image = add_strip(natural_scene(seed=24), 30)
        image[-90:-30, :240] = 12  # black shadow over 30% of the width touches the bar
        self.assertStrip(image, 0, 30)

    def test_dark_band_away_from_the_edges_is_ignored(self):
        image = natural_scene(seed=25)
        image[300:330] = 5
        self.assertStrip(image, 0, 0)

    def test_scene_without_bar(self):
        self.assertStrip(natural_scene(seed=10), 0, 0)
        self.assertStrip(natural_scene(seed=11, grey=True, low=45, high=150), 0, 0)

    def test_gradually_darkening_soil_is_not_a_strip(self):
        image = soil_scene()
        self.assertGreater((image[-20:].mean(axis=2) < 30).mean(), POLICY["strip_row_dark_share"])
        self.assertStrip(image, 0, 0)

    def test_soft_edged_dark_ground_within_the_height_limit_is_not_a_strip(self):
        image = soil_scene(seed=12, start=0.88, darkest=0.03)
        self.assertGreater((image[-10:].mean(axis=2) < 30).mean(), POLICY["strip_row_dark_share"])
        self.assertStrip(image, 0, 0)

    def test_dark_ground_taller_than_the_limit_is_not_a_strip(self):
        image = natural_scene(seed=13)
        image[400:] = np.random.default_rng(13).integers(0, 22, size=image[400:].shape, dtype=np.uint8)
        self.assertStrip(image, 0, 0)

    def test_sharp_but_irregular_dark_outline_is_not_a_strip(self):
        image = natural_scene(seed=14)
        columns = np.arange(image.shape[1])
        outline = (560 + 30 * np.sin(columns / 23.0)).astype(int)
        rows = np.arange(image.shape[0])[:, None]
        image[rows >= outline[None, :]] = 8
        self.assertStrip(image, 0, 0)

    def test_gradual_night_sky_at_the_top_is_not_a_strip(self):
        image = natural_scene(seed=15, grey=True, low=45, high=150).astype(np.float64)
        image[:150] *= np.linspace(0.05, 1.0, 150)[:, None, None]
        self.assertStrip(np.clip(image, 0, 255).astype(np.uint8), 0, 0)

    def test_height_limits(self):
        limit = int(POLICY["strip_max_fraction"] * 600)
        self.assertStrip(add_strip(natural_scene(seed=16), limit), 0, limit)
        self.assertStrip(add_strip(natural_scene(seed=16), limit + 1), 0, 0)
        self.assertStrip(add_strip(natural_scene(seed=17), 90), 0, 0)  # 15%
        self.assertStrip(add_strip(natural_scene(seed=18), 3), 0, 0)   # 0.5%
        self.assertStrip(add_strip(natural_scene(seed=18), 6), 0, 6)   # 1%

    def test_band_must_be_very_dark(self):
        image = natural_scene(seed=19)
        image[-30:] = 60
        self.assertStrip(image, 0, 0)
        image[-30:] = 29
        self.assertStrip(image, 0, 30)

    def test_uniform_images(self):
        for value in (0, 20, 128, 255):
            with self.subTest(value=value):
                self.assertStrip(np.full((300, 400, 3), value, np.uint8), 0, 0)

    def test_input_shapes_and_types(self):
        image = add_strip(natural_scene(seed=20), 30)
        self.assertStrip(image.mean(axis=2).astype(np.uint8), 0, 30)
        self.assertStrip(np.dstack([image, np.full(image.shape[:2], 255, np.uint8)]), 0, 30)
        self.assertStrip(image.astype(np.float32), 0, 30)
        self.assertStrip(image[..., ::-1], 0, 30)  # BGR
        self.assertStrip(Image.fromarray(image), 0, 30)
        self.assertStrip(image[:, :, :1], 0, 30)

    def test_empty_and_invalid_inputs(self):
        for value in (None, [], np.zeros((0, 0, 3), np.uint8), np.zeros((5, 5, 3), np.uint8),
                      np.zeros((600, 20, 3), np.uint8), np.zeros(10, np.uint8), np.zeros((4, 4, 4, 4)),
                      np.zeros((100, 100, 0), np.uint8), "not an image", [[1, 2], [3]], 7):
            with self.subTest(value=type(value).__name__):
                self.assertEqual(detect_data_strip(value), {"top": 0, "bottom": 0})

    def test_nonfinite_pixels_do_not_raise(self):
        image = add_strip(natural_scene(seed=21), 30).astype(np.float64)
        image[100:110] = np.nan
        image[-2:, :50] = np.inf
        self.assertStrip(image, 0, 30)

    def test_deterministic(self):
        image = add_strip(natural_scene(seed=22), 33)
        self.assertEqual(detect_data_strip(image), detect_data_strip(image.copy()))

    def test_full_resolution_frame_is_fast(self):
        tile = natural_scene(300, 400, seed=23)
        image = np.ascontiguousarray(np.repeat(np.repeat(tile, 10, axis=0), 10, axis=1))
        add_strip(image, 151)
        timings = []
        for _ in range(5):
            started = time.perf_counter()
            result = detect_data_strip(image)
            timings.append(time.perf_counter() - started)
        self.assertEqual(result, {"top": 0, "bottom": 151})
        self.assertLess(min(timings), 0.030)


class InDataStripTests(unittest.TestCase):
    STRIP = {"top": 20, "bottom": 30}

    def test_area_fraction_rule(self):
        self.assertTrue(in_data_strip([10, 280, 50, 300], self.STRIP, 300))
        self.assertTrue(in_data_strip([10, 250, 50, 290], self.STRIP, 300))   # exactly half
        self.assertFalse(in_data_strip([10, 249, 50, 290], self.STRIP, 300))  # just under half
        self.assertTrue(in_data_strip([10, 0, 50, 30], self.STRIP, 300))      # top strip, 2/3
        self.assertFalse(in_data_strip([10, 100, 50, 200], self.STRIP, 300))
        self.assertTrue(in_data_strip([0, 0, 1, 300], {"top": 75, "bottom": 75}, 300))  # both strips sum

    def test_fraction_parameter(self):
        box = [10, 240, 50, 300]  # half inside
        self.assertFalse(in_data_strip(box, self.STRIP, 300, fraction=0.51))
        self.assertTrue(in_data_strip(box, self.STRIP, 300, fraction=0.5))
        self.assertTrue(in_data_strip([10, 200, 50, 271], self.STRIP, 300, fraction=0))
        self.assertFalse(in_data_strip([10, 200, 50, 270], self.STRIP, 300, fraction=0))
        self.assertTrue(in_data_strip([10, 270, 50, 300], self.STRIP, 300, fraction=1.0))
        self.assertFalse(in_data_strip([10, 269, 50, 300], self.STRIP, 300, fraction=1.0))

    def test_box_is_clipped_to_image_rows(self):
        self.assertTrue(in_data_strip([10, 260, 50, 400], self.STRIP, 300))
        self.assertFalse(in_data_strip([10, 300, 50, 400], self.STRIP, 300))
        self.assertTrue(in_data_strip([10, -50, 50, 25], self.STRIP, 300))

    def test_missing_strip_or_invalid_inputs(self):
        box = [10, 280, 50, 300]
        for strip in ({"top": 0, "bottom": 0}, {}, None, {"top": None, "bottom": None}, [0, 30],
                      {"top": -5, "bottom": -5}, {"top": "x", "bottom": 30}):
            with self.subTest(strip=strip):
                self.assertFalse(in_data_strip(box, strip, 300))
        for bad in (None, [], [1, 2, 3], [10, 290, 10, 300], [10, 300, 50, 290], [10, 295, 50, 295],
                    [float("nan"), 280, 50, 300], ["a", 280, 50, 300]):
            with self.subTest(box=bad):
                self.assertFalse(in_data_strip(bad, self.STRIP, 300))
        for height in (None, 0, -300, float("inf")):
            with self.subTest(height=height):
                self.assertFalse(in_data_strip(box, self.STRIP, height))

    def test_oversized_strip_is_clamped_without_double_counting(self):
        self.assertTrue(in_data_strip([0, 0, 10, 100], {"top": 80, "bottom": 80}, 100, fraction=1.0))
        self.assertTrue(in_data_strip([0, 0, 10, 100], {"top": 500, "bottom": 0}, 100, fraction=1.0))
        self.assertIs(type(in_data_strip([0, 90, 10, 100], {"top": 0, "bottom": 10.0}, 100)), bool)


class CameraIdTests(unittest.TestCase):
    def test_folder_mode(self):
        cases = {"cam1/IMG_0001.JPG": "cam1", "site/cam2/x.jpg": "site/cam2", "x.jpg": ".",
                 "site\\cam2\\x.jpg": "site/cam2", "./cam1/x.jpg": "cam1", "cam1//x.jpg": "cam1",
                 "": ".", None: "."}
        for path, expected in cases.items():
            with self.subTest(path=path):
                self.assertEqual(camera_id_for(path), expected)
        self.assertEqual(camera_id_for(Path("a") / "b" / "c.jpg", "FOLDER"), "a/b")

    def test_regex_mode(self):
        pattern = r"(?P<camera>CAM\d+)"
        self.assertEqual(camera_id_for("trail/CAM07_2024/IMG1.jpg", "regex", pattern), "CAM07")
        self.assertEqual(camera_id_for("trail\\CAM07\\IMG1.jpg", "regex", r"^trail/(?P<camera>[^/]+)/"), "CAM07")
        self.assertEqual(camera_id_for("trail/north/IMG1.jpg", "regex", pattern), "trail/north")
        self.assertEqual(camera_id_for("IMG1.jpg", "regex", pattern), ".")
        self.assertEqual(camera_id_for("a/x.jpg", "regex", re.compile(r"(?P<site>a)?(?P<camera>z)?")), "a")
        self.assertEqual(camera_id_for("s/CAM3_x.jpg", "regex", re.compile(pattern)), "CAM3")

    def test_regex_configuration_errors(self):
        for pattern in (None, r"(CAM\d+)", r"(?P<camera>"):
            with self.subTest(pattern=pattern):
                with self.assertRaises(ValueError):
                    camera_id_for("CAM1/x.jpg", "regex", pattern)

    def test_single_and_unknown_modes(self):
        self.assertEqual(camera_id_for("cam1/x.jpg", "single"), "all")
        self.assertEqual(camera_id_for(None, "single", r"ignored("), "all")
        with self.assertRaises(ValueError):
            camera_id_for("cam1/x.jpg", "exif")


# ---------------------------------------------------------------------------
# Adversarial review: malformed input, exact thresholds, layouts, JSON types.
# ---------------------------------------------------------------------------
NONE = {"top": 0, "bottom": 0}
SOURCES = {None, "exif_original", "exif_digitized", "exif_datetime", "file_mtime"}


def noise_scene(height, width, seed=0):
    """Mid-grey random texture: no very dark and no text-bright pixels."""
    return np.random.default_rng(seed).integers(70, 200, size=(height, width, 3), dtype=np.uint8)


class AdversarialMetadataTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.file = self.root / "IMG_0001.JPG"
        Image.new("RGB", (8, 8)).save(self.file, "JPEG")
        os.utime(self.file, (stamp("2024-06-01T12:00:00"), stamp("2024-06-01T12:00:00")))

    def tearDown(self):
        self._tmp.cleanup()

    def assertContract(self, result):
        self.assertEqual(list(result), KEYS)
        self.assertEqual(json.loads(json.dumps(result)), result)
        self.assertIn(result["time_source"], SOURCES)
        for key in ("capture_time", "subsec", "camera_make", "camera_model"):
            self.assertIn(type(result[key]), (str, type(None)), key)
        self.assertIn(type(result["sequence_number"]), (int, type(None)))
        self.assertIs(type(result["clock_suspect"]), bool)
        if result["capture_time"] is not None:
            self.assertRegex(result["capture_time"], r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(\.\d{6})?$")

    def test_extreme_and_impossible_calendar_values(self):
        cases = [("0001:01:01 00:00:00", "0001-01-01T00:00:00", "exif_original", True),
                 ("9999:12:31 23:59:59", "9999-12-31T23:59:59", "exif_original", True),  # far after mtime
                 ("2024:02:29 12:00:00", "2024-02-29T12:00:00", "exif_original", False),  # leap day
                 ("2023:02:29 12:00:00", "2020-02-03T04:05:06", "exif_datetime", False),
                 ("2024:05:01 24:00:00", "2020-02-03T04:05:06", "exif_datetime", False),
                 ("2024:05:01 23:59:60", "2020-02-03T04:05:06", "exif_datetime", False),
                 ("12024:05:01 10:00:00", "2020-02-03T04:05:06", "exif_datetime", False),
                 ("x2024:05:01 10:00:00", "2020-02-03T04:05:06", "exif_datetime", False)]
        for value, expected, source, suspect in cases:
            with self.subTest(value=value):
                image = FakeImage(make_exif({DATETIME: "2020:02:03 04:05:06"}, {ORIGINAL: value}))
                result = read_capture_metadata(image, self.file)
                self.assertContract(result)
                self.assertEqual((result["capture_time"], result["time_source"], result["clock_suspect"]),
                                 (expected, source, suspect))

    def test_non_text_tag_values_fall_through(self):
        for value in (IFDRational(1, 2), (b"2019:04:05 06:07:08",), ["2019:04:05 06:07:08"], 20190405, 1.5, None):
            with self.subTest(value=repr(value)):
                image = FakeImage(make_exif({DATETIME: "2020:02:03 04:05:06", MAKE: value, MODEL: value},
                                            {ORIGINAL: value, DIGITIZED: value, SUBSEC: value}))
                result = read_capture_metadata(image, None)
                self.assertContract(result)
                self.assertEqual((result["capture_time"], result["time_source"], result["subsec"]),
                                 ("2020-02-03T04:05:06", "exif_datetime", None))
                self.assertIsNone(result["camera_make"])

    def test_subsecond_extremes(self):
        for subsec, expected in (("9999999", "2019-04-05T06:07:08.999999"), ("000001", "2019-04-05T06:07:08.000001"),
                                 ("5", "2019-04-05T06:07:08.500000"), ("-5", "2019-04-05T06:07:08"),
                                 ("1e3", "2019-04-05T06:07:08"), ("\u0661\u0662", "2019-04-05T06:07:08")):
            with self.subTest(subsec=subsec):
                result = read_capture_metadata(FakeImage(make_exif(exif_ifd={
                    ORIGINAL: "2019:04:05 06:07:08", SUBSEC_ORIGINAL: subsec})), None)
                self.assertContract(result)
                self.assertEqual(result["capture_time"], expected)

    def test_exif_containers_that_fail_part_way(self):
        class HalfBroken(dict):
            def get_ifd(self, tag):
                raise ValueError("bad Exif IFD offset")
        result = read_capture_metadata(FakeImage(HalfBroken({DATETIME: "2021:01:02 03:04:05", MAKE: "Acme"})), None)
        self.assertEqual((result["capture_time"], result["time_source"], result["camera_make"]),
                         ("2021-01-02T03:04:05", "exif_datetime", "Acme"))
        plain = read_capture_metadata(FakeImage({MAKE: "Acme"}), None)  # a dict with no get_ifd at all
        self.assertEqual(plain["camera_make"], "Acme")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            warned = read_capture_metadata(FakeImage(error=UserWarning("Corrupt EXIF data")), self.file)
        self.assertEqual(warned["time_source"], "file_mtime")
        self.assertContract(warned)

    def test_odd_paths_never_raise_and_empty_path_is_not_the_cwd(self):
        for path in ("", b"", "a\x00b.jpg", self.root, 12, b"cam/IMG_0077.JPG", self.root / "missing" / "x.jpg"):
            with self.subTest(path=repr(path)):
                self.assertContract(read_capture_metadata(None, path))
        self.assertIsNone(read_capture_metadata(None, "")["time_source"])
        self.assertIsNone(read_capture_metadata(None, b"")["time_source"])
        self.assertEqual(read_capture_metadata(None, os.fsencode(self.file))["capture_time"], "2024-06-01T12:00:00")
        self.assertEqual(read_capture_metadata(None, b"cam/IMG_0077.JPG")["sequence_number"], 77)

    def test_sequence_number_edge_cases(self):
        self.assertIsNone(sequence_number("IMG_" + "9" * 5000 + ".jpg"))  # beyond int() digit limit
        self.assertEqual(sequence_number("IMG_" + "1" * 30 + ".jpg"), int("1" * 30))
        self.assertEqual(sequence_number("\u05de\u05e6\u05dc\u05de\u05d4_0042.jpg"), 42)  # Hebrew stem
        self.assertIsNone(sequence_number("IMG_\u0661\u0662.jpg"))  # non-ASCII digits are not a counter
        self.assertEqual(sequence_number("IMG_0010.v2.jpg"), 2)  # last run of the stem, as specified
        self.assertEqual(sequence_number("cam7/"), 7)

    def test_clock_suspect_direct_boundaries(self):
        mtime = datetime(2024, 6, 1, 12)
        self.assertIs(clock_suspect(None, mtime), False)
        self.assertIs(clock_suspect(datetime(2024, 6, 2, 12), mtime), False)
        self.assertIs(clock_suspect(datetime(2024, 6, 2, 12, 0, 0, 1), mtime), True)
        self.assertIs(clock_suspect(datetime(2015, 1, 1, 0, 10), None), False)
        self.assertIs(clock_suspect(datetime(2015, 1, 1, 0, 9, 59, 999999), None), True)
        self.assertIs(clock_suspect(datetime(2015, 1, 1, 0, 0), None), True)
        self.assertIs(clock_suspect(datetime(2014, 12, 31, 12), datetime(2024, 1, 1)), True)
        self.assertIs(clock_suspect(datetime(2016, 1, 1, 12), None), False)
        self.assertIs(clock_suspect(datetime(9999, 12, 31), datetime(1, 1, 1)), True)  # no overflow

    def test_repeated_reads_are_identical(self):
        exif = make_exif({MAKE: "Browning", DATETIME: "2022:01:05 08:00:00"},
                         {ORIGINAL: "2024:05:01 10:20:30", SUBSEC_ORIGINAL: "42"})
        first = read_capture_metadata(FakeImage(exif), self.file)
        for _ in range(3):
            self.assertEqual(read_capture_metadata(FakeImage(exif), self.file), first)
        self.assertContract(first)


class AdversarialDataStripTests(unittest.TestCase):
    def test_thresholds_are_exact_for_rgb_and_single_channel_input(self):
        image = natural_scene(600, 512, seed=40)  # 512 wide: every column is sampled
        image[-30:] = 0
        image[-24:-6, ::2] = POLICY["strip_text_level"]  # half of each text row at exactly the text level
        for view in (image, image[..., 0], image[..., :1], image[..., ::-1], np.dstack([image, image[..., :1]])):
            with self.subTest(shape=view.shape):
                self.assertEqual(detect_data_strip(view), {"top": 0, "bottom": 30})
        image[-24:-6, ::2] = POLICY["strip_text_level"] - 1  # not text: identical verdict on every layout
        self.assertEqual(detect_data_strip(image), detect_data_strip(image[..., 0]))
        self.assertNotEqual(detect_data_strip(image)["bottom"], 30)
        for level, expected in ((POLICY["strip_dark_level"], 0), (POLICY["strip_dark_level"] - 1, 30)):
            band = natural_scene(600, 512, seed=41)
            band[-30:] = level
            with self.subTest(level=level):
                self.assertEqual(detect_data_strip(band)["bottom"], expected)
                self.assertEqual(detect_data_strip(band[..., 0])["bottom"], expected)

    def test_maximum_height_is_exact_when_rows_are_strided(self):
        for height in (1499, 2999):  # working step 2 and 3; 12% is not a multiple of the step
            limit = int(POLICY["strip_max_fraction"] * height)
            base = natural_scene(height, 512, seed=height)
            for rows, expected in ((limit - 2, limit - 2), (limit - 1, limit - 1), (limit, limit), (limit + 1, 0)):
                for top in (False, True):
                    with self.subTest(height=height, rows=rows, top=top):
                        result = detect_data_strip(add_strip(base.copy(), rows, top=top))
                        self.assertEqual((result["top"], result["bottom"]), (expected, 0) if top else (0, expected))

    def test_minimum_side_and_extreme_aspect_ratios(self):
        image = noise_scene(32, 32)
        image[-3:] = 0  # int(12% of 32) = 3 rows
        self.assertEqual(detect_data_strip(image), {"top": 0, "bottom": 3})
        self.assertEqual(detect_data_strip(image[1:]), NONE)  # 31 rows
        self.assertEqual(detect_data_strip(image[:, 1:]), NONE)  # 31 columns
        wide = noise_scene(32, 20000, seed=1)
        wide[:3], wide[-3:] = 5, 0
        self.assertEqual(detect_data_strip(wide), {"top": 3, "bottom": 3})
        tall = noise_scene(20000, 40, seed=2)
        tall[:2400], tall[-1200:] = 3, 0  # exactly 12% at the top, 6% at the bottom (working step 20)
        self.assertEqual(detect_data_strip(tall), {"top": 2400, "bottom": 1200})
        tall[2400] = 3
        self.assertEqual(detect_data_strip(tall), {"top": 0, "bottom": 1200})

    def test_rendered_antialiased_text_bars(self):
        if not features.check("freetype2"):
            self.skipTest("FreeType is not available for anti-aliased text")
        for height, width, rows, size, top in ((1080, 1920, 54, 36, False), (1080, 1920, 54, 36, True),
                                               (600, 800, 24, 16, False)):
            scene = Image.fromarray(natural_scene(300, 400, seed=45)).resize((width, height), Image.BILINEAR)
            draw = ImageDraw.Draw(scene)
            y0 = 0 if top else height - rows
            draw.rectangle([0, y0, width - 1, y0 + rows - 1], fill=(0, 0, 0))
            draw.text((20, y0 + (rows - size) // 2), "TRAIL  24C 75F  05/01/2024 10:20:30  CAM07",
                      font=ImageFont.load_default(size=size), fill=(255, 255, 255))
            expected = {"top": rows, "bottom": 0} if top else {"top": 0, "bottom": rows}
            for quality in (None, 85, 60):
                with self.subTest(height=height, top=top, quality=quality):
                    if quality is None:
                        array = np.asarray(scene)
                    else:
                        buffer = io.BytesIO()
                        scene.save(buffer, "JPEG", quality=quality)
                        array = np.asarray(Image.open(io.BytesIO(buffer.getvalue())).convert("RGB"))
                    self.assertEqual(detect_data_strip(array), expected)

    def test_warnings_as_errors_never_escape(self):
        image = add_strip(natural_scene(seed=46), 30)
        edge = image.astype(np.float64)
        edge[-40:-31] = np.nan
        edge[-31:-28, :100] = np.inf
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self.assertEqual(detect_data_strip(image.astype(np.complex128)), NONE)
            self.assertEqual(detect_data_strip(image.astype(np.float64) * 1e305), NONE)  # luma overflows to inf
            for value in (edge, np.full((64, 64, 3), np.nan), np.full((64, 64, 3), -np.inf),
                          np.full((64, 64, 3), np.finfo(np.float64).max)):
                result = detect_data_strip(value)
                self.assertEqual(set(result), {"top", "bottom"})
                self.assertTrue(all(type(v) is int and v >= 0 for v in result.values()))

    def test_dtypes(self):
        image = add_strip(natural_scene(seed=47), 30)
        for bad in (image.astype(object), image.astype(str), image.astype(np.complex64),
                    np.full((64, 64), b"x"), np.array([[None] * 64] * 64)):
            with self.subTest(dtype=str(bad.dtype)):
                self.assertEqual(detect_data_strip(bad), NONE)
        for good in (image.astype(np.uint16), image.astype(np.int64), image.astype(np.float16)):
            with self.subTest(dtype=str(good.dtype)):
                self.assertEqual(detect_data_strip(good), {"top": 0, "bottom": 30})
        self.assertEqual(detect_data_strip(image.astype(np.int64) - 1000), NONE)  # all "dark": no edge
        self.assertEqual(detect_data_strip(image > 100), NONE)

    def test_memory_layouts_give_identical_results_and_input_is_not_modified(self):
        image = add_strip(add_strip(natural_scene(seed=48), 36), 24, top=True, seed=49)
        before = image.copy()
        padded = np.zeros((600, 900, 3), np.uint8)
        padded[:, 50:850] = image
        readonly = image.copy()
        readonly.setflags(write=False)
        for view in (image, np.asfortranarray(image), padded[:, 50:850], readonly, image[:, :, [0, 1, 2]]):
            result = detect_data_strip(view)
            self.assertEqual(result, {"top": 24, "bottom": 36})
            self.assertEqual(json.loads(json.dumps(result)), result)
        self.assertTrue(np.array_equal(image, before))

    def test_row_order_and_vertical_flip_symmetry(self):
        image = add_strip(natural_scene(seed=50), 33)
        self.assertEqual(detect_data_strip(image[::-1]), {"top": 33, "bottom": 0})
        self.assertEqual(detect_data_strip(image[:, ::-1]), {"top": 0, "bottom": 33})

    def test_large_input_stays_fast(self):
        image = np.full((5184, 6912, 3), 120, np.uint8)  # 36 MP, the largest frame seen from these cameras
        image[-300:] = 0
        started = time.perf_counter()
        result = detect_data_strip(image)
        self.assertEqual(result, {"top": 0, "bottom": 300})
        self.assertLess(time.perf_counter() - started, 0.25)


class AdversarialInDataStripTests(unittest.TestCase):
    STRIP = {"top": 20, "bottom": 30}

    def test_string_and_mapping_boxes_are_invalid(self):
        for box in ("0099", b"0099", "0,0,9,9", {"x1": 0, "y1": 0, "x2": 9, "y2": 9},
                    [10, 280, 50, 300, 1], object()):
            with self.subTest(box=repr(box)):
                self.assertIs(in_data_strip(box, self.STRIP, 300), False)

    def test_numpy_and_iterator_inputs(self):
        self.assertIs(in_data_strip(np.array([10, 280, 50, 300], np.float32),
                                    {"top": np.int64(20), "bottom": np.int64(30)}, np.int64(300)), True)
        self.assertIs(in_data_strip(iter((10, 280, 50, 300)), self.STRIP, 300.0, np.float64(0.5)), True)
        self.assertIs(in_data_strip((10, 280, 50, 300), self.STRIP, 300, fraction=np.float32(0.5)), True)

    def test_bad_fraction_values(self):
        box = [10, 280, 50, 300]
        for fraction in (None, float("nan"), float("inf"), "half"):
            with self.subTest(fraction=fraction):
                self.assertIs(in_data_strip(box, self.STRIP, 300, fraction), False)
        self.assertIs(in_data_strip(box, self.STRIP, 300, 1.01), False)
        self.assertIs(in_data_strip(box, self.STRIP, 300, -1), True)

    def test_extreme_coordinates_are_clipped_without_overflow(self):
        huge = [0, -1e308, 10, 1e308]  # clipped to the full image height: 50 of 300 rows are strip
        self.assertIs(in_data_strip(huge, self.STRIP, 300), False)
        self.assertIs(in_data_strip(huge, self.STRIP, 300, fraction=50 / 300), True)
        self.assertIs(in_data_strip([0, float("inf"), 10, 300], self.STRIP, 300), False)
        self.assertIs(in_data_strip([0, 290, 1e-12, 300], self.STRIP, 300), True)  # thin but positive width
        self.assertIs(in_data_strip([10, 299.999, 50, 300], self.STRIP, 300), True)

    def test_nan_or_huge_strip_values(self):
        nan_top = {"top": float("nan"), "bottom": 30}
        self.assertIs(in_data_strip([10, 0, 50, 20], nan_top, 300), False)  # NaN side counts as no strip
        self.assertIs(in_data_strip([10, 280, 50, 300], nan_top, 300), True)
        self.assertIs(in_data_strip([10, 0, 50, 20], {"top": float("inf"), "bottom": 0}, 300), False)
        self.assertIs(in_data_strip([10, 100, 50, 200], {"top": 1e9, "bottom": 1e9}, 300, fraction=1.0), True)


class AdversarialCameraIdTests(unittest.TestCase):
    def test_bytes_unicode_and_unusual_paths(self):
        hebrew = "\u05d0\u05ea\u05e8/\u05de\u05e6\u05dc\u05de\u05d4 1"  # "site/camera 1"
        cases = {b"cam1/x.jpg": "cam1", hebrew + "/IMG_0001.JPG": hebrew,
                 "\\\\server\\share\\cam\\x.jpg": "//server/share/cam",
                 "cam 1\\x.jpg": "cam 1", "../x.jpg": "..", "a/./b/x.jpg": "a/b"}
        for path, expected in cases.items():
            with self.subTest(path=repr(path)):
                self.assertEqual(camera_id_for(path), expected)
                self.assertIs(type(camera_id_for(path)), str)

    def test_invalid_pattern_types_are_configuration_errors(self):
        for pattern in (123, b"(?P<camera>x)", re.compile(b"(?P<camera>x)"), ["(?P<camera>x)"], ""):
            with self.subTest(pattern=repr(pattern)), self.assertRaises(ValueError):
                camera_id_for("x/y.jpg", "regex", pattern)

    def test_regex_fallbacks_and_determinism(self):
        self.assertEqual(camera_id_for("a/b.jpg", "regex", r"(?P<camera>CAM\d+)?b"), "a")  # group did not take part
        self.assertEqual(camera_id_for(None, "regex", r"(?P<camera>.*)"), ".")  # empty match
        self.assertEqual(camera_id_for("x/CAM1/CAM2/y.jpg", "regex", r"(?P<camera>CAM\d)"), "CAM1")  # first match
        self.assertEqual(camera_id_for("cam/X.jpg", "Regex", r"(?i)(?P<camera>x)"), "X")
        outputs = {camera_id_for("s/c/x.jpg", mode, pattern) for mode, pattern in
                   (("folder", None), ("folder", None), ("FOLDER", r"ignored("))}
        self.assertEqual(outputs, {"s/c"})
        for mode in (None, "", " folder", "folders"):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                camera_id_for("s/c/x.jpg", mode)


if __name__ == "__main__":
    unittest.main()
