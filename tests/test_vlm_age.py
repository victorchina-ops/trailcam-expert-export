"""Optional VLM age expert (``trailcam.vlm_age``) without a server, a GPU or ``requests``.

``requests`` is replaced in ``sys.modules`` by a scripted fake (CI does not
install it and no test may reach a real Ollama server). Covers the numbered
full-photo overlay (``marked_image``), per-person crops with clamped context,
the mosaic layout, and ``OllamaAge``: availability, the chat request, one retry
of an unparseable (runaway) answer, whole-photo counts, per-person
classification (left-to-right numbering, ``max_people`` by height, batches of
16 mosaic panels, partial / invalid answers, errors), and ``_parse``.
"""
from __future__ import annotations

import base64
import copy
import io
import json
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

from trailcam import vlm_age
from trailcam.vlm_age import (AGES, COLORS, CROPS_PROMPT, MARKS_PROMPT, MOSAIC_PROMPT, PHOTO_PROMPT, PHOTO_SCHEMA,
                              PROMPT_VERSION, SCHEMA, OllamaAge, marked_image, mosaic, person_crops)

BACKGROUND = (90, 110, 70)
LABEL = (16, 16, 16)       # "#101010"
SHEET = (32, 32, 32)       # "#202020"


def rgb(hex_colour):
    return tuple(int(hex_colour[i:i + 2], 16) for i in (1, 3, 5))


def coded_image(width, height):
    """Every pixel encodes its own position: R = x % 256, G = y % 256, B = 16 * (x // 256) + y // 256."""
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    array = np.stack([x % 256, y % 256, 16 * (x // 256) + y // 256], axis=-1).astype(np.uint8)
    return Image.fromarray(array, "RGB")


def person(pid, box):
    return {"person_id": pid, "xyxy": [float(v) for v in box]}


def decode(b64):
    image = Image.open(io.BytesIO(base64.b64decode(b64)))
    image.load()
    return image


# ----------------------------------------------------------------------------- fake requests

class HTTPError(Exception):
    pass


class ConnectionError(Exception):  # noqa: A001 - mirrors requests.ConnectionError's name in error strings
    pass


class Response:
    def __init__(self, payload=None, status=200, json_error=None):
        self.payload, self.status, self.json_error = payload, status, json_error

    def raise_for_status(self):
        if self.status >= 400:
            raise HTTPError(f"{self.status} Server Error")

    def json(self):
        if self.json_error:
            raise self.json_error
        return copy.deepcopy(self.payload)


def chat(content):
    """An Ollama /api/chat reply whose message content is ``content`` (JSON-encoded unless a str)."""
    text = content if isinstance(content, str) else json.dumps(content)
    return Response({"model": "gemma4:26b", "message": {"role": "assistant", "content": text}, "done": True})


class FakeRequests:
    """``requests`` stand-in: scripted replies (or exceptions) in call order; records every call."""

    def __init__(self, posts=(), tags=None):
        self.replies = list(posts)
        self.tags = tags
        self.posts, self.gets = [], []
        self.module = types.ModuleType("requests")
        self.module.post, self.module.get = self.post, self.get
        self.module.HTTPError, self.module.ConnectionError = HTTPError, ConnectionError

    def post(self, url, json=None, timeout=None):
        self.posts.append({"url": url, "json": copy.deepcopy(json), "timeout": timeout})
        if not self.replies:
            raise AssertionError("unexpected request")
        reply = self.replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return reply

    def get(self, url, timeout=None):
        self.gets.append({"url": url, "timeout": timeout})
        if isinstance(self.tags, BaseException):
            raise self.tags
        return self.tags

    def installed(self):
        return patch.dict(sys.modules, {"requests": self.module})

    def prompts(self):
        return [p["json"]["messages"][0]["content"] for p in self.posts]

    def images(self, index=0):
        return [decode(b) for b in self.posts[index]["json"]["messages"][0]["images"]]


def ages_reply(*pairs):
    return chat({"people": [{"id": number, "age": age} for number, age in pairs]})


# ----------------------------------------------------------------------------- drawing helpers

class MarkedImageTests(unittest.TestCase):
    def test_numbers_follow_the_given_order_and_map_to_person_ids(self):
        image = Image.new("RGB", (800, 600), BACKGROUND)
        persons = [person("cand_0003", [500, 200, 600, 500]), person("cand_0001", [100, 200, 200, 500])]
        canvas, mapping = marked_image(image, persons)
        self.assertEqual(mapping, {1: "cand_0003", 2: "cand_0001"})
        self.assertEqual(canvas.size, (800, 600))
        self.assertEqual(canvas.getpixel((500, 350)), rgb(COLORS[0]))   # left edge of box 1
        self.assertEqual(canvas.getpixel((100, 350)), rgb(COLORS[1]))   # left edge of box 2
        self.assertEqual(canvas.getpixel((150, 350)), BACKGROUND)       # boxes are outlines only

    def test_label_is_a_dark_tag_above_the_box(self):
        image = Image.new("RGB", (800, 600), BACKGROUND)
        canvas, _ = marked_image(image, [person("p", [300, 200, 400, 500])])
        font = vlm_age._font(max(18, round(800 / 55)))
        left, top, right, bottom = ImageDraw.Draw(Image.new("RGB", (1, 1))).textbbox((0, 0), "1", font=font)
        ly = 200 - (bottom - top) - 8
        self.assertEqual(canvas.getpixel((302, ly + 2)), LABEL)        # inside the tag, left of the digit
        self.assertEqual(canvas.getpixel((300, ly + 1)), rgb(COLORS[0]))  # tag outline in the box colour

    def test_input_is_not_drawn_on(self):
        image = Image.new("RGB", (800, 600), BACKGROUND)
        canvas, _ = marked_image(image, [person("p", [300, 200, 400, 500])])
        self.assertIsNot(canvas, image)
        self.assertEqual(image.getcolors(), [(800 * 600, BACKGROUND)])

    def test_large_photos_are_downscaled_with_their_boxes(self):
        image = Image.new("RGB", (3200, 1600), BACKGROUND)
        canvas, mapping = marked_image(image, [person("p", [400, 400, 800, 1200])])
        self.assertEqual(canvas.size, (1600, 800))
        self.assertEqual(mapping, {1: "p"})
        width = max(3, round(1600 / 450))                              # outline width 4 at 1600 px
        for x in range(200, 200 + width):
            self.assertEqual(canvas.getpixel((x, 400)), rgb(COLORS[0]), x)
        self.assertEqual(canvas.getpixel((200 + width + 2, 400)), BACKGROUND)
        self.assertEqual(canvas.getpixel((400, 400)), rgb(COLORS[0]))  # right edge at 800 * 0.5
        self.assertEqual(image.size, (3200, 1600))

    def test_colours_cycle_after_eight_people(self):
        image = Image.new("RGB", (1500, 400), BACKGROUND)
        persons = [person(f"p{n}", [20 + 150 * n, 100, 100 + 150 * n, 300]) for n in range(9)]
        canvas, mapping = marked_image(image, persons)
        self.assertEqual(list(mapping), list(range(1, 10)))
        self.assertEqual(canvas.getpixel((20 + 150 * 8, 200)), rgb(COLORS[0]))
        self.assertEqual(canvas.getpixel((20 + 150 * 7, 200)), rgb(COLORS[7]))


class PersonCropTests(unittest.TestCase):
    def setUp(self):
        self.image = coded_image(1000, 800)

    def assert_crop(self, crop, origin, size):
        self.assertEqual(crop.size, size)
        self.assertEqual(crop.getpixel((0, 0)), self.image.getpixel(origin))
        self.assertEqual(crop.getpixel((size[0] - 1, size[1] - 1)),
                         self.image.getpixel((origin[0] + size[0] - 1, origin[1] + size[1] - 1)))

    def test_context_margins_and_mapping(self):
        crops, mapping = person_crops(self.image, [person("b", [100, 100, 200, 300]), person("a", [600, 300, 640, 400])])
        self.assertEqual(mapping, {1: "b", 2: "a"})
        # 0.35 * width left/right, 0.12 * height above/below: (65, 76) .. (235, 324).
        self.assert_crop(crops[0], (65, 76), (170, 248))
        # 600 - 14 = 586, 300 - 12 = 288, 640 + 14 = 654, 400 + 12 = 412.
        self.assert_crop(crops[1], (586, 288), (68, 124))

    def test_context_is_clamped_to_the_image(self):
        crops, _ = person_crops(self.image, [person("tl", [0, 0, 50, 100]), person("br", [950, 700, 1000, 800])])
        self.assert_crop(crops[0], (0, 0), (68, 112))       # ceil(50 + 17.5), ceil(100 + 12)
        self.assert_crop(crops[1], (932, 688), (68, 112))   # floor(950 - 17.5), floor(700 - 12); clamped right/bottom

    def test_fractional_boxes_round_outwards(self):
        crops, _ = person_crops(self.image, [person("p", [10.4, 20.6, 50.2, 80.1])], context=(0, 0))
        self.assert_crop(crops[0], (10, 20), (41, 61))

    def test_large_crops_are_reduced_to_max_side(self):
        crops, _ = person_crops(self.image, [person("all", [0, 0, 1000, 800])])
        self.assertEqual(max(crops[0].size), 448)
        self.assertAlmostEqual(crops[0].width / crops[0].height, 1000 / 800, places=2)
        small, _ = person_crops(self.image, [person("all", [0, 0, 1000, 800])], max_side=100)
        self.assertEqual(max(small[0].size), 100)
        self.assertEqual(self.image.size, (1000, 800))

    def test_no_people(self):
        self.assertEqual(person_crops(self.image, []), ([], {}))


class MosaicTests(unittest.TestCase):
    COLOURS = [(200, 30, 30), (30, 200, 30), (30, 30, 200), (200, 200, 30), (200, 30, 200)]

    def crops(self, count, size=(100, 300)):
        return [Image.new("RGB", size, self.COLOURS[n % len(self.COLOURS)]) for n in range(count)]

    def test_sheet_size_is_four_columns_of_320_px_cells(self):
        for count, rows in ((1, 1), (4, 1), (5, 2), (8, 2), (16, 4), (17, 5)):
            with self.subTest(count=count):
                self.assertEqual(mosaic(self.crops(count)).size, (1280, 320 * rows))

    def test_crops_are_centred_letterboxed_and_placed_row_by_row(self):
        sheet = mosaic(self.crops(5, size=(400, 1200)))
        for index in range(5):
            x0, y0 = (index % 4) * 320, (index // 4) * 320
            self.assertEqual(sheet.getpixel((x0 + 160, y0 + 160)), self.COLOURS[index], index)
        # A 400 x 1200 crop is fitted into 312 px (104 x 312) and centred: x 108..211, y 4..315.
        for x, y, colour in ((108, 160, self.COLOURS[0]), (211, 160, self.COLOURS[0]), (107, 160, SHEET),
                             (212, 160, SHEET), (160, 315, self.COLOURS[0]), (160, 317, SHEET),
                             (320 + 108, 160, self.COLOURS[1]), (108, 320 + 160, self.COLOURS[4])):
            self.assertEqual(sheet.getpixel((x, y)), colour, (x, y))
        self.assertEqual(sheet.getpixel((480, 480)), SHEET)   # empty cell of the last row

    def test_small_crops_are_not_enlarged(self):
        sheet = mosaic([Image.new("RGB", (20, 20), (250, 250, 250)), Image.new("RGB", (100, 300), (5, 250, 5))])
        self.assertEqual(sheet.getpixel((150, 150)), (250, 250, 250))   # 20 x 20 at 150..169
        self.assertEqual(sheet.getpixel((169, 169)), (250, 250, 250))
        self.assertEqual(sheet.getpixel((149, 160)), SHEET)
        self.assertEqual(sheet.getpixel((170, 160)), SHEET)
        self.assertEqual(sheet.getpixel((320 + 110, 10)), (5, 250, 5))   # 100 x 300 at 110..209, 10..309
        self.assertEqual(sheet.getpixel((320 + 109, 160)), SHEET)
        self.assertEqual(sheet.getpixel((320 + 210, 160)), SHEET)

    def test_every_panel_has_a_dark_number_tag(self):
        sheet = mosaic(self.crops(6, size=(400, 400)))   # tiles cover the whole cell except a 4 px border
        for index in range(6):
            x0, y0 = (index % 4) * 320, (index // 4) * 320
            self.assertEqual(sheet.getpixel((x0 + 5, y0 + 5)), LABEL, index)

    def test_crops_are_not_modified(self):
        crops = self.crops(2, size=(600, 600))
        mosaic(crops)
        self.assertEqual([c.size for c in crops], [(600, 600), (600, 600)])


# ----------------------------------------------------------------------------- OllamaAge

class ConstructionTests(unittest.TestCase):
    def test_modes(self):
        for mode in ("photo", "marks", "mosaic", "crops"):
            self.assertEqual(OllamaAge(mode=mode).mode, mode)
        for mode in ("photo+mosaic", "Photo", "", None):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                OllamaAge(mode=mode)

    def test_defaults_and_config(self):
        engine = OllamaAge()
        self.assertEqual((engine.model, engine.host, engine.mode, engine.max_people, engine.crops_per_request),
                         ("gemma4:26b", "http://localhost:11434", "photo", 40, 8))
        engine = OllamaAge("m:1", "http://gpu:11434///", "mosaic", max_people=5)
        self.assertEqual(engine.host, "http://gpu:11434")
        self.assertEqual(engine.config, {"model": "m:1", "mode": "mosaic", "prompt_version": PROMPT_VERSION,
                                         "max_people": 5})

    def test_schemas(self):
        self.assertEqual(AGES, ("adult", "teen", "child", "unclear"))
        self.assertEqual(SCHEMA["properties"]["people"]["items"]["properties"]["age"]["enum"], list(AGES))
        self.assertEqual(PHOTO_SCHEMA["required"], ["people_total", "adults", "teens", "children", "unclear"])


class AvailableTests(unittest.TestCase):
    def check(self, tags, model="gemma4:26b"):
        fake = FakeRequests(tags=tags)
        with fake.installed():
            answer = OllamaAge(model, "http://host:1/").available()
        if fake.gets:
            self.assertEqual(fake.gets, [{"url": "http://host:1/api/tags", "timeout": 5}])
        self.assertEqual(fake.posts, [])
        return answer

    def test_listed_model_by_name_or_model_key(self):
        self.assertTrue(self.check(Response({"models": [{"name": "gemma3:4b"}, {"name": "gemma4:26b"}]})))
        self.assertTrue(self.check(Response({"models": [{"model": "gemma4:26b"}]})))

    def test_missing_model(self):
        self.assertFalse(self.check(Response({"models": [{"name": "gemma4:31b", "model": "gemma4:31b"}]})))
        self.assertFalse(self.check(Response({"models": []})))
        self.assertFalse(self.check(Response({})))

    def test_server_down_or_garbage(self):
        self.assertFalse(self.check(ConnectionError("connection refused")))
        self.assertFalse(self.check(TimeoutError("timed out")))
        self.assertFalse(self.check(Response(json_error=ValueError("not JSON"))))

    def test_non_dict_json_is_not_available(self):
        # Changed: the reply is parsed inside the try; a JSON list/str/number/null is False, not an exception.
        for payload in ([], ["gemma4:26b"], "gemma4:26b", 3, None, {"models": None}, {"models": "gemma4:26b"},
                        {"models": ["gemma4:26b", None, 7]}):
            with self.subTest(payload=payload):
                self.assertFalse(self.check(Response(payload)))
        # Non-dict entries are skipped, a later valid entry still counts.
        self.assertTrue(self.check(Response({"models": ["gemma4:26b", None, {"name": "gemma4:26b"}]})))

    def test_bare_name_means_latest(self):
        # Changed: names are normalised, so "gemma4" and "gemma4:latest" are the same model.
        self.assertTrue(self.check(Response({"models": [{"name": "gemma4:latest"}]}), model="gemma4"))
        self.assertTrue(self.check(Response({"models": [{"name": "gemma4"}]}), model="gemma4:latest"))
        self.assertTrue(self.check(Response({"models": [{"model": "gemma4:latest"}]}), model="gemma4"))
        self.assertTrue(self.check(Response({"models": [{"name": "library/gemma4:latest"}]}), model="library/gemma4"))
        # A bare name is only :latest, never another tag of the same family.
        self.assertFalse(self.check(Response({"models": [{"name": "gemma4:26b"}]}), model="gemma4"))
        self.assertFalse(self.check(Response({"models": [{"name": "gemma4:latest"}]}), model="gemma4:26b"))
        self.assertFalse(self.check(Response({"models": [{"name": "gemma4:26b-latest"}]}), model="gemma4"))

    def test_names_are_compared_case_insensitively_and_trimmed(self):
        self.assertTrue(self.check(Response({"models": [{"name": "Gemma4:26B"}]}), model="gemma4:26b"))
        self.assertTrue(self.check(Response({"models": [{"name": "gemma4:26b"}]}), model=" GEMMA4:26b "))
        self.assertTrue(self.check(Response({"models": [{"name": None, "model": "qwen2.5vl:7b"}]}), model="qwen2.5vl:7b"))
        self.assertFalse(self.check(Response({"models": [{"name": None}]}), model="gemma4"))


class ChatTests(unittest.TestCase):
    def setUp(self):
        self.engine = OllamaAge("gemma4:26b", "http://host:1", "mosaic", timeout=12.5)
        self.picture = Image.new("RGB", (64, 48), (10, 20, 30))

    def test_request_body_and_answer(self):
        fake = FakeRequests([chat({"people": [{"id": 1, "age": "adult"}]})])
        with fake.installed():
            answer, data = self.engine._chat("Prompt text", [self.picture, self.picture], 4)
        self.assertEqual(answer, {"people": [{"id": 1, "age": "adult"}]})
        self.assertEqual(data["message"]["role"], "assistant")
        call, = fake.posts
        self.assertEqual((call["url"], call["timeout"]), ("http://host:1/api/chat", 12.5))
        body = call["json"]
        self.assertEqual({k: body[k] for k in ("model", "stream", "think", "format")},
                         {"model": "gemma4:26b", "stream": False, "think": False, "format": SCHEMA})
        self.assertEqual(body["options"], {"temperature": 0, "num_ctx": 8192, "num_predict": 48 + 20 * 4})
        message, = body["messages"]
        self.assertEqual((message["role"], message["content"]), ("user", "Prompt text"))
        images = fake.images()
        self.assertEqual([(i.format, i.size) for i in images], [("JPEG", (64, 48))] * 2)

    def test_custom_schema(self):
        fake = FakeRequests([chat({"adults": 1})])
        with fake.installed():
            self.engine._chat("p", [self.picture], 3, PHOTO_SCHEMA)
        self.assertEqual(fake.posts[0]["json"]["format"], PHOTO_SCHEMA)

    def test_runaway_answer_is_retried_once_with_a_larger_budget(self):
        runaway = '{"people": [{"id": 1, "age": "adult"}, {"id": 2, "age": "chi'
        fake = FakeRequests([chat(runaway), chat({"people": [{"id": 2, "age": "child"}]})])
        with fake.installed():
            answer, _ = self.engine._chat("p", [self.picture], 5)
        self.assertEqual(answer, {"people": [{"id": 2, "age": "child"}]})
        self.assertEqual([p["json"]["options"]["num_predict"] for p in fake.posts], [48 + 20 * 5, 48 + 30 * 5])
        self.assertEqual(fake.posts[0]["json"]["messages"], fake.posts[1]["json"]["messages"])

    def test_two_unparseable_answers_raise(self):
        for bad in ("not json at all", '{"people": [', ""):
            with self.subTest(bad=bad):
                fake = FakeRequests([chat(bad), chat(bad), chat({"people": []})])
                with fake.installed(), self.assertRaisesRegex(ValueError, "Unparseable model answer"):
                    self.engine._chat("p", [self.picture], 1)
                self.assertEqual(len(fake.posts), 2)     # one retry only

    def test_reply_without_a_message_is_retried(self):
        fake = FakeRequests([Response({"error": "model is loading"}), chat({"people": []})])
        with fake.installed():
            self.assertEqual(self.engine._chat("p", [self.picture], 1)[0], {"people": []})
        fake = FakeRequests([Response({"error": "x"}), Response({"done": True})])
        with fake.installed(), self.assertRaisesRegex(ValueError, "Unparseable"):
            self.engine._chat("p", [self.picture], 1)

    def test_http_and_connection_errors_are_not_retried(self):
        for failure in (Response({"error": "boom"}, status=500), ConnectionError("refused")):
            fake = FakeRequests([failure, chat({"people": []})])
            with fake.installed(), self.assertRaises((HTTPError, ConnectionError)):
                self.engine._chat("p", [self.picture], 1)
            self.assertEqual(len(fake.posts), 1)


class CountPhotoTests(unittest.TestCase):
    COUNTS = {"people_total": 4, "adults": 2, "teens": 1, "children": 1, "unclear": 0}

    def count(self, replies, image=None, **options):
        fake = FakeRequests(replies)
        engine = OllamaAge("gemma4:26b", "http://host:1", "photo", **options)
        with fake.installed():
            answer = engine.count_photo(image or Image.new("RGB", (640, 480), BACKGROUND))
        return answer, fake

    def test_counts_and_request(self):
        answer, fake = self.count([chat(self.COUNTS)])
        self.assertEqual(answer["counts"], self.COUNTS)
        self.assertIsNone(answer["error"])
        self.assertGreaterEqual(answer["seconds"], 0)
        self.assertEqual({k: answer[k] for k in ("model", "mode", "prompt_version", "max_people")},
                         {"model": "gemma4:26b", "mode": "photo", "prompt_version": PROMPT_VERSION, "max_people": 40})
        body = fake.posts[0]["json"]
        self.assertEqual((body["messages"][0]["content"], body["format"]), (PHOTO_PROMPT, PHOTO_SCHEMA))
        self.assertEqual(body["options"]["num_predict"], 48 + 20 * 3)
        self.assertEqual([i.size for i in fake.images()], [(640, 480)])
        json.dumps(answer, allow_nan=False)  # cached as JSON by __main__

    def test_total_is_recomputed_from_the_groups(self):
        answer, _ = self.count([chat({**self.COUNTS, "people_total": 9})])
        self.assertEqual(answer["counts"]["people_total"], 4)
        answer, _ = self.count([chat({"people_total": 0, "adults": 1, "teens": 0, "children": 2, "unclear": 3})])
        self.assertEqual(answer["counts"], {"people_total": 6, "adults": 1, "teens": 0, "children": 2, "unclear": 3})

    def test_raw_count_values_are_validated_before_any_coercion(self):
        for key in PHOTO_SCHEMA["required"]:
            for value in (True, False, -1, 1.8, 1.0, "2", None, [], {}):
                with self.subTest(key=key, value=value):
                    answer, fake = self.count([chat({**self.COUNTS, key: value})])
                    self.assertIsNone(answer["counts"])
                    self.assertIn("non-negative JSON integers", answer["error"])
                    self.assertEqual(len(fake.posts), 1)
        answer, _ = self.count([chat(dict.fromkeys(PHOTO_SCHEMA["required"], 0))])
        self.assertEqual(answer["counts"], dict.fromkeys(PHOTO_SCHEMA["required"], 0))
        self.assertIsNone(answer["error"])

    def test_large_photos_are_reduced_to_2048_px(self):
        big = Image.new("RGB", (4000, 3000), BACKGROUND)
        _, fake = self.count([chat(self.COUNTS)], image=big)
        self.assertEqual([i.size for i in fake.images()], [(2048, 1536)])
        self.assertEqual(big.size, (4000, 3000))

    def test_runaway_json_is_retried_once(self):
        answer, fake = self.count([chat('{"people_total": 3, "adults": 3, "te'), chat(self.COUNTS)])
        self.assertEqual((answer["counts"], answer["error"]), (self.COUNTS, None))
        self.assertEqual(len(fake.posts), 2)

    def test_second_invalid_answer_is_an_error(self):
        answer, fake = self.count([chat("{"), chat("{\"adults\": 1")])
        self.assertIsNone(answer["counts"])
        self.assertTrue(answer["error"].startswith("ValueError: Unparseable model answer"), answer["error"])
        self.assertEqual(len(fake.posts), 2)

    def test_server_down_is_an_error_not_an_exception(self):
        answer, fake = self.count([ConnectionError("connection refused")])
        self.assertEqual((answer["counts"], answer["error"]), (None, "ConnectionError: connection refused"))
        self.assertEqual(answer["model"], "gemma4:26b")
        answer, _ = self.count([Response({"error": "out of memory"}, status=500)])
        self.assertEqual((answer["counts"], answer["error"]), (None, "HTTPError: 500 Server Error"))

    def test_missing_group_is_an_error_without_retry(self):
        answer, fake = self.count([chat({"people_total": 1, "adults": 1, "teens": 0, "unclear": 0}), chat(self.COUNTS)])
        self.assertIsNone(answer["counts"])
        self.assertTrue(answer["error"].startswith("KeyError"), answer["error"])
        self.assertEqual(len(fake.posts), 1)


class ClassifyTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (1000, 600), BACKGROUND)
        # Given out of left-to-right order on purpose.
        self.persons = [person("right", [700, 100, 760, 400]), person("left", [100, 150, 150, 350]),
                        person("middle", [400, 100, 460, 450])]

    def classify(self, replies, persons=None, mode="mosaic", **options):
        fake = FakeRequests(replies)
        engine = OllamaAge("gemma4:26b", "http://host:1", mode, **options)
        with fake.installed():
            answer = engine.classify(self.image, self.persons if persons is None else persons)
        return answer, fake

    def test_no_people_makes_no_request(self):
        answer, fake = self.classify([], persons=[])
        self.assertEqual(answer, {"ages": {}, "seconds": 0.0, "requests": 0, "error": None,
                                  "model": "gemma4:26b", "mode": "mosaic", "prompt_version": PROMPT_VERSION,
                                  "max_people": 40})
        self.assertEqual(fake.posts, [])

    def test_mosaic_numbers_people_left_to_right(self):
        answer, fake = self.classify([ages_reply((1, "child"), (2, "adult"), (3, "teen"))])
        self.assertEqual(answer["ages"], {"left": "child", "middle": "adult", "right": "teen"})  # teen kept here
        self.assertEqual((answer["requests"], answer["error"]), (1, None))
        call, = fake.posts
        self.assertEqual(call["json"]["messages"][0]["content"], MOSAIC_PROMPT + " Numbers shown: 1 to 3.")
        self.assertEqual((call["json"]["format"], call["json"]["options"]["num_predict"]), (SCHEMA, 48 + 20 * 3))
        self.assertEqual([i.size for i in fake.images()], [(1280, 320)])
        json.dumps(answer, allow_nan=False)

    def test_equal_left_edges_are_numbered_top_to_bottom(self):
        persons = [person("low", [100, 300, 150, 500]), person("high", [100, 50, 150, 250])]
        answer, _ = self.classify([ages_reply((1, "adult"), (2, "child"))], persons=persons)
        self.assertEqual(answer["ages"], {"high": "adult", "low": "child"})

    def test_partial_answers_keep_only_what_was_answered(self):
        answer, _ = self.classify([ages_reply((1, "child"), (3, "adult"))])
        self.assertEqual(answer["ages"], {"left": "child", "right": "adult"})
        self.assertIsNone(answer["error"])

    def test_a_malformed_entry_does_not_drop_the_other_answers(self):
        # Changed: _parse skips non-dict entries instead of losing the whole batch.
        reply = chat({"people": ["1 adult", {"id": 1, "age": "adult"}, None, 2, {"id": 3, "age": "child"}]})
        answer, _ = self.classify([reply])
        self.assertEqual(answer["ages"], {"left": "adult", "right": "child"})
        self.assertEqual((answer["requests"], answer["error"]), (1, None))

    def test_invalid_entries_are_ignored(self):
        reply = chat({"people": [{"id": 0, "age": "adult"}, {"id": 4, "age": "adult"}, {"id": None, "age": "adult"},
                                 {"id": "x", "age": "adult"}, {"id": 1, "age": "elderly"}, {"id": 1, "age": "Child"},
                                 {"id": 2}, {"id": "2", "age": "unclear"}, {"id": 2, "age": "adult"},
                                 {"id": 3.0, "age": "child"}]})
        answer, _ = self.classify([reply])
        # "2" and 3.0 are read as numbers; the first valid answer for a number wins.
        self.assertEqual(answer["ages"], {"middle": "unclear", "right": "child"})

    def test_empty_or_missing_people_list(self):
        for content in ({"people": []}, {}, None):
            with self.subTest(content=content):
                answer, _ = self.classify([chat(content)])
                self.assertEqual((answer["ages"], answer["requests"], answer["error"]), ({}, 1, None))

    def test_max_people_keeps_the_tallest_then_numbers_left_to_right(self):
        persons = [person("short", [0, 300, 40, 400]), person("tallest", [500, 50, 560, 400]),
                   person("tall", [200, 150, 250, 400])]
        answer, fake = self.classify([ages_reply((1, "adult"), (2, "child"), (3, "adult"))], persons=persons,
                                     max_people=2)
        self.assertEqual(answer["ages"], {"tall": "adult", "tallest": "child"})   # number 3 does not exist
        self.assertEqual(fake.prompts(), [MOSAIC_PROMPT + " Numbers shown: 1 to 2."])
        self.assertEqual(answer["max_people"], 2)

    def test_mosaic_batches_of_sixteen_panels(self):
        persons = [person(f"p{n:02d}", [10 + 50 * n, 100, 40 + 50 * n, 300]) for n in range(17)]
        answer, fake = self.classify([ages_reply(*[(n, "adult") for n in range(1, 17)]), ages_reply((1, "child"))],
                                     persons=list(reversed(persons)))
        self.assertEqual(fake.prompts(), [MOSAIC_PROMPT + " Numbers shown: 1 to 16.",
                                          MOSAIC_PROMPT + " Numbers shown: 1 to 1."])
        self.assertEqual([fake.images(i)[0].size for i in range(2)], [(1280, 1280), (1280, 320)])
        self.assertEqual([p["json"]["options"]["num_predict"] for p in fake.posts], [48 + 20 * 16, 48 + 20])
        self.assertEqual(answer["ages"], {**{f"p{n:02d}": "adult" for n in range(16)}, "p16": "child"})
        self.assertEqual((answer["requests"], answer["error"]), (2, None))

    def test_a_failed_batch_keeps_earlier_answers(self):
        persons = [person(f"p{n:02d}", [10 + 50 * n, 100, 40 + 50 * n, 300]) for n in range(17)]
        answer, fake = self.classify([ages_reply((1, "child"), (2, "adult")), ConnectionError("reset")],
                                     persons=persons)
        self.assertEqual(answer["ages"], {"p00": "child", "p01": "adult"})
        self.assertEqual((answer["requests"], answer["error"]), (1, "ConnectionError: reset"))
        self.assertEqual(len(fake.posts), 2)

    def test_server_down(self):
        answer, _ = self.classify([ConnectionError("connection refused")])
        self.assertEqual((answer["ages"], answer["requests"], answer["error"]),
                         ({}, 0, "ConnectionError: connection refused"))

    def test_runaway_answer_is_retried_inside_one_request(self):
        answer, fake = self.classify([chat('{"people": [{"id": 1, "age": "adult"}, {"id'),
                                      ages_reply((1, "adult"), (2, "adult"), (3, "child"))])
        self.assertEqual(answer["ages"], {"left": "adult", "middle": "adult", "right": "child"})
        self.assertEqual((answer["requests"], len(fake.posts), answer["error"]), (1, 2, None))
        answer, fake = self.classify([chat("{"), chat("{")])
        self.assertEqual((answer["ages"], answer["requests"]), ({}, 0))
        self.assertTrue(answer["error"].startswith("ValueError: Unparseable"), answer["error"])

    def test_marks_mode_sends_the_numbered_photo(self):
        answer, fake = self.classify([ages_reply((1, "adult"), (2, "child"), (3, "unclear"))], mode="marks")
        self.assertEqual(answer["ages"], {"left": "adult", "middle": "child", "right": "unclear"})
        self.assertEqual(fake.prompts(), [MARKS_PROMPT + " Numbers shown: 1 to 3."])
        sent, = fake.images()
        self.assertEqual(sent.size, (1000, 600))
        self.assertEqual(self.image.getcolors(), [(1000 * 600, BACKGROUND)])  # drawn on a copy

    def test_crops_mode_sends_separate_images_per_batch(self):
        answer, fake = self.classify([ages_reply((1, "adult"), (2, "child")), ages_reply((1, "teen"))],
                                     mode="crops", crops_per_request=2)
        self.assertEqual(answer["ages"], {"left": "adult", "middle": "child", "right": "teen"})
        self.assertEqual(fake.prompts(), [CROPS_PROMPT + " There are 2 images.", CROPS_PROMPT + " There are 1 images."])
        self.assertEqual([len(fake.images(i)) for i in range(2)], [2, 1])
        self.assertEqual(answer["requests"], 2)

    def test_persons_are_not_mutated(self):
        before = copy.deepcopy(self.persons)
        self.classify([ages_reply((1, "adult"))])
        self.assertEqual(self.persons, before)


class ParseTests(unittest.TestCase):
    def test_parse(self):
        mapping = {1: "a", 2: "b"}
        self.assertEqual(OllamaAge._parse({"people": [{"id": 2, "age": "teen"}, {"id": 1, "age": "unclear"}]}, mapping),
                         {"b": "teen", "a": "unclear"})
        self.assertEqual(OllamaAge._parse(None, mapping), {})
        self.assertEqual(OllamaAge._parse({"people": [{"id": 2, "age": "child"}, {"id": 2, "age": "adult"}]}, mapping),
                         {"b": "child"})
        self.assertEqual(OllamaAge._parse({"people": [{"id": 1, "age": "adult"}]}, {}), {})

    def test_non_dict_entries_are_skipped_not_the_whole_batch(self):
        # Changed: one malformed entry used to drop every answer of the request.
        mapping = {1: "a", 2: "b", 3: "c"}
        people = ["1: adult", None, 7, ["id", 2], {"id": 1, "age": "adult"}, True, {"id": 3, "age": "child"}]
        self.assertEqual(OllamaAge._parse({"people": people}, mapping), {"a": "adult", "c": "child"})
        for answer in ({"people": "adult"}, {"people": None}, {"people": {"id": 1, "age": "adult"}}, ["x"], "x", 3):
            with self.subTest(answer=answer):
                self.assertEqual(OllamaAge._parse(answer, mapping), {})


if __name__ == "__main__":
    unittest.main()
