"""Optional adult/child second opinion from a local vision-language model.

Not part of the default pipeline: it needs a local Ollama server with a vision
model (tested: ``gemma4:26b`` on a 24 GB GPU, ~2.5 s per photo). Images never
leave the machine. Two ways to ask:

* ``photo`` (default): the whole photo, one request; the model returns how
  many visible people are adults / teens / children / unclear. Most accurate
  per photo on the local validation (children-present F1 ~0.8), but the
  answer is not tied to individual detections.
* ``marks``: the whole photo with every detected person boxed and
  numbered; the model returns an age group per number. It sees context
  (nearby adults, strollers, carried children).
* ``mosaic``: each person cropped with some surrounding context, up to 16
  crops tiled into ONE numbered image per request (several separate images
  per request made Gemma answer for only one or a few of them).
* ``crops``: separate crop images per request (kept for comparison; not
  recommended for the reason above).

Ages are appearance judgements (adult / teen / child / unclear), never
identity. When the server or model is unavailable every call returns
``None`` and the geometry estimate stands.
"""
from __future__ import annotations

import base64
import io
import json
import math
import time

AGES = ("adult", "teen", "child", "unclear")
PROMPT_VERSION = "vlm_age_v1"
MARKS_PROMPT = (
    "This is a trail-camera photograph. Every detected person is outlined by a coloured box with a white "
    "number on a dark label at the box's top-left corner. For EACH number, judge the age group of the person "
    "inside that box from visible evidence: body proportions (head size relative to body, limb length), "
    "height compared with nearby adults at a similar distance, clothing and context (held by the hand, carried, "
    "in a stroller or baby carrier). Answer 'child' only for a clearly pre-teen child (roughly under 12, "
    "including toddlers and carried babies), 'teen' for adolescents, 'adult' for grown-ups, and 'unclear' when "
    "the person is too small, dark, blurred or hidden to judge. If a box contains no person, answer 'unclear'. "
    "Return one entry for every number shown.")
CROPS_PROMPT = (
    "Each attached image is a crop from a trail-camera photograph showing ONE detected person near its centre "
    "(image 1 is person 1, image 2 is person 2, and so on). For each person judge the age group from visible "
    "evidence: body proportions (head size relative to body, limb length), clothing and context. Answer 'child' "
    "only for a clearly pre-teen child (roughly under 12, including toddlers and carried babies), 'teen' for "
    "adolescents, 'adult' for grown-ups, 'unclear' if the crop is too small, dark or blurred to judge. Return "
    "one entry per image, using the image number as id.")
MOSAIC_PROMPT = (
    "This image is a grid of numbered panels. Each panel is a crop from a trail-camera photograph showing ONE "
    "detected person near its centre; the white number on a dark label identifies the panel. For EACH number judge "
    "the age group of the person in that panel from visible evidence: body proportions (head size relative to body, "
    "limb length), clothing and context. Answer 'child' only for a clearly pre-teen child (roughly under 12, "
    "including toddlers and carried babies), 'teen' for adolescents, 'adult' for grown-ups, 'unclear' if the panel is "
    "too small, dark or blurred to judge. Return one entry for every number shown.")
PHOTO_PROMPT = (
    "This is a trail-camera photograph. Count the real people visible anywhere in it (including partly hidden, "
    "distant, riding or carried people; each person once; the black data bar at the bottom is not scene content) "
    "and split them by apparent age group: 'children' = clearly pre-teen children (roughly under 12, including "
    "toddlers and carried babies), 'teens' = adolescents, 'adults' = grown-ups, 'unclear' = people too small, dark, "
    "blurred or hidden to judge. Judge from body proportions (head size relative to body, limb length), height "
    "compared with nearby adults at a similar distance and context (held by the hand, carried, stroller). "
    "people_total must equal adults + teens + children + unclear.")
PHOTO_SCHEMA = {"type": "object", "properties": {k: {"type": "integer", "minimum": 0} for k in
                ("people_total", "adults", "teens", "children", "unclear")},
                "required": ["people_total", "adults", "teens", "children", "unclear"]}
SCHEMA = {"type": "object", "properties": {"people": {"type": "array", "items": {
    "type": "object", "properties": {"id": {"type": "integer"}, "age": {"type": "string", "enum": list(AGES)}},
    "required": ["id", "age"]}}}, "required": ["people"]}
COLORS = ("#ff3b30", "#34c759", "#0a84ff", "#ffcc00", "#ff2d95", "#5ac8fa", "#ff9500", "#af52de")


def _jpeg_b64(image, quality=90):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _font(size):
    from PIL import ImageFont
    for name in ("arialbd.ttf", "C:/Windows/Fonts/arialbd.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def marked_image(image, persons, max_side=1600):
    """Downscaled RGB copy with numbered person boxes; returns (image, {number: person_id})."""
    from PIL import ImageDraw
    scale = min(1.0, max_side / max(image.size))
    canvas = image.resize((round(image.width * scale), round(image.height * scale))) if scale < 1 else image.copy()
    draw = ImageDraw.Draw(canvas)
    width = max(3, round(max(canvas.size) / 450))
    font = _font(max(18, round(max(canvas.size) / 55)))
    mapping = {}
    for number, person in enumerate(persons, 1):
        x1, y1, x2, y2 = [v * scale for v in person["xyxy"]]
        color = COLORS[(number - 1) % len(COLORS)]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=width)
        text = str(number)
        box = draw.textbbox((0, 0), text, font=font)
        tw, th = box[2] - box[0], box[3] - box[1]
        lx, ly = max(0, x1), max(0, y1 - th - 8)
        draw.rectangle((lx, ly, lx + tw + 8, ly + th + 8), fill="#101010", outline=color, width=2)
        draw.text((lx + 4, ly + 2), text, fill="white", font=font)
        mapping[number] = person["person_id"]
    return canvas, mapping


def person_crops(image, persons, context=(0.35, 0.12), max_side=448):
    """One crop per person with context margins (fractions of box width/height)."""
    crops, mapping = [], {}
    for number, person in enumerate(persons, 1):
        x1, y1, x2, y2 = person["xyxy"]
        w, h = x2 - x1, y2 - y1
        box = (max(0, math.floor(x1 - context[0] * w)), max(0, math.floor(y1 - context[1] * h)),
               min(image.width, math.ceil(x2 + context[0] * w)), min(image.height, math.ceil(y2 + context[1] * h)))
        crop = image.crop(box)
        crop.thumbnail((max_side, max_side))
        crops.append(crop)
        mapping[number] = person["person_id"]
    return crops, mapping


def mosaic(crops, cell=320, columns=4):
    """Tile crops (letterboxed into square cells) into one image with numbered labels."""
    from PIL import Image, ImageDraw
    rows = math.ceil(len(crops) / columns)
    sheet = Image.new("RGB", (columns * cell, rows * cell), "#202020")
    draw = ImageDraw.Draw(sheet)
    font = _font(28)
    for index, crop in enumerate(crops):
        tile = crop.copy()
        tile.thumbnail((cell - 8, cell - 8))
        x0, y0 = (index % columns) * cell, (index // columns) * cell
        sheet.paste(tile, (x0 + (cell - tile.width) // 2, y0 + (cell - tile.height) // 2))
        text = str(index + 1)
        box = draw.textbbox((0, 0), text, font=font)
        draw.rectangle((x0 + 2, y0 + 2, x0 + box[2] - box[0] + 14, y0 + box[3] - box[1] + 14), fill="#101010")
        draw.text((x0 + 8, y0 + 4), text, fill="white", font=font)
    return sheet


class OllamaAge:
    """Ask a local Ollama vision model for per-person age groups."""

    def __init__(self, model="gemma4:26b", host="http://localhost:11434", mode="photo",
                 timeout=180.0, max_people=40, crops_per_request=8):
        if mode not in ("photo", "marks", "mosaic", "crops"):
            raise ValueError("mode must be photo, marks, mosaic or crops")
        self.model, self.host, self.mode = model, host.rstrip("/"), mode
        self.timeout, self.max_people, self.crops_per_request = timeout, max_people, crops_per_request
        self.config = {"model": model, "mode": mode, "prompt_version": PROMPT_VERSION, "max_people": max_people}

    def available(self):
        """True when the server answers and lists the model (``name`` means ``name:latest``)."""
        def norm(name):
            name = str(name or "").strip().lower()
            return name if ":" in name.rsplit("/", 1)[-1] else name + ":latest"
        try:
            import requests
            tags = requests.get(self.host + "/api/tags", timeout=5).json()
            models = tags.get("models", []) if isinstance(tags, dict) else []
            wanted = norm(self.model)
            return any(isinstance(m, dict) and wanted in (norm(m.get("name")), norm(m.get("model"))) for m in models)
        except Exception:
            return False

    def _chat(self, prompt, images, expected, schema=None):
        """One request; the answer length is capped (runaway generations break the JSON)."""
        import requests
        body = {"model": self.model, "stream": False, "format": schema or SCHEMA, "think": False,
                "options": {"temperature": 0, "num_ctx": 8192, "num_predict": 48 + 20 * expected},
                "messages": [{"role": "user", "content": prompt, "images": [_jpeg_b64(i) for i in images]}]}
        last = None
        for attempt in range(2):
            response = requests.post(self.host + "/api/chat", json=body, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            try:
                return json.loads(data["message"]["content"]), data
            except (KeyError, ValueError) as exc:
                last = exc
                body["options"]["num_predict"] = 48 + 30 * expected
        raise ValueError(f"Unparseable model answer: {last}")

    def count_photo(self, image, max_side=2048):
        """Whole-photo age-group counts: {"counts": {...} | None, "seconds", "error"}."""
        started, counts, error = time.perf_counter(), None, None
        try:
            canvas = image.copy()
            canvas.thumbnail((max_side, max_side))
            answer, _ = self._chat(PHOTO_PROMPT, [canvas], 3, PHOTO_SCHEMA)
            values = {k: answer[k] for k in PHOTO_SCHEMA["required"]}
            if any(type(value) is not int or value < 0 for value in values.values()):
                raise ValueError("Age counts must be non-negative JSON integers, not booleans, fractions or strings")
            counts = values
            counts["people_total"] = counts["adults"] + counts["teens"] + counts["children"] + counts["unclear"]
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        return {"counts": counts, "seconds": round(time.perf_counter() - started, 3), "error": error, **self.config}

    def classify(self, image, persons):
        """Return {"ages": {person_id: age}, "seconds", "requests", "error"} for up to max_people persons.

        ``image`` is the EXIF-oriented RGB PIL image in original pixel
        coordinates; ``persons`` need ``person_id`` and ``xyxy``. Larger people
        are asked first when there are more than ``max_people``.
        """
        started = time.perf_counter()
        chosen = sorted(persons, key=lambda p: -(p["xyxy"][3] - p["xyxy"][1]))[: self.max_people]
        chosen.sort(key=lambda p: (p["xyxy"][0], p["xyxy"][1]))  # stable left-to-right numbering
        ages, requests_made, error = {}, 0, None
        if not chosen:
            return {"ages": {}, "seconds": 0.0, "requests": 0, "error": None, **self.config}
        try:
            if self.mode == "marks":
                canvas, mapping = marked_image(image, chosen)
                answer, _ = self._chat(MARKS_PROMPT + f" Numbers shown: 1 to {len(mapping)}.", [canvas], len(mapping))
                requests_made += 1
                ages.update(self._parse(answer, mapping))
            elif self.mode == "mosaic":
                crops, mapping = person_crops(image, chosen)
                numbers = sorted(mapping)
                for start in range(0, len(numbers), 16):
                    batch = numbers[start:start + 16]
                    sheet = mosaic([crops[n - 1] for n in batch])
                    answer, _ = self._chat(MOSAIC_PROMPT + f" Numbers shown: 1 to {len(batch)}.", [sheet], len(batch))
                    requests_made += 1
                    local = {i + 1: mapping[n] for i, n in enumerate(batch)}
                    ages.update(self._parse(answer, local))
            else:
                crops, mapping = person_crops(image, chosen)
                numbers = sorted(mapping)
                for start in range(0, len(numbers), self.crops_per_request):
                    batch = numbers[start:start + self.crops_per_request]
                    answer, _ = self._chat(CROPS_PROMPT + f" There are {len(batch)} images.",
                                           [crops[n - 1] for n in batch], len(batch))
                    requests_made += 1
                    local = {i + 1: mapping[n] for i, n in enumerate(batch)}
                    ages.update(self._parse(answer, local))
        except Exception as exc:  # the geometry estimate stands
            error = f"{type(exc).__name__}: {exc}"
        return {"ages": ages, "seconds": round(time.perf_counter() - started, 3), "requests": requests_made,
                "error": error, **self.config}

    @staticmethod
    def _parse(answer, mapping):
        ages = {}
        people = answer.get("people", []) if isinstance(answer, dict) else []
        for entry in people if isinstance(people, list) else []:
            if not isinstance(entry, dict):
                continue
            try:
                number = int(entry.get("id"))
            except (TypeError, ValueError):
                continue
            age = entry.get("age")
            if number in mapping and age in AGES and mapping[number] not in ages:
                ages[mapping[number]] = age
        return ages
