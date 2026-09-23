"""Diverse, reproducible visual samples from the exported image inventory."""
from __future__ import annotations
import math
from collections import Counter
from pathlib import Path

COLORS = {'person': '#4bd5ff', 'bicycle': '#ffd166', 'backpack': '#ed8aff',
          'dog': '#80ed99', 'baby stroller': '#ff9f68', 'motorcycle': '#ff6577'}
EDGES = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12),
         (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4)]


def features(item):
    result = item['result']
    tags = set()
    if result.get('status') != 'ok':
        tags.add('analysis_error')
    combined = result.get('combined', {})
    for field, value in combined.get('counts', {}).items():
        if value and field != 'people_total':
            tags.add(field)
    people = combined.get('counts', {}).get('people_total', 0)
    tags.add('no_people' if not people else 'one_person' if people == 1 else 'small_group' if people <= 4 else 'crowd')
    for direction, count in (combined.get('direction_counts') or {}).items():
        if count:
            tags.add('direction_' + direction)
    if combined.get('children'):
        tags.add('children')
    if combined.get('large_bags'):
        tags.add('large_bags')
    if (result.get('post') or {}).get('event_image_count', 1) > 1:
        tags.add('multi_image_event')
    if combined.get('orientation_conflicts'):
        tags.add('orientation_disagreement')
    if any(v.get('status') == 'no_majority' for v in result.get('votes', {}).values()):
        tags.add('count_disagreement')
    return tags


def select_examples(items, limit=20):
    # Prefer new/changed images, then fill with cached images. Rare categories
    # get more weight than common people, with stable path-based tie breaking.
    pool = sorted(items, key=lambda item: item['relative_path'].casefold())
    selected, covered = [], Counter()
    all_tags = {id(item): features(item) for item in pool}
    frequencies = Counter(tag for tags in all_tags.values() for tag in tags)
    for cached in (False, True):
        candidates = [item for item in pool if bool(item['cache_hit']) == cached]
        while candidates and len(selected) < limit:
            def score(item):
                return sum((1 + 1 / math.sqrt(frequencies[tag])) / (1 + covered[tag])
                           for tag in all_tags[id(item)])
            chosen = max(candidates, key=score)
            candidates.remove(chosen)
            selected.append(chosen)
            covered.update(all_tags[id(chosen)])
    return selected


def make_contact_sheet(items, output_path, run_id, limit=20):
    from PIL import Image, ImageDraw, ImageFont, ImageOps
    selected = select_examples(items, limit)
    if not selected:
        return []
    def font(size):
        for name in ('DejaVuSans.ttf', 'C:/Windows/Fonts/arial.ttf'):
            try:
                return ImageFont.truetype(name, size)
            except OSError:
                pass
        return ImageFont.load_default()
    heading, normal, small, labels = font(25), font(17), font(15), font(12)
    cols = min(4, len(selected))
    tile_w, image_h, tile_h = 560, 380, 478
    header_h = 128
    rows = math.ceil(len(selected) / cols)
    sheet = Image.new('RGB', (cols * tile_w, header_h + rows * tile_h), '#101820')
    draw = ImageDraw.Draw(sheet)
    draw.text((16, 12), 'Trail camera visual validation | ' + run_id, font=heading, fill='white')
    draw.text((16, 46), 'YOLOE boxes (conf >= 0.25) + translucent masks; yellow pose keypoints; white boxes = accepted people.', font=normal, fill='#d2dbe5')
    draw.text((16, 73), 'Cyan: people | pink: backpacks | green: dogs | gold: bicycles | orange: other objects. Predictions need review.', font=small, fill='#d2dbe5')
    draw.text((16, 99), 'Person labels: A=adult, C=child, ?=age unknown; L/R/T/A/S=left/right/toward/away/stationary (m=motion, f=facing); B=large bag.', font=small, fill='#d2dbe5')
    manifest = []
    for index, item in enumerate(selected):
        x0, y0 = (index % cols) * tile_w, header_h + (index // cols) * tile_h
        result = item['result']
        try:
            with Image.open(item['path']) as opened:
                from .vision import to_rgb8
                original = to_rgb8(ImageOps.exif_transpose(opened))
            original.thumbnail((tile_w - 12, image_h - 8))
            canvas = original.convert('RGBA')
            overlay = Image.new('RGBA', canvas.size)
            od = ImageDraw.Draw(overlay)
            sx = canvas.width / result.get('width', canvas.width)
            sy = canvas.height / result.get('height', canvas.height)
            detections = result.get('experts', {}).get('yoloe', {}).get('detections', [])
            for det in (d for d in detections if d.get('confidence', 0) >= .25):
                label = det.get('label', det.get('class_name', 'object'))
                color = COLORS.get(label, '#ff9f68')
                polygon = det.get('mask_polygon_xy') or []
                if len(polygon) >= 3:
                    od.polygon([(p[0] * sx, p[1] * sy) for p in polygon], fill=color + '26')
                box = det.get('bbox_xyxy', det.get('xyxy'))
                if box:
                    xy = [box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy]
                    od.rectangle(xy, outline=color, width=2)
                    if label != 'person':
                        od.text((xy[0], max(0, xy[1] - 16)), label, font=small, fill=color,
                                stroke_width=1, stroke_fill='black')
            for det in (d for d in result.get('experts', {}).get('pose', {}).get('detections', []) if d.get('confidence', 0) >= .25):
                kp = det.get('keypoints', [])
                if len(kp) != 17:
                    continue
                for a, b in EDGES:
                    if kp[a][2] >= .6 and kp[b][2] >= .6:
                        od.line((kp[a][0] * sx, kp[a][1] * sy, kp[b][0] * sx, kp[b][1] * sy), fill='#ffe55a', width=2)
                for px, py, confidence in kp:
                    if confidence >= .6:
                        od.ellipse((px * sx - 2, py * sy - 2, px * sx + 2, py * sy + 2), fill='#ffe55a')
            accepted = [p for p in result.get('persons', []) if p.get('accepted')]
            for number, person in enumerate(accepted, 1):
                box = person['bbox_xyxy']
                c = person.get('combined', {})
                xy = [box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy]
                od.rectangle(xy, outline='white', width=1)
                age = {'adult': 'A', 'child': 'C'}.get(c.get('age'), '?')
                direction = {'left': 'L', 'right': 'R', 'toward': 'T', 'away': 'A', 'stationary': 'S'}.get(c.get('direction'), '?')
                source = {'motion': 'm', 'facing': 'f'}.get(c.get('direction_source'), '')
                text = f"{number} {age} {direction}{source}" + (' B' if c.get('large_bag') is True else '')
                tw = od.textlength(text, font=labels)
                od.text((max(0, min(xy[0], canvas.width - tw - 1)), max(0, xy[1] - 14)), text, font=labels, fill='white', stroke_width=1, stroke_fill='black')
            canvas = Image.alpha_composite(canvas, overlay).convert('RGB')
            sheet.paste(canvas, (x0 + (tile_w - canvas.width) // 2, y0 + (image_h - canvas.height) // 2))
        except (OSError, ValueError) as exc:
            draw.text((x0 + 10, y0 + 90), 'Image unavailable: ' + type(exc).__name__, font=normal, fill='#ff6577')
        counts = result.get('combined', {}).get('counts', {})
        path_label = item['relative_path']
        while path_label and draw.textlength(f'{index + 1:02d}  {path_label}', font=normal) > tile_w - 20:
            path_label = '...' + path_label[4:]
        draw.text((x0 + 8, y0 + image_h + 3), f'{index + 1:02d}  {path_label}', font=normal, fill='white')
        found = [f'{k.replace("_", " ")}={v}' for k, v in counts.items() if v and k != 'cars_trucks_buses']
        combined = result.get('combined', {})
        if combined.get('children'):
            found.insert(1, f"children={combined['children']}")
        event = (result.get('post') or {}).get('event_id')
        if event and (result.get('post') or {}).get('event_image_count', 1) > 1:
            found.append(f"event frames={result['post']['event_image_count']}")
        label = ', '.join(found) or 'No accepted detections'
        # Bound text explicitly to avoid overflowing neighboring panels.
        while label and draw.textlength(label, font=small) > tile_w - 20:
            label = label[:-5] + '...'
        draw.text((x0 + 8, y0 + image_h + 31), label, font=small, fill='#d2dbe5')
        state = 'cached' if item['cache_hit'] else 'analyzed this run'
        draw.text((x0 + 8, y0 + image_h + 56), item['image_id'] + ' | ' + state + ' | ' + result.get('status', 'error'), font=small, fill='#91a3b5')
        manifest.append({'panel': index + 1, 'image_id': item['image_id'], 'relative_path': item['relative_path'],
                         'cache_hit': item['cache_hit'], 'selection_features': sorted(features(item))})
    sheet.save(output_path, quality=92)
    return manifest
