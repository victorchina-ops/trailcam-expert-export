"""Content-addressed local cache and atomic exports."""
from __future__ import annotations
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import uuid
from pathlib import Path


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        tmp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding='utf-8')
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


# Only these modules change inference output. Post-processing (fusion,
# acceptance thresholds, age geometry, events, export) is recomputed on every
# run from cached raw expert output, so editing it never forces re-inference.
INFERENCE_MODULES = ('vision.py', 'attributes.py', 'imageinfo.py', 'appearance.py', 'pose_geometry.py')
CACHE_FORMAT = 2


def configuration(models_dir, device, threads, profile='standard', empty_frame_gate=False):
    import inspect
    from . import roster
    root = Path(__file__).parent
    model_files = sorted(p for p in Path(models_dir).rglob('*') if p.is_file()
                         and p.suffix in ('.pt', '.json', '.pdiparams', '.yml'))
    # Runtime versions and actual model content invalidate stale predictions.
    packages = {}
    for name in ('torch', 'ultralytics', 'paddlepaddle', 'paddlepaddle-gpu', 'paddlex', 'numpy', 'Pillow',
                 'opencv-python', 'opencv-python-headless'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    candidate_code = ''.join(inspect.getsource(f) for f in (roster.box_iou, roster.strip_fraction, roster.build_candidates))
    value = {'schema_version': CACHE_FORMAT, 'requested_device': device, 'profile': profile,
             'empty_frame_gate': bool(empty_frame_gate),
             'code': {name: file_hash(root / name) for name in INFERENCE_MODULES if (root / name).is_file()},
             'candidate_code': hashlib.sha256(candidate_code.encode()).hexdigest(),
             'candidate_policy': {**{k: roster.POLICY[k] for k in ('raw_floor', 'match_iou', 'strip_fraction')},
                                  'expert_order': list(roster.EXPERT_ORDER)},
             'models': {p.relative_to(models_dir).as_posix(): file_hash(p) for p in model_files},
             'packages': packages}
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(encoded).hexdigest(), {**value, 'threads': threads}


class Cache:
    """Raw inference results (experts, candidate people, attributes) by SHA256."""

    def __init__(self, root, fingerprint):
        self.root = Path(root) / fingerprint

    def get(self, sha):
        try:
            value = json.loads((self.root / (sha + '.json')).read_text(encoding='utf-8'))
            if not isinstance(value, dict) or value.get('status') != 'ok' or value.get('sha256') != sha:
                return None
            if value.get('_cache_format') != CACHE_FORMAT:
                return None
            # Valid JSON can still have a damaged or obsolete schema; recompute
            # instead of failing later while post-processing a cache hit.
            persons = value['persons']
            if not isinstance(persons, list) or not all(
                    isinstance(person, dict) and isinstance(person['person_id'], str)
                    and _box(person['bbox_xyxy']) and isinstance(person['members'], dict)
                    and person['attributes']['status'] == 'ok' for person in persons):
                return None
            for person in persons:
                if not _box(person['xyxy']) or not all(
                        isinstance(m, dict) and _number(m['confidence']) is not None
                        and isinstance(m['detection_index'], int) for m in person['members'].values()):
                    return None
            for name in ('yoloe', 'yolo26n', 'megadetector', 'pose'):
                detections = value['experts'][name]['detections']
                if not isinstance(detections, list) or not all(
                        isinstance(d, dict) and isinstance(d['label'], str) and _number(d['confidence']) is not None
                        and _box(d['xyxy']) for d in detections):
                    return None
            if not isinstance(value['metadata'], dict) or not isinstance(value['data_strip'], dict):
                return None
            if not isinstance(value['width'], int) or not isinstance(value['height'], int):
                return None
            return value
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            return None

    def put(self, sha, result):
        if result.get('status') == 'ok':
            atomic_json(self.root / (sha + '.json'), {**result, 'sha256': sha, '_cache_format': CACHE_FORMAT})


def _number(value):
    return float(value) if type(value) in (int, float) and math.isfinite(value) else None


def _box(value):
    return isinstance(value, list) and len(value) == 4 and all(_number(v) is not None for v in value) \
        and value[2] >= value[0] and value[3] >= value[1]


def spreadsheet_safe(value):
    # Never execute a filename as an Excel/LibreOffice formula.
    if isinstance(value, str) and (value.startswith(('\t', '\r', '\n'))
                                   or value.lstrip().startswith(('=', '+', '-', '@'))):
        return "'" + value
    return value


def write_csv(path, rows, fields):
    path = Path(path)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with tmp.open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction='raise')
            writer.writeheader()
            writer.writerows({key: spreadsheet_safe(value) for key, value in row.items()} for row in rows)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
