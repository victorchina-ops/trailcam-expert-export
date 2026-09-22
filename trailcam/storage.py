"""Content-addressed local cache and atomic exports."""
from __future__ import annotations
import csv
import hashlib
import importlib.metadata
import json
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


def configuration(models_dir, device, threads):
    root = Path(__file__).parent
    model_files = sorted(p for p in Path(models_dir).rglob('*') if p.is_file()
                         and p.suffix in ('.pt', '.json', '.pdiparams', '.yml'))
    # Runtime versions and actual model content invalidate stale predictions.
    packages = {}
    for name in ('torch', 'ultralytics', 'paddlepaddle', 'paddlepaddle-gpu', 'paddlex', 'numpy', 'Pillow'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    value = {'schema_version': 1, 'requested_device': device, 'threads': threads,
             'code': {p.name: file_hash(p) for p in sorted(root.glob('*.py'))},
             'models': {p.relative_to(models_dir).as_posix(): file_hash(p) for p in model_files},
             'packages': packages}
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(encoded).hexdigest(), value


class Cache:
    def __init__(self, root, fingerprint):
        self.root = Path(root) / fingerprint

    def get(self, sha):
        try:
            value = json.loads((self.root / (sha + '.json')).read_text(encoding='utf-8'))
            if not isinstance(value, dict) or value.get('status') != 'ok' or value.get('sha256') != sha:
                return None
            if value.get('_cache_format') != 1:
                return None
            # Valid JSON can still have a damaged or obsolete schema. Recompute
            # instead of failing later while exporting a purported cache hit.
            from .export import FIELDS, make_row
            from .fusion import COUNT_FIELDS
            persons = value['persons']
            if not isinstance(persons, list) or not all(
                    isinstance(person, dict) and isinstance(person['person_id'], str)
                    and len(person['bbox_xyxy']) == 4
                    and person['attributes']['status'] == 'ok' for person in persons):
                return None
            for name in ('yoloe', 'yolo26n', 'megadetector', 'pose'):
                expert = value['experts'][name]
                if not isinstance(expert['counts'], dict) or not isinstance(expert['detections'], list):
                    return None
            combined = value['combined']
            if any(combined['counts'].get(field) is not None and (
                    type(combined['counts'][field]) is not int or combined['counts'][field] < 0)
                   for field in COUNT_FIELDS):
                return None
            if combined['counts']['people_total'] != len(persons):
                return None
            # Reuse the export contract so newly required fields cannot be
            # silently omitted from this cache validator.
            if set(make_row(value, {})) != set(FIELDS):
                return None
            return value
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            return None

    def put(self, sha, result):
        if result.get('status') == 'ok':
            atomic_json(self.root / (sha + '.json'), {**result, 'sha256': sha, '_cache_format': 1})


def spreadsheet_safe(value):
    # Never execute a filename as an Excel/LibreOffice formula.
    if isinstance(value, str) and value.lstrip().startswith(('=', '+', '-', '@', '\t', '\r', '\n')):
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
