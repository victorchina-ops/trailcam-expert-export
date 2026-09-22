"""Directory-to-CSV entry point. Heavy dependencies load only for new images."""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from . import __version__
from .export import FIELDS, make_row
from .fusion import fuse
from .storage import Cache, atomic_json, configuration, file_hash, write_csv

ROOT = Path(__file__).resolve().parents[1]
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def discover(input_dir, excluded=(), recursive=True):
    excluded = [Path(p).resolve() for p in excluded]
    result = []
    for current, dirs, files in os.walk(input_dir):
        current = Path(current)
        dirs[:] = sorted(d for d in dirs if not d.startswith('.') and not (current / d).is_symlink()
                          and not any((current / d).resolve().is_relative_to(e) for e in excluded)) if recursive else []
        for name in sorted(files):
            path = current / name
            if path.suffix.lower() in EXTENSIONS and not path.is_symlink():
                result.append(path)
    return sorted(result, key=lambda p: p.relative_to(input_dir).as_posix().casefold())


def arguments(argv=None):
    settings = {'input_dir': 'images', 'output_dir': 'exports', 'device': 'auto', 'threads': 8,
                'recursive': True, 'contact_sheet': True, 'contact_sheet_size': 20}
    settings_path = ROOT / 'settings.json'
    if settings_path.exists():
        settings.update(json.loads(settings_path.read_text(encoding='utf-8-sig')))
    parser = argparse.ArgumentParser(description='Analyze local trail-camera photos with small vision models; no LLM needed.')
    parser.add_argument('directory', nargs='?', help='Input directory (or use --input)')
    parser.add_argument('--input', help='Input directory; default is images/')
    parser.add_argument('--output', default=settings['output_dir'], help='Timestamped export directory')
    parser.add_argument('--models', default='models', help='Cached model directory')
    parser.add_argument('--cache', default='.cache', help='Incremental inference cache directory')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default=settings['device'])
    parser.add_argument('--threads', type=int, default=settings['threads'])
    parser.add_argument('--contact-sheet', action=argparse.BooleanOptionalAction, default=settings['contact_sheet'])
    parser.add_argument('--contact-sheet-size', type=int, default=settings['contact_sheet_size'])
    parser.add_argument('--recursive', action=argparse.BooleanOptionalAction, default=settings['recursive'])
    parser.add_argument('--force', action='store_true', help='Reanalyze images instead of reusing cache')
    parser.add_argument('--offline', action='store_true', help='Require previously downloaded model files')
    parser.add_argument('--export-new-only', action='store_true', help='CSV includes only images analyzed in this run')
    parser.add_argument('--version', action='version', version=__version__)
    args = parser.parse_args(argv)
    if args.input and args.directory:
        parser.error('Use either a positional input directory or --input, not both.')
    if args.threads < 1 or not 1 <= args.contact_sheet_size <= 100:
        parser.error('--threads must be positive; --contact-sheet-size must be 1..100.')
    args.input = args.input or args.directory or settings['input_dir']
    for name in ('input', 'output', 'models', 'cache'):
        value = Path(getattr(args, name)).expanduser()
        setattr(args, name, (value if value.is_absolute() else ROOT / value).resolve())
    return args


def run(args):
    began = time.perf_counter()
    if not args.input.is_dir():
        raise ValueError(f'Input folder does not exist: {args.input}')
    if args.output == args.input or args.cache == args.input or args.models == args.input:
        raise ValueError('Input folder must differ from output, model, and cache folders.')
    args.output.mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)
    paths = discover(args.input, (args.output, args.cache, args.models, ROOT / '.venv'), args.recursive)
    print(f'Found {len(paths)} images in {args.input}', flush=True)
    run_id = datetime.now().astimezone().strftime('%Y-%m-%d_%H-%M-%S_%f')
    stem = 'export_' + run_id
    lock = args.cache / 'run.lock'
    try:
        lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError(f'Another run may be active. If a prior run crashed, close it and remove {lock}.')
    with os.fdopen(lock_fd, 'w') as f:
        f.write(f'{os.getpid()}\n{run_id}\n')
    vision = attributes = None
    items, rows = [], []
    startup_seconds = 0.0
    try:
        if paths:
            from .model_setup import ensure_models
            ensure_models(args.models, offline=args.offline)
        fingerprint, config = configuration(args.models, args.device, args.threads)
        cache = Cache(args.cache, fingerprint)
        for index, path in enumerate(paths, 1):
            image_start = time.perf_counter()
            relative = path.relative_to(args.input).as_posix()
            sha, hit = '', False
            try:
                sha = file_hash(path)
                result = None if args.force else cache.get(sha)
                hit = result is not None
                if result is None:
                    if vision is None:
                        start = time.perf_counter()
                        from .vision import VisionEngine
                        from .attributes import AttributeEngine
                        new_vision = VisionEngine(args.models, args.device, args.threads)
                        new_attributes = AttributeEngine(args.models, args.device, args.threads)
                        vision, attributes = new_vision, new_attributes
                        startup_seconds += time.perf_counter() - start
                    start = time.perf_counter()
                    result = vision.analyze(path)
                    attr = attributes.analyze(path, result['persons'])
                    by_id = {person['person_id']: person['attributes'] for person in attr['persons']}
                    for person in result['persons']:
                        person['attributes'] = by_id.get(person['person_id'], {'status': 'error', 'error': 'Missing attribute output'})
                    errors = [p['attributes'].get('error') for p in result['persons'] if p['attributes'].get('status') != 'ok']
                    result.update(attribute_device=attr['device'],
                                  status='partial_error' if errors else 'ok',
                                  error='; '.join(str(e) for e in errors) if errors else None,
                                  analyzed_at_utc=datetime.now(timezone.utc).isoformat())
                    result['attribute_runtime'] = {key: value for key, value in attr.items() if key != 'persons'}
                    result.setdefault('timings', {})['attributes_seconds'] = attr['seconds']
                    result['analysis_seconds'] = round(time.perf_counter() - start, 6)
                    fuse(result)
                    cache.put(sha, result)
            except Exception as exc:
                result = {'status': 'error', 'error': f'{type(exc).__name__}: {exc}',
                          'analyzed_at_utc': datetime.now(timezone.utc).isoformat()}
                print(f'  ERROR {relative}: {result["error"]}', file=sys.stderr, flush=True)
            image_id = 'img_' + (sha[:16] if sha else str(index).zfill(6))
            item = {'path': path, 'relative_path': relative, 'image_id': image_id,
                    'cache_hit': hit, 'result': result}
            items.append(item)
            if not args.export_new_only or not hit:
                rows.append(make_row(result, {'run_id': run_id, 'image_id': image_id, 'relative_path': relative,
                                              'sha256': sha, 'cache_hit': hit, 'configuration_hash': fingerprint,
                                              'current_run_seconds': round(time.perf_counter() - image_start, 6)}))
            print(f'[{index}/{len(paths)}] {"cached" if hit else result["status"]}: {relative}', flush=True)
        csv_path = args.output / (stem + '.csv')
        write_csv(csv_path, rows, FIELDS)
        selection, contact_error, contact_path = [], None, None
        if args.contact_sheet and items:
            try:
                from .contacts import make_contact_sheet
                contact_path = args.output / (stem + '_contactsheet.jpg')
                selection = make_contact_sheet(items, contact_path, run_id, args.contact_sheet_size)
            except Exception as exc:
                contact_error = f'{type(exc).__name__}: {exc}'
                print('Contact sheet failed; CSV is saved: ' + contact_error, file=sys.stderr)
        errors = sum(item['result'].get('status') != 'ok' for item in items)
        summary = {'run_id': run_id, 'version': __version__, 'input_dir': str(args.input),
                   'csv': csv_path.name, 'image_count': len(items), 'csv_rows': len(rows),
                   'cached_images': sum(item['cache_hit'] for item in items),
                   'analyzed_images': sum(not item['cache_hit'] for item in items),
                   'images_with_errors': errors, 'model_load_seconds': round(startup_seconds, 6),
                   'total_seconds': round(time.perf_counter() - began, 6),
                   'configuration_hash': fingerprint, 'configuration': config,
                   'contact_sheet': contact_path.name if contact_path and not contact_error else None,
                   'contact_sheet_error': contact_error, 'contact_sheet_selection': selection}
        atomic_json(args.output / (stem + '_run.json'), summary)
        print(f'CSV: {csv_path}\nAnalyzed: {summary["analyzed_images"]}; cached: {summary["cached_images"]}; '
              f'errors: {errors}; elapsed: {summary["total_seconds"]:.1f}s', flush=True)
        if selection:
            print(f'Contact sheet: {contact_path} ({len(selection)} images)', flush=True)
        return 1 if errors or contact_error else 0
    finally:
        lock.unlink(missing_ok=True)


def main(argv=None):
    try:
        return run(arguments(argv))
    except (ValueError, OSError, RuntimeError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
