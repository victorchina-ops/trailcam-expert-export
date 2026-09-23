"""Directory-to-CSV entry point. Heavy dependencies load only for new images."""
from __future__ import annotations
import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from . import __version__
from .export import EVENT_FIELDS, FIELDS, SUMMARY_FIELDS, make_event_row, make_row, summary_rows
from .postprocess import postprocess, prepare
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
                'recursive': True, 'contact_sheet': True, 'contact_sheet_size': 20,
                'profile': 'standard', 'empty_frame_gate': False, 'camera_id_mode': 'folder',
                'camera_id_pattern': None, 'event_gap_seconds': 60, 'near_fraction': 0.0,
                'events': False, 'geometry_age': False, 'large_bags': False,
                'age_model': None, 'age_mode': 'photo', 'ollama_host': 'http://localhost:11434'}
    settings_path = ROOT / 'settings.json'
    if settings_path.exists():
        loaded = json.loads(settings_path.read_text(encoding='utf-8-sig'))
        if not isinstance(loaded, dict):
            raise ValueError(f'{settings_path} must contain a JSON object of settings.')
        settings.update(loaded)
    parser = argparse.ArgumentParser(description='Analyze local trail-camera photos with small vision models; no LLM needed.')
    for name in ('input_dir', 'output_dir'):
        if not isinstance(settings[name], str) or not settings[name]:
            parser.error(f'settings.json: {name} must be a folder path string.')
    if settings['camera_id_pattern'] is not None and not isinstance(settings['camera_id_pattern'], str):
        parser.error('settings.json: camera_id_pattern must be a string or null.')
    parser.add_argument('directory', nargs='?', help='Input directory (or use --input)')
    parser.add_argument('--input', help='Input directory; default is images/')
    parser.add_argument('--output', default=settings['output_dir'], help='Parent directory for dated per-run export folders')
    parser.add_argument('--models', default='models', help='Cached model directory')
    parser.add_argument('--cache', default='.cache', help='Incremental inference cache directory')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default=settings['device'])
    parser.add_argument('--threads', type=int, default=settings['threads'])
    parser.add_argument('--profile', choices=('standard', 'fast'), default=settings['profile'],
                        help='standard: 1280 px people/pose (default); fast: 960 px for slow CPUs')
    parser.add_argument('--empty-frame-gate', action=argparse.BooleanOptionalAction, default=settings['empty_frame_gate'],
                        help='Opt-in speed shortcut: skip heavier experts when fast detectors see nothing; may miss objects')
    parser.add_argument('--camera-id', dest='camera_id_mode', choices=('folder', 'regex', 'single'),
                        default=settings['camera_id_mode'], help='How images are grouped into cameras (default: parent folder)')
    parser.add_argument('--camera-pattern', default=settings['camera_id_pattern'],
                        help='Regex with a named group "camera" applied to the relative path (with --camera-id regex)')
    for flag, description in (('events', 'Experimental cross-image tracking and extra event CSVs'),
                              ('geometry_age', 'Experimental age estimates from relative height'),
                              ('large_bags', 'Experimental bag-size estimates')):
        parser.add_argument('--' + flag.replace('_', '-'), action=argparse.BooleanOptionalAction,
                            default=settings[flag], help=description + ' (default off)')
    parser.add_argument('--event-gap', type=float, default=settings['event_gap_seconds'],
                        help='Seconds between photos that start a new event (default 60)')
    parser.add_argument('--near-fraction', type=float, default=settings['near_fraction'],
                        help='Near zone for people_near: expected standing height at the person\'s position '
                             '>= this fraction of image height (default 0 counts everyone; positive values enable an experimental near zone)')
    parser.add_argument('--age-model', default=settings['age_model'],
                        help='OPTIONAL local Ollama vision model for adult/child (e.g. gemma4:26b; needs a GPU). '
                             'Off by default; --geometry-age separately enables experimental height estimates.')
    parser.add_argument('--age-mode', choices=('photo', 'mosaic', 'photo+mosaic'), default=settings['age_mode'],
                        help='photo: age-group counts per photo (most accurate); mosaic: an age per detected person')
    parser.add_argument('--ollama-host', default=settings['ollama_host'])
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
    # settings.json values become argparse defaults, which skip ``choices``.
    for name, allowed in (('device', ('auto', 'cpu', 'cuda')), ('profile', ('standard', 'fast')),
                          ('camera_id_mode', ('folder', 'regex', 'single')),
                          ('age_mode', ('photo', 'mosaic', 'photo+mosaic'))):
        if getattr(args, name) not in allowed:
            parser.error(f'Invalid {name} {getattr(args, name)!r}; choose one of: {", ".join(allowed)}.')
    for name in ('threads', 'contact_sheet_size'):
        if type(getattr(args, name)) is not int:
            parser.error(f'{name} must be an integer.')
    for name in ('recursive', 'contact_sheet', 'empty_frame_gate', 'events', 'geometry_age', 'large_bags'):
        if type(getattr(args, name)) is not bool:
            parser.error(f'{name} must be true or false.')
    try:
        args.event_gap = float(args.event_gap)
    except (TypeError, ValueError):
        parser.error('--event-gap must be a number of seconds.')
    if not math.isfinite(args.event_gap) or args.event_gap <= 0:
        parser.error('--event-gap must be a finite positive number of seconds.')
    try:
        args.near_fraction = float(args.near_fraction)
    except (TypeError, ValueError):
        parser.error('--near-fraction must be a number.')
    if not math.isfinite(args.near_fraction) or not 0 <= args.near_fraction < 1:
        parser.error('--near-fraction must be between 0 and 1.')
    if args.camera_id_mode == 'regex':
        import re
        try:
            if 'camera' not in re.compile(args.camera_pattern or '').groupindex:
                parser.error('--camera-pattern needs a named group (?P<camera>...).')
        except re.error as exc:
            parser.error(f'Invalid --camera-pattern: {exc}')
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
        fingerprint, config = configuration(args.models, args.device, args.threads, args.profile, args.empty_frame_gate)
        cache = Cache(args.cache, fingerprint)
        post_settings = {'camera_id_mode': args.camera_id_mode, 'camera_id_pattern': args.camera_pattern,
                         'event_gap_seconds': args.event_gap, 'near_fraction': args.near_fraction,
                         'events': args.events, 'geometry_age': args.geometry_age, 'large_bags': args.large_bags}
        age_model = start_age_model(args)
        for index, path in enumerate(paths, 1):
            image_start = time.perf_counter()
            relative = path.relative_to(args.input).as_posix()
            sha, hit = '', False
            try:
                sha = file_hash(path)
                result = None if args.force else cache.get(sha)
                hit = result is not None
                if hit:
                    result['metadata'] = current_metadata(path, result.get('metadata'))
                if result is None:
                    if vision is None:
                        start = time.perf_counter()
                        from .vision import VisionEngine
                        from .attributes import AttributeEngine
                        new_vision = VisionEngine(args.models, args.device, args.threads, args.profile, args.empty_frame_gate)
                        new_attributes = AttributeEngine(args.models, args.device, args.threads)
                        vision, attributes = new_vision, new_attributes
                        startup_seconds += time.perf_counter() - start
                    start = time.perf_counter()
                    result = vision.analyze(path)
                    decoded = result.pop('_decoded_image', None)
                    attr = attributes.analyze(path, result['persons'], decoded, result.get('analysis_scale', 1.0))
                    del decoded
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
                    cache.put(sha, result)
            except Exception as exc:
                result = {'status': 'error', 'error': f'{type(exc).__name__}: {exc}',
                          'analyzed_at_utc': datetime.now(timezone.utc).isoformat()}
                print(f'  ERROR {relative}: {result["error"]}', file=sys.stderr, flush=True)
            image_id = 'img_' + (sha[:16] if sha else str(index).zfill(6))
            item = {'path': path, 'relative_path': relative, 'image_id': image_id, 'sha256': sha,
                    'cache_hit': hit, 'result': result}
            prepare(item, post_settings)  # fuse + compact now; keeps memory bounded
            if age_model and item.get('prepared'):
                add_vlm_age(age_model, args, item, path, fingerprint)
            item['current_run_seconds'] = round(time.perf_counter() - image_start, 6)
            items.append(item)
            print(f'[{index}/{len(paths)}] {"cached" if hit else result["status"]}: {relative}', flush=True)
        post = postprocess(items, post_settings)
        for item in items:
            if not args.export_new_only or not item['cache_hit']:
                metadata = {'run_id': run_id, 'image_id': item['image_id'], 'relative_path': item['relative_path'],
                            'sha256': item['sha256'], 'cache_hit': item['cache_hit'], 'configuration_hash': fingerprint,
                            'current_run_seconds': item['current_run_seconds']}
                try:
                    rows.append(make_row(item['result'], metadata))
                except Exception as exc:  # one damaged record must not abort the export
                    item['result'] = {'status': 'error', 'error': f'Export failed: {type(exc).__name__}: {exc}'}
                    rows.append(make_row(item['result'], metadata))
        run_output = args.output / stem
        run_output.mkdir(exist_ok=False)
        csv_path = run_output / (stem + '.csv')
        write_csv(csv_path, rows, FIELDS)
        events_path = summary_path = None
        if args.events:
            events_path = run_output / (stem + '_events.csv')
            write_csv(events_path, [make_event_row(event, run_id) for event in post['events']], EVENT_FIELDS)
            summary_path = run_output / (stem + '_camera_days.csv')
            write_csv(summary_path, summary_rows(post['events'], run_id), SUMMARY_FIELDS)
        selection, contact_error, contact_path = [], None, None
        if args.contact_sheet and items:
            try:
                from .contacts import make_contact_sheet
                contact_path = run_output / (stem + '_contactsheet.jpg')
                selection = make_contact_sheet(items, contact_path, run_id, args.contact_sheet_size)
            except Exception as exc:
                contact_error = f'{type(exc).__name__}: {exc}'
                print('Contact sheet failed; CSV is saved: ' + contact_error, file=sys.stderr)
        errors = sum(item['result'].get('status') != 'ok' for item in items)
        post_errors = sum(1 for item in items if item['result'].get('postprocess_error'))
        summary = {'run_id': run_id, 'version': __version__, 'input_dir': str(args.input),
                   'export_subfolder': run_output.name, 'csv': csv_path.name, 'events_csv': events_path.name if events_path else None,
                   'camera_days_csv': summary_path.name if summary_path else None, 'event_count': len(post['events']),
                   'camera_calibrations': post['calibrations'], 'postprocess_settings': post['settings'],
                   'images_with_postprocess_errors': post_errors,
                   'age_model': args.age_model, 'age_mode': args.age_mode if args.age_model else None,
                   'age_model_failures': age_model['failures'] if age_model else None,
                   'image_count': len(items), 'csv_rows': len(rows),
                   'cached_images': sum(item['cache_hit'] for item in items),
                   'analyzed_images': sum(not item['cache_hit'] for item in items),
                   'images_with_errors': errors, 'model_load_seconds': round(startup_seconds, 6),
                   'total_seconds': round(time.perf_counter() - began, 6),
                   'configuration_hash': fingerprint, 'configuration': config,
                   'contact_sheet': contact_path.name if contact_path and not contact_error else None,
                   'contact_sheet_error': contact_error, 'contact_sheet_selection': selection}
        atomic_json(run_output / (stem + '_run.json'), summary)
        if events_path:
            print(f'Experimental events: {events_path} ({len(post["events"])} events)', flush=True)
        print(f'CSV: {csv_path}\nAnalyzed: {summary["analyzed_images"]}; cached: {summary["cached_images"]}; '
              f'errors: {errors}; elapsed: {summary["total_seconds"]:.1f}s', flush=True)
        if selection:
            print(f'Contact sheet: {contact_path} ({len(selection)} images)', flush=True)
        return 1 if errors or contact_error else 0
    finally:
        lock.unlink(missing_ok=True)


VLM_MAX_CONSECUTIVE_FAILURES = 3


def start_age_model(args):
    """Optional Ollama age expert, or None when off.

    When the server is unreachable the engines are still returned in offline
    mode: cached answers are reused and other photos keep the geometry estimate.
    """
    if not args.age_model:
        return None
    from .vlm_age import OllamaAge
    engines = {mode: OllamaAge(args.age_model, args.ollama_host, mode)
               for mode in args.age_mode.split('+')}
    online = next(iter(engines.values())).available()
    if not online:
        print(f'Age model {args.age_model} is not available at {args.ollama_host}; '
              'using cached answers where present and geometry otherwise.', file=sys.stderr, flush=True)
    return {'engines': engines, 'online': online, 'failures': 0, 'consecutive': 0, 'fingerprint': None}


def _valid_vlm_answer(answer, mode):
    from .vlm_age import AGES, PHOTO_SCHEMA
    if not isinstance(answer, dict) or answer.get('error'):
        return False
    if mode == 'photo':
        counts = answer.get('counts')
        return isinstance(counts, dict) and all(type(counts.get(k)) is int and counts[k] >= 0
                                                for k in PHOTO_SCHEMA['required'])
    ages = answer.get('ages')
    return isinstance(ages, dict) and all(isinstance(k, str) and v in AGES for k, v in ages.items())


def add_vlm_age(state, args, item, path, fingerprint=''):
    """Attach optional VLM age answers to one prepared image. Never raises.

    Answers are cached per model, mode and prompt version, keyed by the photo's
    SHA256, the inference fingerprint and (for per-person answers) the exact
    people asked about, so a re-analysis never attaches ages to other people.
    """
    import hashlib
    from PIL import Image, ImageOps
    from .storage import atomic_json
    from .vlm_age import PROMPT_VERSION
    from .vision import to_rgb8
    result = item['result']
    persons_signature = hashlib.sha256(json.dumps(
        sorted((p['person_id'], [round(v) for v in p['xyxy']]) for p in result['persons'])).encode()).hexdigest()[:16]
    image, answers, errors = None, {}, []
    for mode, engine in state['engines'].items():
        if mode == 'photo' and result.get('gated_empty'):
            answers[mode] = None  # detectors saw nothing: no VLM call for an empty trigger
            continue
        if mode == 'mosaic' and not result['persons']:
            answers[mode] = {'ages': {}, 'error': None} if state['online'] else None
            continue
        key = f"{args.age_model.replace(':', '_').replace('/', '_')}_{mode}_{PROMPT_VERSION}"
        suffix = hashlib.sha256((fingerprint + (persons_signature if mode == 'mosaic' else '')).encode()).hexdigest()[:16]
        cache_file = args.cache / 'vlm_age' / key / f"{item['sha256']}_{suffix}.json"
        if not args.force:
            try:
                cached = json.loads(cache_file.read_text(encoding='utf-8'))
                if _valid_vlm_answer(cached, mode):
                    answers[mode] = cached
                    continue
            except (OSError, ValueError):
                pass
        if not state['online']:
            continue
        try:
            if image is None:
                with Image.open(path) as source:
                    image = to_rgb8(ImageOps.exif_transpose(source))
            answer = engine.count_photo(image) if mode == 'photo' else engine.classify(image, result['persons'])
        except Exception as exc:  # the geometry estimate stands
            answer = {'error': f'{type(exc).__name__}: {exc}'}
        partial = (mode == 'mosaic' and answer.get('error') and isinstance(answer.get('ages'), dict)
                   and answer['ages'] and _valid_vlm_answer({**answer, 'error': None}, mode))
        if _valid_vlm_answer(answer, mode):
            answers[mode] = answer
            state['consecutive'] = 0
            try:
                atomic_json(cache_file, answer)
            except OSError as exc:
                errors.append(f'{mode}: cache write failed: {exc}')
        elif partial:
            answers[mode] = answer  # use the ages we have; not cached, asked again next run
            errors.append(f"{mode}: partial answer: {answer['error']}")
            state['failures'] += 1
        else:
            errors.append(f"{mode}: {answer.get('error') or 'invalid answer'}")
            state['failures'] += 1
            state['consecutive'] += 1
            if state['consecutive'] >= VLM_MAX_CONSECUTIVE_FAILURES:
                state['online'] = False
                print(f'Age model failed {state["consecutive"]} times in a row; continuing with cached answers '
                      f'and geometry only. Last error: {errors[-1]}', file=sys.stderr, flush=True)
    answered = {m: a for m, a in answers.items() if a}
    result['vlm_age'] = {'model': args.age_model if answered else None, 'photo': answers.get('photo'),
                         'persons': answers.get('mosaic'), 'errors': errors or None}


def current_metadata(path, cached=None):
    """Capture metadata read from the file itself (header only, no pixel decode)."""
    try:
        from PIL import Image
        from .imageinfo import read_capture_metadata
        with Image.open(path) as image:
            return read_capture_metadata(image, Path(path))
    except Exception:
        return cached


def main(argv=None):
    # Hebrew or other non-encodable file names must never abort a run when the
    # console or a redirected log uses a legacy Windows code page.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors='backslashreplace')
        except (AttributeError, ValueError, OSError):
            pass
    try:
        return run(arguments(argv))
    except (ValueError, OSError, RuntimeError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
