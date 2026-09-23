"""Cross-image post-processing: cameras, age geometry, events and direction.

Runs on every export over the whole current inventory (cached and new
images), after per-image ``fusion.fuse``. It never loads models. Results are
written into each image result (``combined`` and ``post``) and returned as a
list of event summaries plus per-camera calibration diagnostics.

Images are keyed by relative path, never by content hash: byte-identical
copies in two folders are separate appearances with separate events.
"""
from __future__ import annotations

import copy
from collections import Counter, defaultdict

from .fusion import TRAVEL_DIRECTIONS, fuse

DEFAULTS = {'camera_id_mode': 'folder', 'camera_id_pattern': None, 'event_gap_seconds': 60.0,
            'near_fraction': 0.0, 'events': False, 'geometry_age': False, 'large_bags': False}
# Near zone: people whose expected standing height at their position (camera
# calibration, else their own measured height) is at least this fraction of
# the image height, i.e. people passing the camera rather than a distant
# background crowd. This experimental filter is off (0) by default.
# Detections kept after fusion for the CSV evidence and the contact sheet;
# everything below is only needed inside ``fuse`` and is dropped to bound
# memory on cards with thousands of photos.
KEEP_CONFIDENCE = 0.25


def _camera_key(item):
    """Calibration group: camera id, camera model and image size.

    Different camera models (or resolutions) in one folder never share a
    height calibration. A camera moved or re-aimed between visits should get
    its own folder or ``--camera-pattern`` group.
    """
    result = item['result']
    info = result.get('metadata') or {}
    model = ' '.join(str(info.get(k) or '').strip() for k in ('camera_make', 'camera_model')).strip() or 'unknown'
    return f"{item['camera_id']}|{model}@{result['width']}x{result['height']}"


def compact(result):
    """Drop inference detail that nothing after ``fuse`` reads."""
    persons = result.get('persons', [])
    result['candidate_counts'] = {'total': len(persons), 'rejected': sum(not p.get('accepted') for p in persons)}
    result['persons'] = [p for p in persons if p.get('accepted')]
    for person in result['persons']:
        person.pop('mask_polygon_xy', None)
        (person.get('orientation_evidence') or {}).pop('features', None)
    for name, expert in result.get('experts', {}).items():
        kept = []
        for d in expert.get('detections', []):
            if d.get('confidence', 0) < KEEP_CONFIDENCE:
                continue
            if name != 'yoloe':
                d.pop('mask_polygon_xy', None)
            if d.get('orientation_evidence'):
                d['orientation_evidence'].pop('features', None)
            kept.append(d)
        expert['detections'] = kept
        expert['detections_kept_from_confidence'] = KEEP_CONFIDENCE


def _fail(item, stage, exc):
    result = item['result']
    result['postprocess_error'] = f'{stage}: {type(exc).__name__}: {exc}'
    result['status'] = 'error'
    result['error'] = 'Post-processing failed: ' + result['postprocess_error']


def prepare(item, settings=None):
    """Per-image stage: fuse, assign the camera id and drop inference detail.

    Called right after each image is analysed or read from the cache, so a run
    never holds thousands of raw inference records. Idempotent.
    """
    from .imageinfo import camera_id_for
    if item.get('prepared'):
        return True
    if item['result'].get('status') not in ('ok', 'partial_error'):
        item['prepared'] = False
        return False
    settings = {**DEFAULTS, **(settings or {})}
    try:
        fuse(item['result'], large_bags=settings['large_bags'])
        item['camera_id'] = camera_id_for(item['relative_path'], settings['camera_id_mode'],
                                          settings['camera_id_pattern'])
        compact(item['result'])
    except Exception as exc:  # A damaged record must not stop the export.
        _fail(item, 'fusion', exc)
        result = item['result']
        result['persons'] = []
        for expert in (result.get('experts') or {}).values():
            if isinstance(expert, dict):
                expert['detections'] = []
        item['prepared'] = False
        return False
    item['prepared'] = True
    return True


def postprocess(items, settings=None):
    from .age_geometry import classify, expected_height, fit_camera, person_geometry
    from .events import associate, group_events, summarize_event, track_direction
    settings = {**DEFAULTS, **(settings or {})}
    usable = [item for item in items if prepare(item, settings)]
    from .imageinfo import camera_id_for
    for item in usable:  # cheap; keeps ids consistent with these settings
        item['camera_id'] = camera_id_for(item['relative_path'], settings['camera_id_mode'],
                                          settings['camera_id_pattern'])

    # Age: one height calibration per camera group (see _camera_key).
    by_camera = defaultdict(list)
    for item in list(usable):
        result = item['result']
        strip = result.get('data_strip') or {}
        try:
            for person in result['persons']:
                person['geometry'] = person_geometry({
                    'person_id': person['person_id'], 'image_id': item['relative_path'],
                    'camera_id': item['camera_id'], 'box': person['xyxy'], 'keypoints': person.get('keypoints'),
                    'image_width': result['width'], 'image_height': result['height'],
                    'strip_top': strip.get('top', 0), 'strip_bottom': strip.get('bottom', 0)})
        except Exception as exc:
            _fail(item, 'geometry', exc)
            usable.remove(item)
            continue
        by_camera[_camera_key(item)].extend(p['geometry'] for p in result['persons'])
    calibrations = ({key: fit_camera(geoms) for key, geoms in sorted(by_camera.items())}
                    if settings['geometry_age'] or settings['near_fraction'] > 0 else {})
    for item in usable:
        result = item['result']
        calibration = calibrations.get(_camera_key(item))
        people = result['persons']
        methods = Counter()
        vlm = result.get('vlm_age') or {}
        vlm_people = ((vlm.get('persons') or {}).get('ages')) or {}
        for person in people:
            peers = [p['geometry'] for p in people if p is not person]
            decision = (classify(person['geometry'], calibration, peers) if settings['geometry_age'] else
                        {'age': 'unknown', 'method': 'disabled', 'reasons': ['geometry_age_disabled']})
            person['age_evidence'] = decision
            person['combined'].update(age=decision['age'], age_method=decision['method'])
            answer = vlm_people.get(person['person_id'])
            if answer in ('adult', 'child', 'teen'):
                # A vision-language answer replaces geometry; teens are not children.
                person['combined'].update(age={'teen': 'unknown'}.get(answer, answer), age_method='vlm',
                                          vlm_age=answer, geometry_age=decision['age'])
            if person['combined']['age'] != 'unknown':
                methods[person['combined']['age_method']] += 1
        near_limit = float(settings['near_fraction'] or 0) * result['height']
        depth_window = 0.02 * result['height']

        def own(person):
            geometry, box = person['geometry'] or {}, person['xyxy']
            return geometry.get('height_px') or (box[3] - box[1]), geometry.get('foot_y', box[3])

        for person in people:
            reference = expected_height(person['geometry'] or {}, calibration)
            source = 'camera_calibration'
            if reference is None:
                height, foot = own(person)
                # Heights of people at the same depth or farther (higher foot row): a
                # person closer to the camera than someone near is near too.
                peers = [own(p)[0] for p in people if own(p)[1] <= foot + depth_window]
                reference, source = max([height, *peers]), 'height_at_similar_depth'
            person['combined'].update(near=bool(reference >= near_limit), near_reference_px=round(reference, 1),
                                      near_source=source)
        ages = Counter(p['combined']['age'] for p in people)
        result['combined'].update(adults=ages['adult'], children=ages['child'], age_unknown=ages['unknown'],
                                  age_status=('vlm_estimate' if methods['vlm'] else
                                              'geometry_estimate' if settings['geometry_age'] else 'unknown'),
                                  age_by_camera_calibration=methods['camera_calibration'],
                                  age_by_relative_height=methods['relative_height'],
                                  people_near=sum(p['combined']['near'] for p in people),
                                  age_by_vlm=methods['vlm'] if vlm_people or (vlm.get('persons') or {}).get('ages') == {} else None,
                                  vlm_age_counts=(vlm.get('photo') or {}).get('counts'),
                                  age_model=vlm.get('model'))
        result['post'] = {'camera_id': item['camera_id'],
                          'camera_calibration_status': (calibration or {}).get('status', 'disabled'),
                          'events_enabled': settings['events'], 'geometry_age_enabled': settings['geometry_age'],
                          'large_bags_enabled': settings['large_bags']}

    for key, calibration in calibrations.items():
        group = [i for i in usable if _camera_key(i) == key]
        people_total = sum(i['result']['combined']['counts']['people_total'] for i in group)
        people_near = sum(i['result']['combined'].get('people_near', 0) for i in group)
        calibration['near_zone'] = {
            'near_fraction': settings['near_fraction'], 'people_total': people_total, 'people_near': people_near,
            'status': ('check_near_fraction' if people_total >= 10 and people_near == 0 else 'ok')}

    events = []
    if settings['events']:
        # Events and cross-frame association.
        records, by_key = [], {}
        for item in usable:
            result = item['result']
            info = result.get('metadata') or {}
            people = []
            for person in result['persons']:
                geometry = person.get('geometry') or {}
                box = person['xyxy']
                foot_x = geometry.get('foot_x')
                foot_y = geometry.get('foot_y')
                people.append({'person_id': person['person_id'], 'box': box,
                               'foot': [(box[0] + box[2]) / 2 if foot_x is None else foot_x,
                                        box[3] if foot_y is None else foot_y],
                               'height_px': geometry.get('height_px') or (box[3] - box[1]),
                               'appearance': person.get('appearance'), 'age': person['combined']['age'],
                               'facing': person['combined']['facing'], 'large_bag': person['combined']['large_bag']})
            counts = result['combined']['counts']
            key = item['relative_path']
            records.append({'image_id': key, 'camera_id': item['camera_id'],
                            'capture_time': info.get('capture_time'), 'sequence_number': info.get('sequence_number'),
                            'time_source': info.get('time_source'), 'relative_path': key,
                            'width': result['width'], 'height': result['height'], 'people': people,
                            'camera_make': info.get('camera_make'), 'camera_model': info.get('camera_model'),
                            'objects': {**{f: counts.get(f) for f in ('bicycles', 'strollers', 'motorcycles', 'atv_utv',
                                                                       'other_vehicles', 'dogs', 'backpacks')},
                                        'large_bags': result['combined']['large_bags'],
                                        'large_bags_uncertain': result['combined']['large_bags_uncertain']}})
            by_key[key] = item
        for group in group_events(records, settings['event_gap_seconds']):
            tracks = associate(group)
            summary = summarize_event(group, tracks)
            if not settings['large_bags']:
                summary.update(large_bags=None, large_bags_uncertain=None)
            if len(group) > 1 and any(r.get('time_source') in (None, 'file_mtime') for r in group):
                # Copied files share modification times; such events may merge visits.
                summary['review_reasons'] = sorted(set(summary.get('review_reasons') or []) | {'times_from_file_dates'})
                summary['needs_review'] = True
            events.append(summary)
            persons = {(r['image_id'], p['person_id']): p for r in group
                       for p in by_key[r['image_id']]['result']['persons']}
            summary['people_near_max_frame'] = max(
                (by_key[r['image_id']]['result']['combined'].get('people_near', 0) for r in group), default=0)
            photo_counts = [by_key[r['image_id']]['result']['combined'].get('vlm_age_counts') for r in group]
            photo_counts = [c for c in photo_counts if c]
            for key in ('adults', 'teens', 'children', 'unclear'):
                summary[f'vlm_{key}_max_frame'] = max((c[key] for c in photo_counts), default=None)
            summary['people_near_unique'] = sum(
                any((persons.get(tuple(o)) or {}).get('combined', {}).get('near') for o in track['observations'])
                for track in tracks)
            for track in tracks:
                direction = track.get('direction') or track_direction(track, group)
                for image_id, person_id in track['observations']:
                    person = persons.get((image_id, person_id))
                    if person is None:
                        continue
                    person['track_id'] = track['track_id']
                    person['track_ambiguous'] = bool(track.get('ambiguous'))
                    person['track_review_reasons'] = track.get('reasons', [])
                    person['motion'] = direction
                    if direction['source'] == 'motion' and not track.get('ambiguous'):
                        person['combined'].update(direction=direction['direction'], direction_source='motion')
            for frame, record in enumerate(group, 1):
                result = by_key[record['image_id']]['result']
                result['post'].update(event_id=summary['event_id'], event_frame=frame, event_image_count=len(group))
                result['needs_review'] = bool(result.get('needs_review') or summary['needs_review'])
    for item in usable:
        result = item['result']
        people = result['persons']
        for person in people:
            person.pop('appearance', None)  # only needed for association
            if person['combined'].get('direction_source') == 'pending':
                facing = person['combined']['facing']
                person['combined'].update(direction=facing, direction_source='facing' if facing != 'unclear' else 'none')
        directions = Counter(p['combined']['direction'] for p in people)
        result['combined']['direction_counts'] = {d: directions[d] for d in TRAVEL_DIRECTIONS}
        result['combined']['direction_from_motion'] = sum(p['combined']['direction_source'] == 'motion' for p in people)
        result['combined']['direction_from_facing'] = sum(p['combined']['direction_source'] == 'facing' for p in people)
    return {'events': events, 'calibrations': calibrations, 'settings': settings}


def postprocess_copy(items, settings=None):
    """Post-process deep copies (used by evaluation and tuning scripts)."""
    copies = [{**item, 'result': copy.deepcopy(item['result'])} for item in items]
    return copies, postprocess(copies, settings)
