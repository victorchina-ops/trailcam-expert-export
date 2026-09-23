"""CSV rows: one per image, one per event, one per camera and day.

Image rows count appearances in that photograph. Event rows de-duplicate the
same people across the photographs of one event; use them for visitor totals.
JSON cells retain detailed expert evidence.
"""
import json
from collections import Counter, defaultdict

from .fusion import COUNT_FIELDS, DIRECTIONS, ORIENTATIONS, TRAVEL_DIRECTIONS, thresholded

EVIDENCE_CONFIDENCE = 0.25
# Group-size classes: more robust than exact counts (per photo, v2 near-zone
# counts fall in the manual label's class 79% of the time, within one class 98%).
# Range labels use an en dash: Excel turns "3-4" / "5-10" into dates.
COUNT_CLASSES = ((0, 0, '0'), (1, 1, '1'), (2, 2, '2'), (3, 4, '3\u20134'), (5, 10, '5\u201310'), (11, None, '>10'))


def count_class(value):
    """Class label for a count; blank for unknown (None)."""
    if value is None:
        return None
    for low, high, label in COUNT_CLASSES:
        if value >= low and (high is None or value <= high):
            return label
    return None
CELL_LIMIT = 32000
OBJECT_FIELDS = ('bicycles', 'strollers', 'motorcycles', 'atv_utv', 'other_vehicles', 'dogs', 'backpacks')
MAIN_FIELDS = (['people_total', 'people_near', 'people_class', 'people_near_class', 'adults', 'children', 'age_unknown']
               + [f'dir_{d}' for d in TRAVEL_DIRECTIONS]
               + ['direction_from_motion', 'direction_from_facing']
               + list(OBJECT_FIELDS) + ['large_bags', 'large_bags_uncertain',
                                        'cars_trucks_buses', 'kick_scooters'])
BASE_FIELDS = ['run_id', 'image_id', 'relative_path', 'sha256', 'camera_id', 'capture_time', 'time_source',
               'clock_suspect', 'sequence_number', 'event_id', 'event_frame', 'event_image_count',
               'status', 'error', 'cache_hit', 'analyzed_at_utc', 'width', 'height', 'profile',
               'vision_device', 'attribute_device', 'analysis_seconds', 'current_run_seconds',
               'configuration_hash', 'gated_empty', 'data_strip_bottom', 'needs_review']
FIELDS = BASE_FIELDS + MAIN_FIELDS
FIELDS += [f'unverified_{field}' for field in OBJECT_FIELDS]
FIELDS += ['age_model', 'vlm_adults', 'vlm_teens', 'vlm_children', 'vlm_age_unclear', 'age_by_vlm']
FIELDS += ['candidates_total', 'candidates_rejected', 'age_by_camera_calibration', 'age_by_relative_height',
           'camera_calibration_status']
FIELDS += [f'facing_{d}' for d in DIRECTIONS] + [f'orientation_{o}' for o in ORIENTATIONS]
FIELDS += ['orientation_conflicts'] + [f'carrying_backpack_{b}' for b in ('yes', 'no', 'unknown')]
FIELDS += [f'{prefix}_{field}' for prefix in ('yoloe', 'yolo26n', 'megadetector') for field in COUNT_FIELDS]
FIELDS += [f'{prefix}_{field}' for field in COUNT_FIELDS
           for prefix in ('vote', 'mean', 'vote_status', 'vote_model_count', 'vote_spread')]
FIELDS += ['age_status', 'large_bags_status', 'events_enabled', 'geometry_age_enabled', 'large_bags_enabled']
FIELDS += [f'attribute_age_{a}' for a in ('under18', '18_60', 'over60', 'unknown')]
FIELDS += [f'attribute_orientation_{o}' for o in ORIENTATIONS]
FIELDS += [f'attribute_presentation_proxy_{p}' for p in ('feminine', 'masculine', 'unclear')]
FIELDS += ['pose_people_total'] + [f'pose_direction_{d}' for d in DIRECTIONS]
FIELDS += [f'pose_orientation_{o}' for o in ORIENTATIONS]
FIELDS += [f'combined_source_{field}' for field in COUNT_FIELDS]
FIELDS += ['persons_json', 'objects_json', 'expert_evidence_json', 'attribute_runtime_json', 'timings_json', 'notes']

EVENT_FIELDS = ['run_id', 'event_id', 'camera_id', 'start', 'end', 'duration_seconds', 'image_count',
                'people_unique', 'people_max_frame', 'people_near_max_frame', 'people_near_unique',
                'people_unique_class', 'people_near_unique_class', 'adults', 'children', 'age_unknown',
                'vlm_adults_max_frame', 'vlm_teens_max_frame', 'vlm_children_max_frame', 'vlm_unclear_max_frame']
EVENT_FIELDS += [f'dir_{d}' for d in TRAVEL_DIRECTIONS] + ['direction_from_motion', 'direction_from_facing']
EVENT_FIELDS += list(OBJECT_FIELDS) + ['large_bags', 'large_bags_uncertain', 'needs_review', 'review_reasons',
                                       'images', 'notes']
SUMMARY_FIELDS = ['run_id', 'camera_id', 'date', 'events', 'images', 'people_max_frame', 'people_unique',
                  'people_near_max_frame', 'people_near_unique',
                  'adults', 'children',
                  'age_unknown'] + [f'dir_{d}' for d in TRAVEL_DIRECTIONS] + list(OBJECT_FIELDS) + [
                  'large_bags', 'events_needing_review']

IMAGE_NOTE = ('Counts are appearances in this photograph. Direction is apparent facing unless direction_source '
              'is motion from optional experimental event tracking. Adult/child counts preserve unknown by default; '
              'geometry age and large-bag estimates are experimental opt-ins. Native attribute age and presentation '
              'outputs are unvalidated model predictions; presentation proxy is not gender identity. '
              'people_near equals all people unless an experimental near zone is requested. '
              'Blank means unsupported/disabled/unavailable; zero means no accepted detection or classification.')
EVENT_NOTE = ('Experimental: people_unique estimates matched tracks and may merge or split visitors; '
              'people_max_frame is the largest single-photo count. Review flagged associations. '
              'Object columns are the maximum over event photos; disabled estimates stay blank.')


def json_cell(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def _r(value, digits=2):
    return round(value, digits) if isinstance(value, float) else value


def _compact_person(person):
    """Small per-person record; full evidence stays in the inference cache."""
    combined = person.get('combined') or {}
    age = person.get('age_evidence') or {}
    geometry = person.get('geometry') or {}
    motion = person.get('motion') or {}
    return {'id': person.get('person_id'), 'box': [round(v) for v in person.get('bbox_xyxy', [])],
            'conf': _r(person.get('confidence')),
            'experts': {n: _r(m['confidence']) for n, m in (person.get('members') or {}).items()},
            'age': combined.get('age'), 'age_method': combined.get('age_method'), 'height_ratio': _r(age.get('ratio')),
            'near': combined.get('near'), 'near_source': combined.get('near_source'),
            'height_px': _r(geometry.get('height_px'), 1), 'complete': geometry.get('complete'),
            'direction': combined.get('direction'), 'direction_source': combined.get('direction_source'),
            'lateral': _r(motion.get('lateral')), 'radial': _r(motion.get('radial')),
            'facing': combined.get('facing'), 'orientation': combined.get('orientation'),
            'backpack': combined.get('carrying_backpack'), 'large_bag': combined.get('large_bag'),
            'associated_backpack': combined.get('associated_backpack'),
            'backpack_source': combined.get('backpack_source'),
            'attributes': person.get('attributes', {}),
            'pose': {'orientation': (person.get('orientation_evidence') or {}).get('orientation'),
                     'facing_direction': (person.get('orientation_evidence') or {}).get('facing_direction'),
                     'confidence': person.get('pose_confidence')},
            'track_ambiguous': person.get('track_ambiguous'), 'track_review_reasons': person.get('track_review_reasons'),
            'track': person.get('track_id')}


def make_row(result, metadata):
    row = dict.fromkeys(FIELDS)
    row.update(metadata)
    for key in ('status', 'error', 'analyzed_at_utc', 'width', 'height', 'needs_review', 'profile', 'gated_empty'):
        row[key] = result.get(key)
    row['vision_device'] = result.get('device')
    row['attribute_device'] = result.get('attribute_device')
    row['analysis_seconds'] = result.get('analysis_seconds')
    info = result.get('metadata') or {}
    for key in ('capture_time', 'time_source', 'clock_suspect', 'sequence_number'):
        row[key] = info.get(key)
    row['data_strip_bottom'] = (result.get('data_strip') or {}).get('bottom')
    post = result.get('post') or {}
    for key in ('camera_id', 'event_id', 'event_frame', 'event_image_count', 'camera_calibration_status',
                'events_enabled', 'geometry_age_enabled', 'large_bags_enabled'):
        row[key] = post.get(key)
    if result.get('status') in ('ok', 'partial_error') and 'combined' in result:
        combined = result['combined']
        counts = combined['counts']
        row.update({field: counts.get(field) for field in COUNT_FIELDS})
        for key in ('people_near', 'adults', 'children', 'age_unknown', 'large_bags', 'large_bags_uncertain',
                    'orientation_conflicts', 'direction_from_motion', 'direction_from_facing',
                    'age_by_camera_calibration', 'age_by_relative_height', 'age_status', 'large_bags_status'):
            row[key] = combined.get(key)
        vlm_counts = combined.get('vlm_age_counts') or {}
        row.update(age_model=combined.get('age_model'), age_by_vlm=combined.get('age_by_vlm'),
                   vlm_adults=vlm_counts.get('adults'), vlm_teens=vlm_counts.get('teens'),
                   vlm_children=vlm_counts.get('children'), vlm_age_unclear=vlm_counts.get('unclear'))
        row['people_class'] = count_class(counts.get('people_total'))
        row['people_near_class'] = count_class(combined.get('people_near'))
        if combined.get('direction_counts') is not None:
            row.update({f'dir_{d}': combined['direction_counts'].get(d, 0) for d in TRAVEL_DIRECTIONS})
        row.update({f'unverified_{f}': combined['unverified_counts'].get(f) for f in OBJECT_FIELDS})
        row.update({f'facing_{d}': v for d, v in combined['facing_counts'].items()})
        row.update({f'orientation_{o}': v for o, v in combined['orientation_counts'].items()})
        row.update({f'carrying_backpack_{b}': v for b, v in combined['carrying_backpack_counts'].items()})
        persons = result.get('persons', [])
        attributes = [p.get('attributes') or {} for p in persons if p.get('accepted')]
        for field, prefix, labels in (('native_age_label', 'attribute_age', ('under18', '18_60', 'over60', 'unknown')),
                                      ('native_orientation_label', 'attribute_orientation', ORIENTATIONS)):
            tally = Counter(a.get(field, 'unknown') for a in attributes)
            row.update({f'{prefix}_{label}': tally[label] for label in labels})
        presentation = Counter(a.get('presentation_proxy', 'unclear') for a in attributes)
        for label in ('feminine', 'masculine', 'unclear'):
            row['attribute_presentation_proxy_' + label] = presentation[label + '_presentation_proxy' if label != 'unclear' else label]
        pose = result.get('experts', {}).get('pose', {})
        row['pose_people_total'] = pose.get('counts', {}).get('people_total')
        if not pose.get('skipped'):
            evidence = [d.get('orientation_evidence') or {} for d in thresholded(result).get('pose', []) if d.get('label') == 'person']
            for field, prefix, labels, unknown in (('orientation', 'pose_orientation', ORIENTATIONS, 'unknown'),
                                                  ('facing_direction', 'pose_direction', DIRECTIONS, 'unclear')):
                tally = Counter(e.get(field, unknown) for e in evidence)
                row.update({f'{prefix}_{label}': tally[label] for label in labels})
        row.update({f'combined_source_{field}': combined.get('count_sources', {}).get(field) for field in COUNT_FIELDS})
        candidates = result.get('candidate_counts') or {
            'total': len(persons), 'rejected': sum(not p.get('accepted') for p in persons)}
        row['candidates_total'] = candidates['total']
        row['candidates_rejected'] = candidates['rejected']
        for name in ('yoloe', 'yolo26n', 'megadetector'):
            expert_counts = result['experts'].get(name, {}).get('counts', {})
            row.update({f'{name}_{field}': expert_counts.get(field) for field in COUNT_FIELDS})
        for field, vote in result['votes'].items():
            row.update({f'vote_{field}': vote['value'], f'mean_{field}': vote['mean'],
                        f'vote_status_{field}': vote['status'], f'vote_model_count_{field}': vote['model_count'],
                        f'vote_spread_{field}': vote['spread']})
        row['persons_json'] = json_cell([_compact_person(p) for p in persons if p.get('accepted')])
        row['objects_json'] = json_cell({f: [o for o in found] for f, found in result.get('objects', {}).items() if found})
        # Evidence at the counting threshold keeps typical cells small; very
        # crowded photos can still exceed a spreadsheet's 32,767-character cell
        # limit (read those with Python). The inference cache keeps every
        # raw detection.
        row['expert_evidence_json'] = json_cell({
            name: {**{k: v for k, v in expert.items() if k not in ('detections', 'keypoint_names')},
                   'detections': [[d['label'], round(d['confidence'], 3)] + [round(v, 1) for v in d['xyxy']]
                                  for d in expert.get('detections', []) if d['confidence'] >= EVIDENCE_CONFIDENCE]}
            for name, expert in result['experts'].items()})
        row['attribute_runtime_json'] = json_cell(result.get('attribute_runtime', {}))
        row['timings_json'] = json_cell(result.get('timings', {}))
    row['notes'] = IMAGE_NOTE
    return row


def make_event_row(event, run_id):
    row = dict.fromkeys(EVENT_FIELDS)
    row.update({k: v for k, v in event.items() if k in EVENT_FIELDS})
    row['run_id'] = run_id
    row['people_unique_class'] = count_class(event.get('people_unique'))
    row['people_near_unique_class'] = count_class(event.get('people_near_unique'))
    images = json_cell(event.get('images', []))
    if len(images) > CELL_LIMIT:  # spreadsheet cell limit; very long events keep first/last only
        ids = event.get('images') or []
        images = json_cell({'first': ids[0], 'last': ids[-1], 'count': len(ids), 'truncated': True})
    row['images'] = images
    reasons = event.get('review_reasons')
    row['review_reasons'] = '; '.join(reasons) if isinstance(reasons, (list, tuple)) else reasons
    row['notes'] = EVENT_NOTE
    return row


def summary_rows(events, run_id):
    """Sum de-duplicated event totals per camera and capture date."""
    groups = defaultdict(list)
    for event in events:
        start = event.get('start') or ''
        groups[(event.get('camera_id'), start[:10] if start else 'unknown')].append(event)
    rows = []
    for (camera, date), group in sorted(groups.items(), key=lambda kv: (str(kv[0][0]), kv[0][1])):
        row = {'run_id': run_id, 'camera_id': camera, 'date': date, 'events': len(group),
               'images': sum(e.get('image_count') or 0 for e in group),
               'events_needing_review': sum(bool(e.get('needs_review')) for e in group)}
        for field in ['people_max_frame', 'people_unique', 'people_near_max_frame', 'people_near_unique',
                      'adults', 'children', 'age_unknown', 'large_bags',
                      *OBJECT_FIELDS,
                      *[f'dir_{d}' for d in TRAVEL_DIRECTIONS]]:
            values = [e.get(field) for e in group if e.get(field) is not None]
            row[field] = sum(values) if values else None
        rows.append(row)
    return rows
