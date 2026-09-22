"""One stable row per input image; JSON cells retain detailed expert evidence."""
import json
from .fusion import COUNT_FIELDS, DIRECTIONS, ORIENTATIONS

BASE_FIELDS = ['run_id', 'image_id', 'relative_path', 'sha256', 'status', 'error', 'cache_hit',
               'analyzed_at_utc', 'width', 'height', 'vision_device', 'attribute_device',
               'analysis_seconds', 'current_run_seconds', 'configuration_hash', 'needs_review']
FIELDS = BASE_FIELDS + [f'{prefix}_{field}' for prefix in ('combined', 'yoloe', 'yolo26n', 'megadetector')
                        for field in COUNT_FIELDS]
FIELDS += [f'{prefix}_{field}' for field in COUNT_FIELDS
           for prefix in ('vote', 'mean', 'vote_status', 'vote_model_count', 'vote_spread', 'combined_source')]
FIELDS += ['combined_adults', 'combined_children', 'combined_age_unknown', 'age_status',
           'presentation_status', 'orientation_conflicts']
FIELDS += [f'combined_direction_{d}' for d in DIRECTIONS]
FIELDS += [f'combined_orientation_{o}' for o in ORIENTATIONS]
FIELDS += [f'combined_carrying_backpack_{b}' for b in ('yes', 'no', 'unknown')]
FIELDS += [f'attribute_age_{a}' for a in ('under18', '18_60', 'over60', 'unknown')]
FIELDS += [f'attribute_orientation_{o}' for o in ORIENTATIONS]
FIELDS += [f'attribute_presentation_proxy_{p}' for p in ('feminine', 'masculine', 'unclear')]
FIELDS += ['pose_people_total'] + [f'pose_direction_{d}' for d in DIRECTIONS]
FIELDS += [f'pose_orientation_{o}' for o in ORIENTATIONS]
FIELDS += ['persons_json', 'expert_evidence_json', 'attribute_runtime_json', 'timings_json', 'notes']


def json_cell(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def make_row(result, metadata):
    row = dict.fromkeys(FIELDS)
    row.update(metadata)
    for key in ('status', 'error', 'analyzed_at_utc', 'width', 'height', 'needs_review'):
        row[key] = result.get(key)
    row['vision_device'] = result.get('device')
    row['attribute_device'] = result.get('attribute_device')
    row['analysis_seconds'] = result.get('analysis_seconds')
    if result.get('status') in ('ok', 'partial_error'):
        combined = result['combined']
        for name in ('yoloe', 'yolo26n', 'megadetector'):
            counts = result['experts'].get(name, {}).get('counts', {})
            row.update({f'{name}_{field}': counts.get(field) for field in COUNT_FIELDS})
        row.update({f'combined_{field}': combined['counts'].get(field) for field in COUNT_FIELDS})
        for field, vote in result['votes'].items():
            row.update({f'vote_{field}': vote['value'], f'mean_{field}': vote['mean'],
                        f'vote_status_{field}': vote['status'], f'vote_model_count_{field}': vote['model_count'],
                        f'vote_spread_{field}': vote['spread'],
                        f'combined_source_{field}': combined['count_sources'][field]})
        for key in ('adults', 'children', 'age_unknown'):
            row['combined_' + key] = combined[key]
        for key in ('age_status', 'presentation_status', 'orientation_conflicts'):
            row[key] = combined[key]
        for group, prefix in [('direction_counts', 'combined_direction'),
                              ('orientation_counts', 'combined_orientation'),
                              ('carrying_backpack_counts', 'combined_carrying_backpack')]:
            row.update({prefix + '_' + key: value for key, value in combined[group].items()})
        for group, prefix in [('age_counts', 'attribute_age'), ('orientation_counts', 'attribute_orientation'),
                              ('presentation_proxy_counts', 'attribute_presentation_proxy')]:
            row.update({prefix + '_' + key: value for key, value in result['attribute_summary'][group].items()})
        pose = result['experts'].get('pose', {})
        row['pose_people_total'] = pose.get('counts', {}).get('people_total')
        for group, prefix in [('direction_counts', 'pose_direction'), ('orientation_counts', 'pose_orientation')]:
            row.update({prefix + '_' + key: value for key, value in pose.get(group, {}).items()})
        row['persons_json'] = json_cell(result['persons'])
        row['expert_evidence_json'] = json_cell(result['experts'])
        row['attribute_runtime_json'] = json_cell(result.get('attribute_runtime', {}))
        row['timings_json'] = json_cell(result.get('timings', {}))
    row['notes'] = ('Direction is apparent facing, not measured travel. Adult/child mixture abstains; '
                    'native attribute ages are unvalidated predictions. Presentation proxy is not gender identity. '
                    'Blank means unsupported/unavailable; zero means no accepted detection. '
                    'People are appearances per image, not unique visitors.')
    return row
