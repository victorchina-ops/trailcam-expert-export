"""Transparent, uncalibrated expert combination; unknown is not zero."""
from collections import Counter
from statistics import mean

COUNT_FIELDS = ('people_total', 'bicycles', 'strollers', 'motorcycles', 'atv_utv',
                'other_vehicles', 'dogs', 'backpacks', 'cars_trucks_buses', 'kick_scooters')
DIRECTIONS = ('left', 'right', 'toward', 'away', 'unclear')
ORIENTATIONS = ('front', 'back', 'side', 'unknown')


def count_vote(values):
    eligible = {k: v for k, v in values.items() if type(v) is int and v >= 0}
    frequencies = Counter(eligible.values())
    winner, support = frequencies.most_common(1)[0] if frequencies else (None, 0)
    majority = support >= 2 and support > len(eligible) / 2
    return {'value': winner if majority else None,
            'mean': round(mean(eligible.values()), 6) if eligible else None,
            'spread': max(eligible.values()) - min(eligible.values()) if eligible else None,
            'support': support if majority else 0, 'model_count': len(eligible),
            'status': 'majority' if majority else 'single_expert' if len(eligible) == 1
                      else 'no_majority' if eligible else 'no_eligible_output',
            'models': list(eligible)}


def orientation_fusion(attribute, pose):
    a = attribute if attribute in ORIENTATIONS[:-1] else 'unknown'
    p = pose if pose in ORIENTATIONS[:-1] else 'unknown'
    if a == p and a != 'unknown':
        return a, 'attribute_and_pose'
    if a != 'unknown' and p != 'unknown':
        return 'unknown', 'conflict_abstention'
    if a != 'unknown':
        return a, 'attribute_only'
    if p != 'unknown':
        return p, 'pose_only'
    return 'unknown', 'no_usable_evidence'


def fuse(result):
    experts, persons = result['experts'], result['persons']
    votes, counts, sources = {}, {}, {}
    for field in COUNT_FIELDS:
        # Generic MegaDetector animals cannot vote specifically for dogs.
        names = ('yolo26n', 'yoloe', 'megadetector') if field == 'people_total' else ('yolo26n', 'yoloe')
        values = {name: experts.get(name, {}).get('counts', {}).get(field) for name in names}
        vote = votes[field] = count_vote(values)
        if field == 'people_total':
            counts[field], sources[field] = len(persons), 'yoloe_canonical_person_roster'
        elif vote['value'] is not None:
            counts[field], sources[field] = vote['value'], 'strict_majority'
        else:
            counts[field], sources[field] = values.get('yoloe'), 'yoloe_provisional_fallback'
    orientation_counts = dict.fromkeys(ORIENTATIONS, 0)
    direction_counts = dict.fromkeys(DIRECTIONS, 0)
    carrying = {'yes': 0, 'no': 0, 'unknown': 0}
    attribute_ages = {'under18': 0, '18_60': 0, 'over60': 0, 'unknown': 0}
    attribute_orientations = dict.fromkeys(ORIENTATIONS, 0)
    attribute_presentation = {'feminine': 0, 'masculine': 0, 'unclear': 0}
    conflicts = 0
    for person in persons:
        attr, pose = person.get('attributes', {}), person.get('pose', {})
        a = attr.get('native_orientation_label', 'unknown')
        p = pose.get('orientation', 'unknown')
        orientation, source = orientation_fusion(a, p)
        direction = {'front': 'toward', 'back': 'away'}.get(orientation, 'unclear')
        if orientation == 'side' and p == 'side' and pose.get('facing_direction') in ('left', 'right'):
            direction = pose['facing_direction']
        backpack = attr.get('backpack_presence')
        mask = person.get('mask_backpack') is True
        if mask and backpack is False:
            backpack, bag_source = None, 'attribute_mask_conflict'
        elif mask and backpack is True:
            bag_source = 'attribute_and_spatial_support'
        elif mask:
            backpack, bag_source = None, 'spatial_support_only_unverified'
        else:
            bag_source = 'attribute_only' if type(backpack) is bool else 'unknown'
        person['combined'] = {'orientation': orientation, 'orientation_source': source,
                              'direction': direction, 'carrying_backpack': backpack,
                              'backpack_source': bag_source, 'age': 'unknown',
                              'presentation': 'unclear'}
        orientation_counts[orientation] += 1
        direction_counts[direction] += 1
        carrying['yes' if backpack is True else 'no' if backpack is False else 'unknown'] += 1
        age = attr.get('native_age_label', 'unknown')
        attribute_ages[age if age in attribute_ages else 'unknown'] += 1
        attribute_orientations[a if a in attribute_orientations else 'unknown'] += 1
        presentation = attr.get('presentation_proxy', 'unclear').replace('_presentation_proxy', '')
        attribute_presentation[presentation if presentation in attribute_presentation else 'unclear'] += 1
        conflicts += source == 'conflict_abstention'
    result['votes'] = votes
    result['combined'] = {'counts': counts, 'count_sources': sources,
                          'orientation_counts': orientation_counts, 'direction_counts': direction_counts,
                          'carrying_backpack_counts': carrying,
                          'adults': 0, 'children': 0, 'age_unknown': len(persons),
                          'age_status': 'abstained_unvalidated_classifier',
                          'presentation_status': 'abstained_unvalidated_native_binary_proxy',
                          'orientation_conflicts': conflicts}
    result['attribute_summary'] = {'age_counts': attribute_ages,
                                   'orientation_counts': attribute_orientations,
                                   'presentation_proxy_counts': attribute_presentation}
    result['needs_review'] = bool(conflicts or direction_counts['unclear'] or len(persons)
                                 or any(v['status'] == 'no_majority' for v in votes.values()))
    return result
