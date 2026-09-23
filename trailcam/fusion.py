"""Per-image post-processing: transparent, uncalibrated expert combination.

Runs on cached raw expert output, so every threshold here can change without
re-inference. Unknown is never zero. Age and travel direction need other
images (camera calibration, events) and are finalized in ``postprocess``.
"""
from collections import Counter
from statistics import mean

from . import roster
from .vision import assign_bag, count_objects, iou, semantic_vehicle_dedup

COUNT_FIELDS = ('people_total', 'bicycles', 'strollers', 'motorcycles', 'atv_utv',
                'other_vehicles', 'dogs', 'backpacks', 'cars_trucks_buses', 'kick_scooters')
DIRECTIONS = ('left', 'right', 'toward', 'away', 'unclear')
TRAVEL_DIRECTIONS = ('left', 'right', 'toward', 'away', 'stationary', 'unclear')
ORIENTATIONS = ('front', 'back', 'side', 'unknown')
BAG_LABELS = ('backpack', 'suitcase', 'duffel bag')
# Object verification. A cluster is one physical object seen by >= 1 expert.
# Corroborating experts confirm it; a lone expert must reach ``single``.
# ``veto`` drops a cluster overlapping another object type (e.g. a "dog" that
# is really a crouching person, a "motorcycle" that is really a bicycle).
# MegaDetector "vehicle" is not used: locally it fired on bicycles/strollers.
OBJECT_POLICY = {
    'threshold': 0.25,
    'match_iou': 0.3,
    'fields': {
        'bicycles': {'experts': {'yoloe': ('bicycle',), 'yolo26n': ('bicycle',)}, 'single': 0.25},
        'strollers': {'experts': {'yoloe': ('baby stroller',)}, 'single': 0.25},
        'motorcycles': {'experts': {'yoloe': ('motorcycle',), 'yolo26n': ('motorcycle',)}, 'single': 0.5,
                        'veto': {'labels': ('bicycle',), 'experts': ('yoloe', 'yolo26n'), 'iou': 0.4, 'conf': 0.25}},
        'atv_utv': {'experts': {'yoloe': ('all-terrain vehicle', 'utility terrain vehicle', 'golf cart')},
                    'single': 0.35},
        'other_vehicles': {'experts': {'yoloe': ('car', 'truck', 'bus', 'tractor'), 'yolo26n': ('car', 'truck', 'bus')},
                           'single': 0.5},
        # A generic animal detection cannot confirm dog species (e.g. an ibex).
        'dogs': {'experts': {'yoloe': ('dog',), 'yolo26n': ('dog',)}, 'single': 0.9,
                 'veto': {'labels': ('person',), 'experts': ('yolo26n', 'megadetector'), 'iou': 0.7, 'conf': 0.4}},
        'backpacks': {'experts': {'yoloe': ('backpack',), 'yolo26n': ('backpack',)}, 'single': 0.25},
        'cars_trucks_buses': {'experts': {'yoloe': ('car', 'truck', 'bus'), 'yolo26n': ('car', 'truck', 'bus')},
                              'single': 0.5},
        'kick_scooters': {'experts': {'yoloe': ('kick scooter',)}, 'single': 0.25},
    },
    # These experts only corroborate; they never create an object alone.
    'corroborate_only': ('megadetector',),
    'bag_threshold': 0.25,
    'vehicle_merge_iou': 0.4,
    'vehicle_label_iou': 0.8,
    'version': 'objects_v2_1',
}


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


VEHICLE_GROUPS = ({'all-terrain vehicle', 'utility terrain vehicle', 'golf cart'},
                  {'car', 'truck', 'bus', 'tractor', 'motorcycle'})


def _dedup_vehicle_groups(detections):
    """Near-identical boxes competing for labels are merged inside each group only."""
    output = detections
    for group in VEHICLE_GROUPS:
        members = [d for d in output if d['label'] in group]
        if len(members) < 2:
            continue
        kept, _ = semantic_vehicle_dedup(members)
        keep_ids = {id(d) for d in kept}
        output = [d for d in output if d['label'] not in group or id(d) in keep_ids]
    return output


def thresholded(result, policy=OBJECT_POLICY):
    """Expert detections at the counting threshold, outside the data strip."""
    strip, height = result.get('data_strip'), result.get('height', 0)
    output = {}
    for name, expert in result.get('experts', {}).items():
        kept = [d for d in expert.get('detections', []) if d['confidence'] >= policy['threshold']
                and roster.strip_fraction(d['xyxy'], strip, height) < .5]
        if name in ('yoloe', 'yolo26n'):
            kept = _dedup_vehicle_groups(kept)
        output[name] = kept
    return output


def verified_objects(detections, field, policy=OBJECT_POLICY, all_detections=None):
    """Cluster same-field boxes across experts and keep corroborated objects.

    ``detections`` are thresholded; ``all_detections`` (default: the same)
    supply veto boxes, which use their own confidence floor.
    """
    spec = policy['fields'][field]
    all_detections = detections if all_detections is None else all_detections
    boxes = []
    for name, labels in spec['experts'].items():
        for d in detections.get(name, []):
            if d['label'] in labels:
                boxes.append((d['confidence'], name, d))
    boxes.sort(key=lambda t: (-t[0], t[1]))
    clusters = []
    for confidence, name, d in boxes:
        best, best_iou = None, 0.0
        for cluster in clusters:
            if name in cluster['experts']:
                continue
            value = max(iou(d['xyxy'], m['xyxy']) for m in cluster['experts'].values())
            if value > best_iou:
                best, best_iou = cluster, value
        if best is not None and best_iou >= policy['match_iou']:
            best['experts'][name] = d
        else:
            clusters.append({'experts': {name: d}})
    veto = spec.get('veto')
    veto_boxes = [d['xyxy'] for name in (veto['experts'] if veto else ())
                  for d in all_detections.get(name, [])
                  if d['label'] in veto['labels'] and d['confidence'] >= veto['conf']]
    objects = []
    for cluster in clusters:
        creators = {n: d for n, d in cluster['experts'].items() if n not in policy['corroborate_only']}
        if not creators:
            continue
        top = max(d['confidence'] for d in creators.values())
        anchor = max(creators.values(), key=lambda d: d['confidence'])
        if veto and any(iou(anchor['xyxy'], b) >= veto['iou'] for b in veto_boxes):
            status = 'vetoed'
        elif len(cluster['experts']) >= 2:
            status = 'corroborated'
        elif top >= spec['single']:
            status = 'single_expert_confident'
        else:
            status = 'unverified'
        objects.append({'field': field, 'status': status, 'confidence': top, 'xyxy': anchor['xyxy'],
                        'experts': sorted(cluster['experts']), 'label': anchor['label']})
    return objects


COUNTED = ('corroborated', 'single_expert_confident')


def _merge_vehicle_labels(objects, counts, policy):
    """Resolve competing vehicle categories before deriving their counts.

    Utility carts keep the trail-specific ATV/UTV reading over overlapping
    car/truck readings. ATV versus motorcycle uses confidence. Other competing
    motor-vehicle labels need near-identical boxes and use confidence. The
    car/truck/bus subset is then derived from this same physical-object set;
    it cannot retain an object that the parent category discarded.
    """
    atvs = objects.get('atv_utv', [])
    motorcycles = objects.get('motorcycles', [])
    others = objects.get('other_vehicles', [])
    # A quad labelled both ATV and motorcycle is one vehicle. Process the
    # strongest motorcycles first; a previously suppressed ATV cannot suppress
    # a later motorcycle.
    for motorcycle in sorted(motorcycles, key=lambda o: -o['confidence']):
        if motorcycle['status'] not in COUNTED:
            continue
        twins = [o for o in atvs if o['status'] in COUNTED
                 and iou(motorcycle['xyxy'], o['xyxy']) >= policy['vehicle_merge_iou']]
        if twins and max(o['confidence'] for o in twins) >= motorcycle['confidence']:
            motorcycle['status'] = 'merged_into_atv_utv'
        else:
            for twin in twins:
                twin['status'] = 'merged_into_motorcycles'
    for other in others:
        if other['status'] in COUNTED and any(
                atv['status'] in COUNTED
                and iou(other['xyxy'], atv['xyxy']) >= policy['vehicle_merge_iou'] for atv in atvs):
            other['status'] = 'merged_into_atv_utv'
    # Class-aware NMS can retain a car and a motorcycle on the same box, and
    # different experts can disagree about that label. Keep one interpretation.
    contenders = sorted((o for o in motorcycles + others if o['status'] in COUNTED),
                         key=lambda o: (-o['confidence'], o['field'], tuple(o['xyxy'])))
    kept = []
    for contender in contenders:
        twin = next((o for o in kept if o['field'] != contender['field']
                     and iou(o['xyxy'], contender['xyxy']) >= policy.get('vehicle_label_iou', .8)), None)
        if twin is None:
            kept.append(contender)
        else:
            contender['status'] = 'merged_into_' + twin['field']
    objects['cars_trucks_buses'] = [dict(o, field='cars_trucks_buses') for o in others
                                   if o['label'] in ('car', 'truck', 'bus')]
    for field in ('atv_utv', 'motorcycles', 'other_vehicles', 'cars_trucks_buses'):
        counts[field] = sum(o['status'] in COUNTED for o in objects.get(field, []))


def fuse(result, policy=OBJECT_POLICY, *, large_bags=True):
    """Accept people, verify objects and classify bags for one image result."""
    experts = result['experts']
    detections = thresholded(result, policy)
    raw = {name: expert.get('detections', []) for name, expert in experts.items()}
    for name in experts:
        experts[name]['counts'] = (dict.fromkeys(COUNT_FIELDS) if experts[name].get('skipped')
                                   else count_objects(detections.get(name, []), name))
    votes, counts, sources, objects = {}, {}, {}, {}
    for field in COUNT_FIELDS:
        names = ('yolo26n', 'yoloe', 'megadetector') if field == 'people_total' else ('yolo26n', 'yoloe')
        votes[field] = count_vote({name: experts.get(name, {}).get('counts', {}).get(field) for name in names})
    people = []
    for person in result['persons']:
        accepted, reason = roster.accept(person)
        person['accepted'], person['acceptance'] = accepted, reason
        if accepted:
            people.append(person)
    counts['people_total'], sources['people_total'] = len(people), roster.POLICY['version']
    rematch_pose(result, people)
    for field in COUNT_FIELDS[1:]:
        found = verified_objects(detections, field, policy, raw)
        objects[field] = found
        counts[field] = sum(o['status'] in COUNTED for o in found)
        sources[field] = policy['version']
    _merge_vehicle_labels(objects, counts, policy)
    unverified = {field: sum(o['status'] == 'unverified' for o in found) for field, found in objects.items()}
    from .bags import count_large_bags
    bag_results = _people(result, people, detections, policy, large_bags=large_bags)
    loose = [{'label': d['label'], 'box': d['xyxy'], 'confidence': d['confidence']}
             for d in detections.get('yoloe', []) if d['label'] in ('suitcase', 'duffel bag')
             and not any(d['detection_index'] in p['combined']['bag_detection_indices'] for p in people)]
    bag_totals = (count_large_bags(bag_results, loose) if large_bags
                  else {'large_bags': None, 'large_bags_uncertain': None})
    orientation_counts = Counter(p['combined']['orientation'] for p in people)
    facing_counts = Counter(p['combined']['facing'] for p in people)
    carrying = Counter('yes' if p['combined']['carrying_backpack'] is True else 'no'
                       if p['combined']['carrying_backpack'] is False else 'unknown' for p in people)
    result['votes'] = votes
    result['objects'] = objects
    result['combined'] = {
        'counts': counts, 'count_sources': sources, 'unverified_counts': unverified,
        'large_bags': bag_totals['large_bags'], 'large_bags_uncertain': bag_totals['large_bags_uncertain'],
        'large_bags_status': 'experimental_geometry' if large_bags else 'disabled',
        'loose_luggage': sum(d['confidence'] >= 0.35 for d in loose),
        'orientation_counts': {k: orientation_counts[k] for k in ORIENTATIONS},
        'facing_counts': {k: facing_counts[k] for k in DIRECTIONS},
        'carrying_backpack_counts': {k: carrying[k] for k in ('yes', 'no', 'unknown')},
        'orientation_conflicts': sum(p['combined']['orientation_source'] == 'conflict_abstention' for p in people),
        # Finalized by postprocess (needs camera calibration and events).
        'adults': None, 'children': None, 'age_unknown': len(people),
        'direction_counts': None, 'age_status': 'pending_postprocess',
    }
    result['needs_review'] = bool(result['combined']['orientation_conflicts']
                                  or any(v['status'] == 'no_majority' for v in votes.values())
                                  or any(unverified.values()) or bag_totals['large_bags_uncertain'])
    return result


POSE_REMATCH = {'min_iou': 0.30, 'min_confidence': 0.12}


def rematch_pose(result, people, policy=POSE_REMATCH):
    """Give accepted people without keypoints an unused overlapping pose detection.

    Inference clusters boxes at a strict IoU; a pose box drawn tighter or looser
    than the YOLOE box can end up in its own (often rejected) candidate. Keypoints
    feed age geometry, bags and facing, so they are re-linked here (greedy by
    IoU, each pose detection used once).
    """
    poses = result.get('experts', {}).get('pose', {}).get('detections', [])
    used = {p['members']['pose']['detection_index'] for p in people if 'pose' in p.get('members', {})}
    used |= {p['pose_rematch']['detection_index'] for p in people if p.get('pose_rematch')}
    pairs = []
    for person in people:
        if person.get('keypoints'):
            continue
        for position, pose in enumerate(poses):
            index = pose.get('detection_index', position)
            if index in used or pose.get('confidence', 0) < policy['min_confidence'] or not pose.get('keypoints'):
                continue
            value = iou(person['xyxy'], pose['xyxy'])
            if value >= policy['min_iou']:
                pairs.append((value, index, person['person_id'], person, pose))
    taken = set()
    for value, index, _, person, pose in sorted(pairs, key=lambda t: (-t[0], t[1], t[2])):
        if index in used or person['person_id'] in taken:
            continue
        used.add(index)
        taken.add(person['person_id'])
        person.update(keypoints=pose['keypoints'], pose_confidence=pose['confidence'],
                      orientation_evidence=pose.get('orientation_evidence'),
                      pose_rematch={'detection_index': index, 'iou': round(value, 4)})


def _people(result, people, detections, policy, *, large_bags=True):
    """Per-person facing, backpack and bag-size evidence."""
    from .bags import classify_person_bags
    bags = []
    for d in sorted((d for d in detections.get('yoloe', []) if d['label'] in BAG_LABELS
                     and d['confidence'] >= policy['bag_threshold']), key=lambda d: -d['confidence']):
        if not any(iou(d['xyxy'], b['xyxy']) >= 0.5 for b in bags):
            bags.append(d)
    roster_view = [{'person_id': p['person_id'], 'source_detection_index': n, 'xyxy': p['xyxy']}
                   for n, p in enumerate(people)]
    assigned = {p['person_id']: [] for p in people}
    for bag in bags:
        association = assign_bag(bag['xyxy'], roster_view)
        if association['person_id']:
            assigned[association['person_id']].append(bag)
    results = []
    for person in people:
        attr = person.get('attributes', {})
        evidence = person.get('orientation_evidence') or {}
        orientation, source = orientation_fusion(attr.get('native_orientation_label', 'unknown'),
                                                 evidence.get('orientation', 'unknown'))
        facing = {'front': 'toward', 'back': 'away'}.get(orientation, 'unclear')
        if orientation == 'side' and evidence.get('orientation') == 'side' and evidence.get('facing_direction') in ('left', 'right'):
            facing = evidence['facing_direction']
        mine = assigned[person['person_id']]
        bag_result = classify_person_bags(person['xyxy'], person.get('keypoints'),
                                          [{'label': b['label'], 'box': b['xyxy'], 'confidence': b['confidence'],
                                            'mask_area': None} for b in mine]) if large_bags else {
                                              'large_bag': None, 'reasons': ['large_bag_size_disabled'], 'features': {}}
        backpack = attr.get('backpack_presence')
        has_backpack_box = any(b['label'] == 'backpack' for b in mine)
        if has_backpack_box and backpack is False:
            backpack, bag_source = None, 'attribute_detector_conflict'
        elif has_backpack_box and backpack is True:
            bag_source = 'detector_spatial_support_and_attribute'
        elif has_backpack_box:
            backpack, bag_source = None, 'spatial_support_only_unverified'
        else:
            bag_source = 'attribute_only' if type(backpack) is bool else 'unknown'
        person['combined'] = {'orientation': orientation, 'orientation_source': source, 'facing': facing,
                              'carrying_backpack': backpack, 'backpack_source': bag_source,
                              'associated_backpack': has_backpack_box,
                              'large_bag': bag_result['large_bag'], 'bag_reasons': bag_result['reasons'],
                              'bag_assessed': bool(mine) and large_bags,
                              'bag_detection_indices': [b['detection_index'] for b in mine],
                              'age': 'unknown', 'age_method': 'pending', 'direction': 'unclear',
                              'direction_source': 'pending', 'presentation': 'unclear'}
        results.append(bag_result)
    return results
