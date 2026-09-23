"""Events, cross-frame person association and motion direction.

An event is a run of images from one camera separated by short time gaps. A
track is a hypothesis that detections in different frames of one event show
the same person (similar appearance, size and position); it de-duplicates
counts and is never an identity claim. Direction from motion is image-relative
movement of the foot point; body facing is only a labelled fallback for
single observations. Pure Python and deterministic; scipy is optional (greedy
matching without it). Outputs are plain JSON types with finite numbers.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import math
import numbers
from pathlib import PurePosixPath
import re

RULE_VERSION = "events_v2"
POLICY = {
    "sequence_window": 3,              # |d sequence number| keeping untimed images together
    "assumed_seconds_per_frame": 1.0,  # dt per frame step when a capture time is missing
    "appearance_weight": .60, "size_weight": .25, "displacement_weight": .15,
    "unknown_appearance_cost": .50, "unknown_size_cost": .50, "unknown_displacement_cost": .50,
    "displacement_seconds_factor": 1.2,
    "max_displacement_allowance": 1.0,  # hard gate: mean height * (1 + 1.2 * elapsed seconds)
    "max_cost": .55, "max_appearance_distance": .60, "max_size_ratio": 1.8,
    "ambiguity_margin": .05,           # rival link this close in cost -> ambiguous track
    "radial_threshold": .15, "radial_dominance": 2.0, "lateral_threshold": .25,
    "stationary_seconds": 4.0,
    "review_relative_difference": .50,
}
APPEARANCE_VERSION = "hsv_v1"  # descriptor version understood by the local distance fallback
APPEARANCE_BINS = 256
FACINGS = ("left", "right", "toward", "away")
DIRECTIONS = ("left", "right", "toward", "away", "stationary", "unclear")
AGES = ("adult", "child", "unknown")
OBJECT_FIELDS = ("bicycles", "strollers", "motorcycles", "atv_utv", "other_vehicles", "dogs", "backpacks",
                 "large_bags", "large_bags_uncertain")
PEOPLE_FIELDS = ("people_unique", "people_max_frame", "adults", "children", "age_unknown",
                 *("dir_" + d for d in DIRECTIONS), "direction_from_motion", "direction_from_facing")
SUMMARY_FIELDS = ("event_id", "camera_id", "start", "end", "duration_seconds", "image_count", "images",
                  *PEOPLE_FIELDS, *OBJECT_FIELDS, "needs_review", "review_reasons")
_TIE_BREAK = 1e-9  # a link exactly at max_cost still beats leaving both sides unmatched


def _number(value):
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def _integer(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip().isdecimal():  # isdigit() also accepts e.g. superscripts
        return int(value.strip())
    return None


def _unit(value):
    """Clip a non-negative ratio to [0, 1]; an overflowed (non-finite) ratio counts as 1."""
    return min(1.0, value) if math.isfinite(value) else 1.0


def _moment(value):
    """Capture time as a datetime with any UTC offset kept; None if absent or malformed."""
    if isinstance(value, datetime):
        return value
    if isinstance(value, str) and value.strip():
        try:
            return datetime.fromisoformat(value.strip())
        except ValueError:
            return None
    return None


def _comparable(moment):
    try:
        if moment is None or moment.utcoffset() is None:
            return moment
        return moment.astimezone(timezone.utc).replace(tzinfo=None)
    except (OverflowError, ValueError, TypeError):
        return None


def parse_time(value):
    """ISO capture time -> naive datetime (aware values converted to UTC); None if absent or malformed."""
    return _comparable(_moment(value))


def _seconds(a, b):
    return abs((b - a).total_seconds()) if a is not None and b is not None else None


def _order_key(record):
    moment = parse_time(record.get("capture_time"))
    sequence = _integer(record.get("sequence_number"))
    path = str(record.get("relative_path") or "")
    return (moment is None, moment or datetime.min, sequence is None, sequence or 0,
            path.casefold(), path, str(record.get("image_id") or ""))


def _timed(frames):
    """(comparable, as-recorded) capture times of the frames that have one, in frame order."""
    pairs = ((parse_time(r.get("capture_time")), _moment(r.get("capture_time"))) for r in frames)
    return [pair for pair in pairs if pair[0] is not None]


def _camera(record):
    camera = record.get("camera_id")
    return "" if camera is None else str(camera)


def _view_key(record):
    """Coordinate domain for association; a resized or different camera view starts anew."""
    dimensions = tuple(v if v is not None and v > 0 else None
                       for v in (_number(record.get("width")), _number(record.get("height"))))
    return (_camera(record), *dimensions, *(str(record.get(k) or "").strip()
                                           for k in ("camera_make", "camera_model")))


def _continues(previous, record, gap_seconds):
    if _view_key(previous) != _view_key(record):
        return False
    gap = _seconds(parse_time(previous.get("capture_time")), parse_time(record.get("capture_time")))
    if gap is not None:
        return gap <= gap_seconds
    a, b = _integer(previous.get("sequence_number")), _integer(record.get("sequence_number"))
    return a is not None and b is not None and abs(b - a) <= POLICY["sequence_window"]


def _sanitise(text):
    return re.sub(r"[^\w.-]+", "_", str(text)).strip("._")


def event_id_for(event_records, index=None):
    """``{camera_id}__{first capture time or first filename}`` sanitised, plus ``__NNNN`` when indexed.

    The time is the camera's recorded wall-clock time of the earliest frame
    (a UTC offset orders frames but does not shift the stamp).
    """
    frames = sorted(event_records, key=_order_key)
    first = frames[0] if frames else {}
    timed = _timed(frames)
    if timed:
        stamp = min(timed, key=lambda pair: pair[0])[1].strftime("%Y-%m-%dT%H-%M-%S")
    else:
        path = str(first.get("relative_path") or "").replace("\\", "/")
        stamp = PurePosixPath(path).name if path else str(first.get("image_id") or "")
    text = (_sanitise(_camera(first)) or "root") + "__" + (_sanitise(stamp) or "event")
    return text if index is None else f"{text}__{int(index):04d}"


def group_events(records, gap_seconds=120):
    """Split each camera's ordered images where the time gap exceeds ``gap_seconds``.

    Order is (capture_time, sequence_number, relative_path) with untimed images
    last. When either neighbour lacks a time, sequence numbers within
    +-``sequence_window`` keep them together; lacking both, a new event starts.
    A change of image dimensions or camera make/model also starts a new event.
    Returns shallow record copies carrying ``event_id``; inputs are not mutated.
    """
    gap = _number(gap_seconds)
    if gap is None or gap < 0:
        raise ValueError("gap_seconds must be a non-negative number")
    cameras = {}
    for record in records:
        cameras.setdefault(_camera(record), []).append(record)
    events = []
    for camera in sorted(cameras):
        current = []
        for record in sorted(cameras[camera], key=_order_key):
            if current and not _continues(current[-1], record, gap):
                events.append(current)
                current = []
            current.append(record)
        events.append(current)
    # Ids depend only on the event itself (camera + first capture time, or the
    # first filename): adding photos elsewhere never renumbers existing events.
    # Rare collisions get a suffix in deterministic event order.
    output, seen = [], Counter()
    for event in events:
        base = event_id_for(event)
        seen[base] += 1
        event_id = base if seen[base] == 1 else f"{base}__{seen[base]:02d}"
        output.append([{**record, "event_id": event_id} for record in event])
    return output


def _box(value):
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    box = [_number(v) for v in value]
    if None in box or box[2] <= box[0] or box[3] <= box[1]:
        return None
    return box


def _height(person):
    height = _number(person.get("height_px"))
    if height is not None and height > 0:
        return height
    box = _box(person.get("box"))
    height = box[3] - box[1] if box else None
    return height if height is not None and 0 < height < math.inf else None


def _foot(person):
    foot = person.get("foot")
    if isinstance(foot, (list, tuple)) and len(foot) == 2:
        point = [_number(v) for v in foot]
        if None not in point:
            return point
    box = _box(person.get("box"))
    return [box[0] / 2 + box[2] / 2, box[3]] if box else None


def _log_ratio(numerator, denominator):
    ratio = numerator / denominator
    return math.log(ratio) if 0 < ratio < math.inf else math.log(numerator) - math.log(denominator)


def _histogram(descriptor, part):
    """L1-normalised part histogram, or None when missing or malformed."""
    values = descriptor.get(part)
    if not isinstance(values, (list, tuple)) or len(values) != APPEARANCE_BINS:
        return None
    values = [_number(v) for v in values]
    if None in values or min(values) < 0:
        return None
    total = sum(values)
    if not 0 < total < math.inf:
        return None
    return [v / total for v in values]


def _local_distance(a, b):
    """Fallback with ``appearance.distance`` semantics: mean Hellinger distance of parts valid in both."""
    if not all(isinstance(d, dict) and d.get("valid") is True and d.get("version") == APPEARANCE_VERSION
               for d in (a, b)):
        return None
    values = []
    for part in ("upper", "lower"):
        p, q = _histogram(a, part), _histogram(b, part)
        if p is not None and q is not None:
            coefficient = math.fsum(math.sqrt(x * y) for x, y in zip(p, q))
            values.append(math.sqrt(max(0.0, 1.0 - min(1.0, coefficient))))
    return round(sum(values) / len(values), 6) if values else None


def _distance_function():
    try:
        from .appearance import distance
    except (ImportError, OSError):
        return _local_distance
    return distance


def _clean_distance(measure, a, b):
    if a is None or b is None:
        return None
    try:
        value = _number(measure(a, b))
    except (TypeError, ValueError, KeyError, IndexError, AttributeError, ArithmeticError):
        return None  # a malformed cached descriptor means unknown appearance, not a crash
    return None if value is None else min(1.0, max(0.0, value))


def appearance_distance(a, b):
    """Appearance distance in [0, 1] or None; uses ``appearance.distance`` when importable."""
    return _clean_distance(_distance_function(), a, b)


def _link(earlier, later, seconds, measure, skip_above=None):
    """``link_cost``; with ``skip_above`` the appearance distance is not computed for a link
    that cannot be valid or near the gate (the returned cost is then a lower bound)."""
    p = POLICY
    passed, size, displacement = True, None, None
    ha, hb = _height(earlier), _height(later)
    if ha and hb:
        ratio = max(ha, hb) / min(ha, hb)
        passed = ratio <= p["max_size_ratio"]
        size = _unit(abs(math.log(ratio)) / math.log(2))
        fa, fb = _foot(earlier), _foot(later)
        if fa and fb:
            elapsed = max(0.0, _number(seconds) or 0.0)
            allowance = (ha / 2 + hb / 2) * (1 + p["displacement_seconds_factor"] * elapsed)
            relative_displacement = math.hypot(fb[0] - fa[0], fb[1] - fa[1]) / allowance
            passed = passed and math.isfinite(relative_displacement) and relative_displacement <= p["max_displacement_allowance"]
            displacement = _unit(relative_displacement)
    size_term = p["size_weight"] * (p["unknown_size_cost"] if size is None else size)
    displacement_term = p["displacement_weight"] * (
        p["unknown_displacement_cost"] if displacement is None else displacement)
    if skip_above is not None and (not passed or size_term + displacement_term > skip_above):
        return size_term + displacement_term, passed
    d = _clean_distance(measure, earlier.get("appearance"), later.get("appearance"))
    passed = passed and (d is None or d <= p["max_appearance_distance"])
    appearance_term = p["appearance_weight"] * (p["unknown_appearance_cost"] if d is None else d)
    return appearance_term + size_term + displacement_term, passed


def link_cost(earlier, later, seconds, distance=None):
    """Return ``(cost, hard_gates_passed)`` for linking two person observations.

    cost = 0.6 appearance + 0.25 size mismatch + 0.15 normalised displacement;
    each unknown term costs 0.5. Hard gates: appearance distance and size
    ratio and displacement relative to the time/height allowance. A link is
    valid when the hard gates pass and cost <= max_cost.
    """
    return _link(earlier, later, seconds, distance or _distance_function())


def _people(record):
    people = record.get("people")
    return [p for p in people if isinstance(p, dict)] if isinstance(people, list) else []


def _person_id(person, index):
    value = person.get("person_id")
    return f"person_{index + 1:04d}" if value is None else value


def _solver():
    try:
        from scipy.optimize import linear_sum_assignment
    except (ImportError, OSError):
        return None, "greedy"
    return linear_sum_assignment, "hungarian"


def _assign(costs, valid, solver):
    """Valid (row, col) links maximising the total slack ``max_cost - cost``.

    Hungarian when available (an unmatched pair costs 0, so one strong link is
    never traded for two weak ones), else greedy lowest cost first.
    """
    rows, cols = len(costs), len(costs[0]) if costs else 0
    if not rows or not cols:
        return []
    if solver is not None:
        limit = POLICY["max_cost"] + _TIE_BREAK
        matrix = [[costs[r][c] - limit if valid[r][c] else 0.0 for c in range(cols)] for r in range(rows)]
        chosen = zip(*solver(matrix))
        return sorted((int(r), int(c)) for r, c in chosen if valid[int(r)][int(c)])
    pairs, used_rows, used_cols = [], set(), set()
    for _, r, c in sorted((costs[r][c], r, c) for r in range(rows) for c in range(cols) if valid[r][c]):
        if r not in used_rows and c not in used_cols:
            pairs.append((r, c))
            used_rows.add(r)
            used_cols.add(c)
    return sorted(pairs)


def associate(event_records, max_frame_gap=2, *, distance=None):
    """Link people across the frames of one event into tracks.

    Frame i is matched against tracks whose last observation lies in frames
    i-1 .. i-max_frame_gap (a person missed in between is bridged). A track is
    ``ambiguous`` when a rival link was within ``ambiguity_margin`` of the
    chosen one, or when a new track narrowly missed the cost gate.
    """
    if isinstance(max_frame_gap, bool) or not isinstance(max_frame_gap, numbers.Integral) or max_frame_gap < 1:
        raise ValueError("max_frame_gap must be a positive integer")
    solver, matcher = _solver()
    measure = distance or _distance_function()
    frames = sorted(event_records, key=_order_key)
    times = [parse_time(r.get("capture_time")) for r in frames]
    margin, tracks, active = POLICY["ambiguity_margin"], [], []
    near_gate = POLICY["max_cost"] + margin
    for i, record in enumerate(frames):
        people = _people(record)
        candidates = [t for t in active if i - t["frames"][-1] <= max_frame_gap
                      and _view_key(frames[t["frames"][-1]]) == _view_key(record)]
        costs, valid, passed = [], [], []
        for person in people:
            row = []
            for track in candidates:
                j = track["frames"][-1]
                seconds = _seconds(times[j], times[i])
                if seconds is None:
                    seconds = (i - j) * POLICY["assumed_seconds_per_frame"]
                row.append(_link(track["last"], person, seconds, measure, near_gate))
            costs.append([c for c, _ in row])
            passed.append([ok for _, ok in row])
            valid.append([ok and c <= POLICY["max_cost"] for c, ok in row])
        pairs = _assign(costs, valid, solver)
        by_row, by_col = dict(pairs), {c: r for r, c in pairs}
        active = candidates
        for r, person in enumerate(people):
            observation = _observation(i, record, person, _person_id(person, r))
            c = by_row.get(r)
            if c is not None:
                track, cost = candidates[c], costs[r][c]
                if _clean_distance(measure, track["last"].get("appearance"), person.get("appearance")) is None:
                    track["reasons"].add("missing_appearance")
                if not all((_height(track["last"]), _height(person), _foot(track["last"]), _foot(person))):
                    track["reasons"].add("missing_geometry")
                rivals = [costs[r][k] for k in range(len(candidates)) if k != c and valid[r][k]]
                rivals += [costs[k][c] for k in range(len(people)) if k != r and valid[k][c]]
                if any(v < cost + margin for v in rivals):
                    track["reasons"].add("competing_candidate")
                track["observations"].append(observation)
                track["frames"].append(i)
                track["link_costs"].append(round(cost, 6))
                track["last"] = person
                continue
            reasons = set()
            for k in range(len(candidates)):
                if valid[r][k] and k in by_col and costs[r][k] < costs[by_col[k]][k] + margin:
                    reasons.add("competing_candidate")
                elif passed[r][k] and not valid[r][k] and costs[r][k] <= near_gate:
                    reasons.add("near_gate_candidate")
            track = {"observations": [observation], "frames": [i], "link_costs": [],
                     "reasons": reasons, "last": person}
            tracks.append(track)
            active = active + [track]
    prefix = str(frames[0].get("event_id") or "") if frames else ""
    return [{"track_id": f"{prefix}__t{n:03d}" if prefix else f"t{n:03d}",
             "observations": [(o["image_id"], o["person_id"]) for o in t["observations"]],
             "frames": t["frames"], "link_costs": t["link_costs"],
             "ambiguous": bool(t["reasons"]), "reasons": sorted(t["reasons"]), "matcher": matcher,
             "direction": track_direction(t["observations"], allow_motion=not t["reasons"]),
             "age": track_age(t["observations"])}
            for n, t in enumerate(tracks, 1)]


def _observation(frame_index, record, person, person_id):
    return {"image_id": record.get("image_id"), "person_id": person_id, "frame": frame_index,
            "capture_time": record.get("capture_time"), "sequence_number": record.get("sequence_number"),
            "box": person.get("box"), "foot": person.get("foot"), "height_px": person.get("height_px"),
            "facing": person.get("facing"), "age": person.get("age"), "large_bag": person.get("large_bag")}


def _observation_index(frames):
    index = {}
    for f, record in enumerate(frames):
        for k, person in enumerate(_people(record)):
            key = (record.get("image_id"), _person_id(person, k))
            try:
                index.setdefault(key, _observation(f, record, person, key[1]))
            except TypeError:
                continue  # an unhashable id cannot be referenced by an (image_id, person_id) pair
    return index


def _resolve(index, track):
    observations = []
    items = track.get("observations") if isinstance(track, dict) else None
    for item in items or []:
        try:
            image_id, person_id = item
            hit = index.get((image_id, person_id))
        except (TypeError, ValueError):
            continue
        if hit:
            observations.append(hit)
    return sorted(observations, key=lambda o: o["frame"])


def track_observations(event_records, track):
    """Per-observation person data of one track, in frame order."""
    return _resolve(_observation_index(sorted(event_records, key=_order_key)), track)


def track_direction(track_obs, event_records=None, *, allow_motion=True):
    """Direction from first vs last observation; facing majority for single observations.

    ``track_obs`` is a frame-ordered list of observation dicts (see
    ``track_observations``), or a track dict / list of (image_id, person_id)
    pairs together with its event records. lateral = d foot_x / mean height,
    radial = log(h_last / h_first). Motion is image-relative; ``source`` says
    whether motion, facing or nothing was used. Uncertain associations suppress
    motion and retain only a facing fallback.
    """
    p = POLICY
    result = {"direction": "unclear", "source": "none", "lateral": None, "radial": None, "seconds": None}
    if isinstance(track_obs, dict):
        allow_motion = allow_motion and not track_obs.get("ambiguous", False)
        track_obs = track_observations(event_records or [], track_obs)
    items = list(track_obs or [])
    if event_records is not None and any(not isinstance(o, dict) for o in items):
        items = track_observations(event_records, {"observations": items})
    observations = [o for o in items if isinstance(o, dict)]
    if not observations:
        return result
    first, last = observations[0], observations[-1]
    seconds = _seconds(parse_time(first.get("capture_time")), parse_time(last.get("capture_time")))
    result["seconds"] = None if seconds is None else round(seconds, 6)
    if allow_motion and len(observations) >= 2:
        ha, hb, fa, fb = _height(first), _height(last), _foot(first), _foot(last)
        if ha and hb and fa and fb:
            lateral, radial = (fb[0] - fa[0]) / (ha / 2 + hb / 2), _log_ratio(hb, ha)
            if math.isfinite(lateral) and math.isfinite(radial):
                dominant = abs(radial) * p["radial_dominance"] >= abs(lateral)
                if radial >= p["radial_threshold"] and dominant:
                    direction = "toward"
                elif radial <= -p["radial_threshold"] and dominant:
                    direction = "away"
                elif abs(lateral) >= p["lateral_threshold"]:
                    direction = "right" if lateral > 0 else "left"
                elif seconds is not None and seconds >= p["stationary_seconds"]:
                    direction = "stationary"
                else:
                    direction = "unclear"
                result.update(direction=direction, source="motion",
                              lateral=round(lateral, 6), radial=round(radial, 6))
                return result
    votes = Counter(o.get("facing") for o in observations if o.get("facing") in FACINGS).most_common()
    if votes and (len(votes) == 1 or votes[0][1] > votes[1][1]):
        result.update(direction=votes[0][0], source="facing")
    return result


def _age(value):
    if isinstance(value, dict):
        value = value.get("age")
    return value if isinstance(value, str) and value in AGES else "unknown"


def track_age(track_obs):
    """Child if any child and no adult observation; adult if the reverse; else unknown."""
    labels = {_age(o.get("age")) for o in track_obs or [] if isinstance(o, dict)}
    if "child" in labels and "adult" not in labels:
        return "child"
    if "adult" in labels and "child" not in labels:
        return "adult"
    return "unknown"


def _max_count(frames, field):
    values = []
    for record in frames:
        objects = record.get("objects")
        value = _integer(objects.get(field)) if isinstance(objects, dict) else None
        if value is not None and value >= 0:
            values.append(value)
    return max(values) if values else None


def _track_reasons(track):
    reasons = track.get("reasons")
    listed = [r for r in reasons if isinstance(r, str)] if isinstance(reasons, (list, tuple, set)) else []
    return listed or ["ambiguous_track"]


def summarize_event(event_records, tracks=None):
    """One events-CSV row: de-duplicated people (tracks) and per-frame maxima of objects.

    None means unassessed (no frame had a people list / no frame reported the
    object field); 0 means assessed and absent. ``needs_review`` when the
    unique count differs from the busiest frame by > 50% or a track is
    ambiguous; ``review_reasons`` lists why (sorted). start/end keep the
    recorded time (and UTC offset, if any) of the earliest/latest frame.
    """
    frames = sorted(event_records, key=_order_key)
    tracks = associate(frames) if tracks is None else [t for t in tracks if isinstance(t, dict)]
    timed = _timed(frames)
    start = min(timed, key=lambda pair: pair[0]) if timed else None
    end = max(timed, key=lambda pair: pair[0]) if timed else None
    index = _observation_index(frames)
    ages, directions, sources, reasons = Counter(), Counter(), Counter(), set()
    for track in tracks:
        observations = _resolve(index, track)
        ages[track_age(observations)] += 1
        direction = track_direction(observations, allow_motion=not track.get("ambiguous", False))
        directions[direction["direction"]] += 1
        sources[direction["source"]] += 1
        if track.get("ambiguous"):
            reasons.update(_track_reasons(track))
    unique = len(tracks)
    max_frame = max((len(_people(r)) for r in frames), default=0)
    if abs(unique - max_frame) > POLICY["review_relative_difference"] * max_frame:
        reasons.add("people_unique_vs_max_frame")
    first = frames[0] if frames else {}
    summary = {"event_id": first.get("event_id") or (event_id_for(frames) if frames else None),
               "camera_id": first.get("camera_id"),
               "start": start[1].isoformat() if start else None,
               "end": end[1].isoformat() if end else None,
               "duration_seconds": round(_seconds(start[0], end[0]), 6) if start else None,
               "image_count": len(frames), "images": [r.get("image_id") for r in frames],
               "people_unique": unique, "people_max_frame": max_frame,
               "adults": ages["adult"], "children": ages["child"], "age_unknown": ages["unknown"],
               **{"dir_" + d: directions[d] for d in DIRECTIONS},
               "direction_from_motion": sources["motion"], "direction_from_facing": sources["facing"],
               **{field: _max_count(frames, field) for field in OBJECT_FIELDS},
               "needs_review": bool(reasons), "review_reasons": sorted(reasons)}
    if frames and not any(isinstance(r.get("people"), list) for r in frames):
        summary.update(dict.fromkeys(PEOPLE_FIELDS), needs_review=False, review_reasons=[])
    return summary
