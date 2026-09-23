"""Score manual labels against an images CSV (or events CSV) export; stdlib only.

    python -m trailcam.evaluate --labels labels.csv --export export.csv [--prefix combined_]
                                [--events events.csv [--event-labels event_labels.csv]] [--json out.json]
    python -m trailcam.evaluate --template export.csv --out labels.csv

Semantics: a blank label cell means "not labelled" and is skipped; a blank
prediction cell means "unsupported/unassessable" and is reported as a coverage
gap, never scored as zero. Image labels match export rows by relative_path,
then case-insensitive path. Only labels no path matches fall back to a unique
case-insensitive path-suffix match: a bare filename matches that name in any
folder, but a label naming a different folder never matches (trail cameras
reuse file names). Event labels (an ``event_id`` column) match events CSV rows
by exact event_id. With --events but only image labels, event labels are
derived for events whose every frame is labelled: objects and large bags are
the maximum over frames (the events CSV definition), people_max_frame the
largest frame people_total, and people_unique is checked against the
[max frame, sum of frames] bounds. Presence means count > 0. No model is
loaded, so there is no GPU/CPU choice.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import unicodedata
from itertools import chain
from pathlib import Path

from .storage import write_csv

VERSION = "evaluate_v1"
LABEL_FIELDS = ("people_total", "adults", "children", "age_unclear", "dir_left", "dir_right",
                "dir_toward", "dir_away", "dir_stationary", "dir_unclear", "bicycles", "strollers",
                "motorcycles", "atv_utv", "other_vehicles", "dogs", "backpacks", "large_bags")
EVENT_LABEL_FIELDS = ("people_unique",) + LABEL_FIELDS[1:]
# Scored at event level only when the labels have the column (derived event labels do).
EVENT_EXTRA_FIELDS = ("people_max_frame",)
# Event columns defined as the maximum over the event's frames, so derivable from image labels.
MAX_OVER_FRAMES = ("bicycles", "strollers", "motorcycles", "atv_utv", "other_vehicles", "dogs", "backpacks", "large_bags")
DIRECTIONS = ("left", "right", "toward", "away", "stationary", "unclear")
# Label name -> export name, then fallbacks (v1 exports use combined_direction_<d>).
PREDICTION_NAMES = {"age_unclear": "age_unknown"}
PREDICTION_ALIASES = {f"dir_{d}": (f"direction_{d}",) for d in DIRECTIONS}
LABEL_ALIASES = {"age_unclear": ("age_unknown",)}
# Extra rows scored only when every prediction part exists: uncertain bags counted as large.
DERIVED = {"large_bags_with_uncertain": ("large_bags", ("large_bags", "large_bags_uncertain"))}
GROUPS = {"age": ("adults", "children", "age_unclear"),
          "direction": tuple(f"dir_{d}" for d in DIRECTIONS)}
IMAGE_KEYS = ("relative_path", "filename", "file")
EVENT_CONTEXT = ("camera_id", "start", "end", "image_count", "images")
POLICY = {"within1_fields": ("people_total", "people_unique", "people_max_frame"), "decimals": 4,
          "largest_errors": 20, "primary_field": {"image": "people_total", "event": "people_unique"},
          "csv_field_limit": 100_000_000,
          # Larger cells are typos/pasted ids, not counts (and would overflow float metrics).
          "max_count": 1_000_000}
FORMULA_PREFIXES = ("=", "+", "-", "@")


def _r(value):
    """Rounded plain float (never -0.0), or None."""
    return None if value is None else round(float(value), POLICY["decimals"]) + 0.0


def parse_count(value):
    """Non-negative int up to POLICY max_count; None for blank. Raises ValueError otherwise ("2.0" is 2)."""
    text = "" if value is None else str(value).strip()
    if not text:
        return None
    try:
        number = int(text)
    except ValueError:
        try:
            real = float(text)
        except ValueError:
            raise ValueError(f"not a count: {value!r}") from None
        if not math.isfinite(real) or real != int(real):
            raise ValueError(f"not a count: {value!r}") from None
        number = int(real)
    if number < 0:
        raise ValueError(f"negative count: {value!r}")
    if number > POLICY["max_count"]:
        raise ValueError(f"implausibly large count: {text[:20]!r}")
    return number


def unescape_cell(value):
    """Undo the export's spreadsheet-safety apostrophe on formula-like text."""
    text = "" if value is None else str(value)
    return text[1:] if text.startswith("'") and text[1:].lstrip().startswith(FORMULA_PREFIXES) else text


def normalize_path(value):
    """POSIX separators, Unicode NFC, no leading './', surrounding whitespace removed."""
    text = unicodedata.normalize("NFC", unescape_cell(value).strip().replace("\\", "/"))
    while text.startswith("./"):
        text = text[2:]
    return text


def bare_name(value):
    return normalize_path(value).rsplit("/", 1)[-1].casefold()


def _parts(path):
    """Case-folded path components without empty or '.' parts."""
    return tuple(p for p in path.casefold().split("/") if p not in ("", "."))


def read_csv(path):
    """(columns, rows) of a UTF-8 CSV with optional BOM; header names are stripped."""
    if csv.field_size_limit() < POLICY["csv_field_limit"]:
        csv.field_size_limit(POLICY["csv_field_limit"])
    try:
        with Path(path).open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = [str(c).strip() for c in (reader.fieldnames or [])]
            reader.fieldnames = columns
            return columns, list(reader)
    except UnicodeDecodeError:
        raise ValueError(f"{path}: not UTF-8 text (save it as 'CSV UTF-8')") from None
    except csv.Error as exc:
        raise ValueError(f"{path}: {exc}") from None


def label_kind(columns):
    """'image' when a relative_path/filename column exists, else 'event' with event_id."""
    if any(k in columns for k in IMAGE_KEYS):
        return "image"
    if "event_id" in columns:
        return "event"
    raise ValueError("labels need a relative_path (or filename) column, or an event_id column")


def _columns(rows):
    return list(dict.fromkeys(k for row in rows for k in row if isinstance(k, str)))


def resolve_column(columns, names):
    available = set(columns)
    return next((name for name in names if name in available), None)


def prediction_candidates(field, prefix=""):
    """Export column names tried for a label field, in order."""
    names = (PREDICTION_NAMES.get(field, field),) + PREDICTION_ALIASES.get(field, ())
    prefixes = (prefix,) if not prefix or prefix.endswith("_") else (prefix, prefix + "_")
    return [p + n for p in prefixes for n in names]


def _event_key(row):
    return unescape_cell(row.get("event_id")).strip()


def _claim(candidates, method, used):
    if len(candidates) > 1:
        return ("ambiguous",)
    if candidates[0] in used:
        return ("duplicate",)
    used.add(candidates[0])
    return ("matched", method, candidates[0])


def match_rows(label_rows, prediction_rows, level="image"):
    """Pair label rows with prediction rows. Returns ([(key, label, prediction)], matching).

    Every exact/case-insensitive path match is claimed before any suffix or filename
    fallback, so a fallback never takes a row that another label names; pairs keep
    label order. A later label for an already claimed row is a 'duplicate'."""
    if level not in ("image", "event"):
        raise ValueError(f"unknown level: {level!r}")
    event = level == "event"
    label_rows, prediction_rows = list(label_rows), list(prediction_rows)
    exact, folded, suffixes, wholes = {}, {}, {}, {}
    for index, row in enumerate(prediction_rows):
        key = _event_key(row) if event else normalize_path(row.get("relative_path"))
        if not key:
            continue
        exact.setdefault(key, []).append(index)
        if not event:
            folded.setdefault(key.casefold(), []).append(index)
            parts = _parts(key)
            wholes.setdefault(parts, []).append(index)
            for k in range(1, len(parts) + 1):
                suffixes.setdefault(parts[-k:], []).append(index)
    keys = [_event_key(row) if event else next((k for k in (normalize_path(row.get(c)) for c in IMAGE_KEYS) if k), "")
            for row in label_rows]
    outcomes, used, fallback = [None] * len(keys), set(), []
    for i, key in enumerate(keys):
        candidates = key and (exact.get(key) or (None if event else folded.get(key.casefold())))
        if not key:
            outcomes[i] = ("missing_key",)
        elif candidates:
            outcomes[i] = _claim(candidates, "event_id" if event else "path", used)
        elif event:
            outcomes[i] = ("unmatched",)
        else:
            fallback.append(i)
    for i in fallback:
        parts = _parts(keys[i])
        # Label path is a suffix of the export path, or the export path a suffix of the label path;
        # two distinct candidates already make the label ambiguous, so stop there.
        found = []
        for j in chain(suffixes.get(parts, ()) if parts else (),
                       (j for k in range(1, len(parts) + 1) for j in wholes.get(parts[-k:], ()))):
            if j not in found:
                found.append(j)
                if len(found) > 1:
                    break
        outcomes[i] = (_claim(found, "filename" if len(parts) == 1 else "suffix", used)
                       if found else ("unmatched",))
    matching = {"label_rows": len(label_rows), "matched": 0,
                "matched_by": {"event_id": 0} if event else {"path": 0, "suffix": 0, "filename": 0},
                "unmatched": [], "ambiguous": [], "duplicate": [], "missing_key": 0,
                "prediction_rows": len(prediction_rows)}
    pairs = []
    for key, row, outcome in zip(keys, label_rows, outcomes):
        if outcome[0] == "matched":
            matching["matched_by"][outcome[1]] += 1
            pairs.append((key, row, prediction_rows[outcome[2]]))
        elif outcome[0] == "missing_key":
            matching["missing_key"] += 1
        else:
            matching[outcome[0]].append(key)
    matching["matched"] = len(pairs)
    matching["unlabelled_prediction_rows"] = len(prediction_rows) - len(used)
    matching["prediction_error_rows"] = sum(str(p.get("status") or "").strip() == "error" for _, _, p in pairs)
    return pairs, matching


def presence_metrics(flags):
    """Confusion counts and precision/recall/F1 from (label_present, predicted_present) pairs.

    Precision/recall are None when undefined; F1 = 2tp / (2tp + fp + fn), None when
    there are no positives on either side."""
    flags = [(bool(t), bool(p)) for t, p in flags]
    tp = sum(1 for t, p in flags if t and p)
    fp = sum(1 for t, p in flags if p and not t)
    fn = sum(1 for t, p in flags if t and not p)
    tn = sum(1 for t, p in flags if not t and not p)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": _r(tp / (tp + fp)) if tp + fp else None,
            "recall": _r(tp / (tp + fn)) if tp + fn else None,
            "f1": _r(2 * tp / (2 * tp + fp + fn)) if tp + fp + fn else None}


def count_metrics(pairs, within1=False):
    """MAE, bias (mean prediction - label), exact rate and presence from (label, prediction) ints."""
    pairs = list(pairs)
    n = len(pairs)
    errors = [p - t for t, p in pairs]
    out = {"n": n, "label_sum": sum(t for t, _ in pairs), "prediction_sum": sum(p for _, p in pairs),
           "mae": _r(sum(abs(e) for e in errors) / n) if n else None,
           "bias": _r(sum(errors) / n) if n else None,
           "exact": _r(sum(e == 0 for e in errors) / n) if n else None}
    if within1:
        out["within1"] = _r(sum(abs(e) <= 1 for e in errors) / n) if n else None
    out["presence"] = presence_metrics([(t > 0, p > 0) for t, p in pairs])
    return out


def group_metrics(pairs):
    """Per-image bucket errors from (label_buckets, predicted_buckets) equal-length int lists.

    bucket_mae = mean |error| per bucket cell; bucket_l1 = mean per-image sum of |errors|;
    exact = share of images with every bucket equal."""
    pairs = list(pairs)
    n = len(pairs)
    k = len(pairs[0][0]) if pairs else 0
    diffs = [sum(abs(p - t) for t, p in zip(labels, predicted)) for labels, predicted in pairs]
    return {"n": n, "bucket_mae": _r(sum(diffs) / (n * k)) if n and k else None,
            "bucket_l1": _r(sum(diffs) / n) if n else None,
            "exact": _r(sum(d == 0 for d in diffs) / n) if n else None}


def _check_columns(label_columns, prediction_columns, prediction_rows, specs, level):
    """Refuse inputs that would silently score the wrong thing."""
    key = "event_id" if level == "event" else "relative_path"
    if prediction_rows and key not in prediction_columns:
        raise ValueError(f"prediction CSV has no {key} column")
    if level == "event" and "relative_path" in prediction_columns:
        raise ValueError("event labels need the events CSV, not the images CSV")
    for name, columns, used in (
            ("label", label_columns, list(IMAGE_KEYS) + ["event_id"] + [c for c, _ in specs.values()]),
            ("prediction", prediction_columns, [key] + [c for _, cols in specs.values() for c in cols])):
        repeated = sorted({c for c in used if c and columns.count(c) > 1})
        if repeated:  # csv.DictReader silently keeps only the last of repeated columns
            raise ValueError(f"repeated {name} column(s): {', '.join(repeated)}")


def evaluate(label_rows, prediction_rows, level="image", prefix="", label_columns=None,
             prediction_columns=None):
    """Full report dict for one level ('image' or 'event'); JSON-serialisable, deterministic."""
    if level not in ("image", "event"):
        raise ValueError(f"unknown level: {level!r}")
    label_rows, prediction_rows = list(label_rows), list(prediction_rows)
    label_columns = list(label_columns) if label_columns is not None else _columns(label_rows)
    prediction_columns = (list(prediction_columns) if prediction_columns is not None
                          else _columns(prediction_rows))
    fields = list(EVENT_LABEL_FIELDS if level == "event" else LABEL_FIELDS)
    fields += [f for f in EVENT_EXTRA_FIELDS if level == "event" and f in label_columns]
    specs = {f: (resolve_column(label_columns, (f,) + LABEL_ALIASES.get(f, ())),
                 [resolve_column(prediction_columns, prediction_candidates(f, prefix))])
             for f in fields}
    for name, (base, parts) in DERIVED.items():
        columns = [resolve_column(prediction_columns, prediction_candidates(p, prefix)) for p in parts]
        if all(columns):
            specs[name] = (specs[base][0], columns)
    _check_columns(label_columns, prediction_columns, prediction_rows, specs, level)
    pairs, matching = match_rows(label_rows, prediction_rows, level)
    label_used = list(dict.fromkeys(c for c, _ in specs.values() if c))
    invalid, values = [], []
    for key, label_row, prediction_row in pairs:
        labels, bad = {}, set()
        for column in label_used:
            try:
                labels[column] = parse_count(label_row.get(column))
            except ValueError:
                labels[column] = None
                bad.add(column)
                invalid.append({"key": key, "column": column, "value": str(label_row.get(column))})
        entry = {}
        for field, (label_column, prediction_cols) in specs.items():
            predicted, prediction_bad = 0, False
            for column in prediction_cols:
                try:
                    part = parse_count(prediction_row.get(column)) if column else None
                except ValueError:
                    part, prediction_bad = None, True
                predicted = None if part is None or predicted is None else predicted + part
            entry[field] = (labels.get(label_column), predicted, label_column in bad, prediction_bad)
        values.append(entry)
    fields = {}
    for field, (label_column, prediction_cols) in specs.items():
        info = {"label_column": label_column, "prediction_columns": [c for c in prediction_cols if c],
                "status": ("missing_label_column" if not label_column else
                           "missing_prediction_column" if not all(prediction_cols) else "ok")}
        rows = [entry[field] for entry in values]
        if label_column:
            info["labelled"] = sum(label is not None for label, *_ in rows)
            info["label_invalid"] = sum(label_bad for _, _, label_bad, _ in rows)
        if info["status"] == "ok":
            info["prediction_blank"] = sum(t is not None and p is None and not pb for t, p, _, pb in rows)
            info["prediction_invalid"] = sum(t is not None and pb for t, _, _, pb in rows)
            info.update(count_metrics([(t, p) for t, p, _, _ in rows if t is not None and p is not None],
                                      within1=field in POLICY["within1_fields"]))
        fields[field] = info
    groups = {}
    for name, buckets in GROUPS.items():
        missing_labels = [b for b in buckets if not specs[b][0]]
        missing_predictions = [b for b in buckets if not all(specs[b][1])]
        if missing_labels or missing_predictions:
            groups[name] = {"status": "missing_label_column" if missing_labels else "missing_prediction_column",
                            "buckets": list(buckets), "missing": missing_labels or missing_predictions}
            continue
        labelled = [e for e in values if all(e[b][0] is not None for b in buckets)]
        scored = [([e[b][0] for b in buckets], [e[b][1] for b in buckets])
                  for e in labelled if all(e[b][1] is not None for b in buckets)]
        group = {"status": "ok", "buckets": list(buckets), "labelled": len(labelled),
                 "prediction_unavailable": len(labelled) - len(scored), **group_metrics(scored)}
        if "children" in buckets:
            i = buckets.index("children")
            group["children_presence"] = presence_metrics([(t[i] > 0, p[i] > 0) for t, p in scored])
        groups[name] = group
    primary = POLICY["primary_field"][level]
    errors = sorted(((key, e[primary][0], e[primary][1]) for (key, _, _), e in zip(pairs, values)
                     if primary in e and None not in e[primary][:2] and e[primary][0] != e[primary][1]),
                    key=lambda t: (-abs(t[2] - t[1]), t[0]))
    return {"version": VERSION, "level": level, "prefix": prefix, "matching": matching,
            "fields": fields, "groups": groups, "invalid_label_cells": invalid,
            "largest_errors": [{"key": k, "label": t, "prediction": p, "error": p - t}
                               for k, t, p in errors[:POLICY["largest_errors"]]]}


def derive_event_labels(label_rows, image_rows, event_rows=()):
    """Event label rows from image labels, only for events whose every frame is matched.

    A field gets an event label only when every frame has a valid label for it:
    MAX_OVER_FRAMES fields take the maximum; people_total gives people_max_frame and
    the people_unique bounds people_unique_min (max) / people_unique_max (sum). Frames
    per event = images CSV rows with that event_id (or the events CSV image_count if
    larger). Returns (rows sorted by event_id, {"events_derived", "events_incomplete"})."""
    image_rows = list(image_rows)
    pairs, _ = match_rows(label_rows, image_rows, "image")
    expected, frames = {}, {}
    for row in image_rows:
        key = _event_key(row)
        if key:
            expected[key] = expected.get(key, 0) + 1
    for row in event_rows:
        try:
            count = parse_count(row.get("image_count"))
        except ValueError:
            count = None
        if count and _event_key(row) in expected:
            expected[_event_key(row)] = max(expected[_event_key(row)], count)
    for _, label, image in pairs:
        if _event_key(image):
            frames.setdefault(_event_key(image), []).append(label)
    rows, incomplete = [], 0
    for key in sorted(frames):
        labels = frames[key]
        if len(labels) < expected[key]:
            incomplete += 1
            continue
        row = {"event_id": key}
        for field in MAX_OVER_FRAMES + ("people_total",):
            try:
                counts = [parse_count(label.get(field)) for label in labels]
            except ValueError:
                continue
            if None in counts:
                continue
            if field == "people_total":
                row.update(people_max_frame=max(counts), people_unique_min=max(counts),
                           people_unique_max=sum(counts))
            else:
                row[field] = max(counts)
        rows.append(row)
    return rows, {"events_derived": len(rows), "events_incomplete": incomplete}


def unique_bounds(derived_rows, event_rows):
    """Predicted people_unique vs the [people_unique_min, people_unique_max] label bounds.

    below = fewer unique people than the busiest labelled frame (under-count); above =
    more than the frame sum (over-count, e.g. failed de-duplication)."""
    pairs, _ = match_rows([r for r in derived_rows if r.get("people_unique_min") is not None], event_rows, "event")
    out = {"n": 0, "within": 0, "below": 0, "above": 0, "prediction_blank": 0, "prediction_invalid": 0}
    for _, label, event in pairs:
        try:
            predicted = parse_count(event.get("people_unique"))
        except ValueError:
            out["prediction_invalid"] += 1
            continue
        if predicted is None:
            out["prediction_blank"] += 1
            continue
        out["n"] += 1
        out["below" if predicted < label["people_unique_min"] else
            "above" if predicted > label["people_unique_max"] else "within"] += 1
    out["within_rate"] = _r(out["within"] / out["n"]) if out["n"] else None
    return out


def evaluate_derived_events(label_rows, image_rows, event_rows, event_columns=None):
    """Event-level report from image labels (see derive_event_labels); only derivable fields."""
    event_rows = list(event_rows)
    derived, info = derive_event_labels(label_rows, image_rows, event_rows)
    report = evaluate(derived, event_rows, "event", "", ("event_id",) + MAX_OVER_FRAMES + EVENT_EXTRA_FIELDS,
                      event_columns)
    for part in ("fields", "groups"):
        report[part] = {k: v for k, v in report[part].items() if v["status"] != "missing_label_column"}
    report["source"] = "image_labels"
    report["derivation"] = {**info, "people_unique_bounds": unique_bounds(derived, event_rows)}
    return report


def _cell(value, width, digits=3):
    text = "-" if value is None else f"{value:.{digits}f}" if isinstance(value, float) else str(value)
    return text.rjust(width)


def format_report(report):
    """Plain-text table for one level's report (ASCII only)."""
    m = report["matching"]
    by = ", ".join(f"by {k} {v}" for k, v in m["matched_by"].items())
    lines = [f"{report['level'].title()}-level evaluation (prediction prefix {ascii(report['prefix'])})",
             f"  labels {m['label_rows']}: matched {m['matched']} ({by}), unmatched {len(m['unmatched'])}, "
             f"ambiguous {len(m['ambiguous'])}, duplicate {len(m['duplicate'])}, missing key {m['missing_key']}",
             f"  prediction rows {m['prediction_rows']}: without label {m['unlabelled_prediction_rows']}, "
             f"matched error rows {m['prediction_error_rows']}",
             f"{'field':<27}{'n':>5}{'nopred':>7}{'MAE':>8}{'bias':>8}{'exact':>7}{'within1':>8}"
             f"{'prec':>7}{'recall':>7}{'F1':>7}  tp/fp/fn"]
    if report.get("source") == "image_labels":
        d = report["derivation"]
        lines.insert(1, f"  labels derived from image labels: {d['events_derived']} fully labelled events "
                        f"({d['events_incomplete']} partly labelled skipped); objects = max over frames")
    for field, info in report["fields"].items():
        if info["status"] != "ok":
            lines.append(f"{field:<27}  {info['status'].replace('_', ' ')}")
            continue
        pr = info["presence"]
        lines.append(f"{field:<27}{_cell(info['n'], 5)}"
                     f"{_cell(info['prediction_blank'] + info['prediction_invalid'], 7)}"
                     f"{_cell(info['mae'], 8)}{_cell(info['bias'], 8)}{_cell(info['exact'], 7)}"
                     f"{_cell(info.get('within1'), 8)}{_cell(pr['precision'], 7)}{_cell(pr['recall'], 7)}"
                     f"{_cell(pr['f1'], 7)}  {pr['tp']}/{pr['fp']}/{pr['fn']}")
    for name, group in report["groups"].items():
        if group["status"] != "ok":
            lines.append(f"group {name}: {group['status'].replace('_', ' ')} ({', '.join(group['missing'])})")
            continue
        text = (f"group {name}: n {group['n']}, bucket MAE {_cell(group['bucket_mae'], 0)}, "
                f"per-image L1 {_cell(group['bucket_l1'], 0)}, exact {_cell(group['exact'], 0)}")
        if "children_presence" in group:
            c = group["children_presence"]
            text += (f"; children presence F1 {_cell(c['f1'], 0)} (P {_cell(c['precision'], 0)}, "
                     f"R {_cell(c['recall'], 0)})")
        lines.append(text)
    if report.get("source") == "image_labels":
        b = report["derivation"]["people_unique_bounds"]
        lines.append(f"people_unique within labelled [max frame, frame sum]: n {b['n']}, within {b['within']} "
                     f"({_cell(b['within_rate'], 0)}), below {b['below']}, above {b['above']}, "
                     f"blank {b['prediction_blank']}, invalid {b['prediction_invalid']}")
    if report["invalid_label_cells"]:
        lines.append(f"invalid label cells skipped: {len(report['invalid_label_cells'])} (listed in JSON)")
    return "\n".join(lines)


def _same_file(a, b):
    try:
        return Path(a).exists() and Path(b).exists() and Path(a).samefile(b)
    except OSError:
        return False


def write_template(export_path, out_path, force=False):
    """Blank labels CSV with one row per exported image (or event). Returns (rows, kind)."""
    export_path, out_path = Path(export_path), Path(out_path)
    columns, rows = read_csv(export_path)
    if _same_file(out_path, export_path):
        raise ValueError("template output must differ from the export CSV")
    if out_path.exists() and not force:
        raise FileExistsError(f"refusing to overwrite existing file (use --force): {out_path}")
    if "relative_path" in columns:
        kind, keys = "image", ["relative_path"]
        fields = keys + list(LABEL_FIELDS)
    elif "event_id" in columns:
        kind, keys = "event", ["event_id"] + [c for c in EVENT_CONTEXT if c in columns]
        fields = keys + list(EVENT_LABEL_FIELDS)
    else:
        raise ValueError("export CSV needs a relative_path or event_id column")
    out = [{k: unescape_cell(row.get(k)) for k in keys} for row in rows if unescape_cell(row.get(keys[0])).strip()]
    write_csv(out_path, out, fields + ["notes"])
    return len(out), kind


def build_parser():
    parser = argparse.ArgumentParser(
        prog="python -m trailcam.evaluate",
        description="Score manual labels against a trailcam images CSV (and optionally events CSV).")
    parser.add_argument("--labels", help="Labels CSV: image rows keyed by relative_path/filename, "
                                         "or event rows keyed by event_id (scored against --events)")
    parser.add_argument("--export", help="Images CSV export to score image labels against")
    parser.add_argument("--events", help="Events CSV export: scored against event labels, or against event "
                                         "labels derived from fully labelled events in the image labels")
    parser.add_argument("--event-labels", help="Event labels CSV (event_id column) when --labels holds image labels")
    parser.add_argument("--prefix", default="", help="Prediction column prefix in the images CSV "
                                                     "(default none; combined_ for v1 exports)")
    parser.add_argument("--json", help="Also write the full report as JSON")
    parser.add_argument("--template", help="Write a blank labels template for this images or events CSV")
    parser.add_argument("--out", help="Template output path (with --template)")
    parser.add_argument("--force", action="store_true", help="Let --out overwrite an existing file")
    return parser


def run(args):
    """Load files named by parsed arguments and return {'images': report, 'events': report}."""
    image_labels = event_labels = None
    if args.labels:
        table = read_csv(args.labels)
        if label_kind(table[0]) == "event":
            event_labels = table
        else:
            image_labels = table
    if args.event_labels:
        if event_labels is not None:
            raise ValueError("--labels already holds event labels; pass image labels there or omit --event-labels")
        event_labels = read_csv(args.event_labels)
        if "event_id" not in event_labels[0]:
            raise ValueError("--event-labels needs an event_id column")
    if image_labels is not None and not args.export:
        raise ValueError("image-level labels need --export")
    if event_labels is not None and not args.events:
        raise ValueError("event-level labels (event_id column) need --events")
    if args.export and image_labels is None:
        raise ValueError("--export given but no image-level labels (relative_path column)")
    reports = {}
    if image_labels is not None:
        image_columns, image_rows = read_csv(args.export)
        reports["images"] = evaluate(image_labels[1], image_rows, "image", args.prefix, image_labels[0],
                                     image_columns)
    if args.events:
        columns, rows = read_csv(args.events)
        if event_labels is not None:
            reports["events"] = evaluate(event_labels[1], rows, "event", "", event_labels[0], columns)
        elif image_labels is None:
            raise ValueError("--events needs event labels (event_id column) or image labels")
        elif "event_id" not in image_columns:
            raise ValueError("--events with image labels needs an images CSV with an event_id column (v2 export)")
        else:
            reports["events"] = evaluate_derived_events(image_labels[1], image_rows, rows, columns)
    return reports


def _emit(text, stream=None):
    """Write a line without failing on consoles/pipes that cannot encode it (e.g. cp1252)."""
    stream = stream or sys.stdout
    encoding = getattr(stream, "encoding", None) or "utf-8"
    stream.write(text.encode(encoding, "backslashreplace").decode(encoding) + "\n")


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.template:
        if not args.out:
            parser.error("--template requires --out")
        if any((args.labels, args.export, args.events, args.event_labels, args.json)):
            parser.error("--template cannot be combined with scoring options")
    elif args.out or args.force:
        parser.error("--out and --force are only used with --template")
    elif not (args.labels or args.event_labels):
        parser.error("--labels (or --event-labels) is required, or use --template")
    try:
        if args.template:
            count, kind = write_template(args.template, args.out, args.force)
            _emit(f"Wrote blank {kind} labels template with {count} rows: {args.out}")
            return 0
        if args.json and any(_same_file(args.json, p) for p in
                             (args.labels, args.event_labels, args.export, args.events) if p):
            raise ValueError("--json must not overwrite an input file")
        reports = run(args)
        _emit("\n\n".join(format_report(r) for r in reports.values()))
        if args.json:
            Path(args.json).write_text(json.dumps(reports, indent=2, ensure_ascii=False, allow_nan=False),
                                       encoding="utf-8")
        return 0
    except (OSError, ValueError, csv.Error) as exc:
        _emit(f"error: {exc}", sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
