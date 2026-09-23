"""Synthetic label/export tables; these check scoring semantics, not model accuracy."""
from contextlib import redirect_stderr, redirect_stdout
import csv
import io
import json
import math
from pathlib import Path
import random
import tempfile
import time
import unittest

from trailcam.evaluate import (DIRECTIONS, EVENT_LABEL_FIELDS, LABEL_FIELDS, MAX_OVER_FRAMES, POLICY, _r,
                               bare_name, count_metrics, derive_event_labels, evaluate, evaluate_derived_events,
                               format_report, group_metrics, label_kind, main, match_rows, normalize_path,
                               parse_count, prediction_candidates, presence_metrics, read_csv, unique_bounds,
                               write_template)

PREDICTED = ["people_total", "adults", "children", "age_unknown"] + [f"dir_{d}" for d in DIRECTIONS] + [
    "bicycles", "strollers", "motorcycles", "atv_utv", "other_vehicles", "dogs", "backpacks", "large_bags",
    "large_bags_uncertain"]
V2_COLUMNS = ["image_id", "relative_path", "status"] + PREDICTED
LABEL_COLUMNS = ["relative_path"] + list(LABEL_FIELDS) + ["notes"]


def export_row(path, status="ok", **counts):
    row = dict.fromkeys(V2_COLUMNS, "")
    row.update(relative_path=path, status=status, image_id="img_" + path)
    row.update({k: str(v) for k, v in counts.items()})
    return row


def label_row(path, **counts):
    row = dict.fromkeys(LABEL_COLUMNS, "")
    row["relative_path"] = path
    row.update({k: str(v) for k, v in counts.items()})
    return row


def score(labels, exports, **kwargs):
    kwargs.setdefault("label_columns", LABEL_COLUMNS)
    kwargs.setdefault("prediction_columns", V2_COLUMNS)
    return evaluate(labels, exports, **kwargs)


def write(path, columns, rows):
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return str(path)


def plain_json(value):
    if isinstance(value, dict):
        return all(isinstance(k, str) and plain_json(v) for k, v in value.items())
    if isinstance(value, list):
        return all(plain_json(v) for v in value)
    return value is None or type(value) in (str, int, float, bool)


def cli(argv):
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = main(argv)
    return code, out.getvalue(), err.getvalue()


class ParseTests(unittest.TestCase):
    def test_blank_is_not_labelled(self):
        for value in (None, "", "   ", "\t"):
            self.assertIsNone(parse_count(value))

    def test_integral_values(self):
        self.assertEqual([parse_count(v) for v in ("0", "3", " 4 ", "2.0", "1e1", "-0")], [0, 3, 4, 2, 10, 0])

    def test_rejects_non_counts(self):
        for value in ("2.5", "-1", "-2.0", "abc", "nan", "inf", "True", "1,000", "5+"):
            with self.assertRaises(ValueError, msg=value):
                parse_count(value)

    def test_path_normalisation(self):
        self.assertEqual(normalize_path(" .\\cam1\\IMG_1.JPG "), "cam1/IMG_1.JPG")
        self.assertEqual(normalize_path("'=odd.jpg"), "=odd.jpg")
        self.assertEqual(normalize_path("'plain.jpg"), "'plain.jpg")
        self.assertEqual(normalize_path(None), "")
        self.assertEqual(bare_name("A/B/Img_2.JPG"), "img_2.jpg")

    def test_label_kind(self):
        self.assertEqual(label_kind(["relative_path", "event_id"]), "image")
        self.assertEqual(label_kind(["filename"]), "image")
        self.assertEqual(label_kind(["event_id", "people_unique"]), "event")
        with self.assertRaises(ValueError):
            label_kind(["people_total"])

    def test_prediction_candidates(self):
        self.assertEqual(prediction_candidates("age_unclear"), ["age_unknown"])
        self.assertEqual(prediction_candidates("dir_left", "combined_"),
                         ["combined_dir_left", "combined_direction_left"])
        self.assertEqual(prediction_candidates("dogs", "combined"), ["combineddogs", "combined_dogs"])


class MetricTests(unittest.TestCase):
    def test_count_metrics_known_values(self):
        m = count_metrics([(0, 0), (1, 2), (3, 1), (2, 2)], within1=True)
        self.assertEqual((m["n"], m["label_sum"], m["prediction_sum"]), (4, 6, 5))
        self.assertEqual((m["mae"], m["bias"], m["exact"], m["within1"]), (0.75, -0.25, 0.5, 0.75))
        self.assertEqual(m["presence"], {"tp": 3, "fp": 0, "fn": 0, "tn": 1,
                                         "precision": 1.0, "recall": 1.0, "f1": 1.0})

    def test_within1_only_when_requested(self):
        self.assertNotIn("within1", count_metrics([(1, 1)]))

    def test_empty_metrics_are_undefined_not_zero(self):
        m = count_metrics([], within1=True)
        self.assertEqual(m["n"], 0)
        self.assertIsNone(m["mae"])
        self.assertIsNone(m["within1"])
        self.assertIsNone(m["presence"]["f1"])

    def test_presence_undefined_without_positives(self):
        p = presence_metrics([(False, False)] * 3)
        self.assertEqual((p["tn"], p["precision"], p["recall"], p["f1"]), (3, None, None, None))

    def test_presence_missed_and_false_positives(self):
        p = presence_metrics([(True, False), (False, False)])
        self.assertEqual((p["precision"], p["recall"], p["f1"]), (None, 0.0, 0.0))
        p = presence_metrics([(False, True), (True, True), (True, False)])
        self.assertEqual((p["tp"], p["fp"], p["fn"], p["precision"], p["recall"], p["f1"]),
                         (1, 1, 1, 0.5, 0.5, 0.5))

    def test_group_metrics(self):
        g = group_metrics([([1, 0, 0], [1, 0, 0]), ([1, 1, 0], [2, 0, 0])])
        self.assertEqual(g, {"n": 2, "bucket_mae": 0.3333, "bucket_l1": 1.0, "exact": 0.5})
        self.assertEqual(group_metrics([]), {"n": 0, "bucket_mae": None, "bucket_l1": None, "exact": None})

    def test_rounding_has_no_negative_zero(self):
        value = _r(-0.00001)
        self.assertEqual(math.copysign(1, value), 1.0)
        self.assertEqual(count_metrics([(2, 2)])["bias"], 0.0)
        self.assertIsNone(_r(None))


class MatchTests(unittest.TestCase):
    def setUp(self):
        self.exports = [export_row(p) for p in
                        ("cam1/IMG_0001.JPG", "cam2/IMG_0001.JPG", "cam1/img_0002.jpg", "'=odd.jpg", "cam3/x.jpg")]

    def test_path_then_case_insensitive_then_filename(self):
        labels = [label_row(k) for k in ("cam1\\IMG_0001.JPG", "./CAM2/img_0001.jpg", "IMG_0002.JPG",
                                          "img_0001.jpg", "missing.jpg", "", "=odd.jpg", "cam1/img_0002.jpg")]
        pairs, m = match_rows(labels, self.exports)
        # Pairs keep label order; the exact label for cam1/img_0002.jpg wins over the earlier bare filename.
        self.assertEqual([p[2]["relative_path"] for p in pairs],
                         ["cam1/IMG_0001.JPG", "cam2/IMG_0001.JPG", "'=odd.jpg", "cam1/img_0002.jpg"])
        self.assertEqual(m["matched_by"], {"path": 4, "suffix": 0, "filename": 0})
        self.assertEqual((m["label_rows"], m["matched"], m["missing_key"]), (8, 4, 1))
        self.assertEqual(m["unmatched"], ["missing.jpg"])
        self.assertEqual(m["ambiguous"], ["img_0001.jpg"])
        self.assertEqual(m["duplicate"], ["IMG_0002.JPG"])
        self.assertEqual((m["prediction_rows"], m["unlabelled_prediction_rows"]), (5, 1))

    def test_bare_filename_falls_back_to_unique_name(self):
        pairs, m = match_rows([label_row("X.JPG")], self.exports)
        self.assertEqual(pairs[0][2]["relative_path"], "cam3/x.jpg")
        self.assertEqual(m["matched_by"]["filename"], 1)

    def test_label_naming_another_folder_never_matches(self):
        # Trail cameras reuse names: cam9's IMAG0001 must not be scored against cam1's.
        pairs, m = match_rows([label_row("renamed_folder/X.JPG"), label_row("cam9/IMG_0001.JPG")], self.exports)
        self.assertEqual((pairs, m["unmatched"]), ([], ["renamed_folder/X.JPG", "cam9/IMG_0001.JPG"]))

    def test_filename_column_and_non_ascii_names(self):
        exports = [export_row("מצלמה/תמונה_1.JPG")]
        pairs, m = match_rows([{"filename": "תמונה_1.jpg", "people_total": "1"}], exports)
        self.assertEqual(m["matched"], 1)

    def test_error_rows_are_counted(self):
        exports = [export_row("a.jpg", status="error"), export_row("b.jpg")]
        _, m = match_rows([label_row("a.jpg"), label_row("b.jpg")], exports)
        self.assertEqual(m["prediction_error_rows"], 1)

    def test_event_ids_match_exactly(self):
        events = [{"event_id": "cam1__2026-05-01T10-00-00__1"}, {"event_id": "cam1__2026-05-01T11-00-00__2"}]
        labels = [{"event_id": " cam1__2026-05-01T10-00-00__1 "}, {"event_id": "CAM1__2026-05-01T11-00-00__2"},
                  {"event_id": ""}]
        pairs, m = match_rows(labels, events, "event")
        self.assertEqual(len(pairs), 1)
        self.assertEqual(m["matched_by"], {"event_id": 1})
        self.assertEqual((len(m["unmatched"]), m["missing_key"]), (1, 1))


class EvaluateTests(unittest.TestCase):
    def test_blank_labels_skipped_and_missing_predictions_reported(self):
        labels = [label_row("a.jpg", people_total=2), label_row("b.jpg"),
                  label_row("c.jpg", people_total=1), label_row("d.jpg", people_total=0)]
        exports = [export_row("a.jpg", people_total=3), export_row("b.jpg", people_total=5),
                   export_row("c.jpg", status="error"), export_row("d.jpg", people_total="x")]
        f = score(labels, exports)["fields"]["people_total"]
        self.assertEqual((f["labelled"], f["n"], f["prediction_blank"], f["prediction_invalid"]), (3, 1, 1, 1))
        self.assertEqual((f["mae"], f["bias"], f["exact"], f["within1"]), (1.0, 1.0, 0.0, 1.0))

    def test_zero_prediction_is_scored_but_blank_is_not(self):
        labels = [label_row("a.jpg", dogs=1), label_row("b.jpg", dogs=1)]
        exports = [export_row("a.jpg", dogs=0), export_row("b.jpg")]
        f = score(labels, exports)["fields"]["dogs"]
        self.assertEqual((f["n"], f["prediction_blank"], f["presence"]["fn"]), (1, 1, 1))

    def test_backpacks_scored_independently_of_bag_size(self):
        labels = [label_row("a.jpg", backpacks=2, large_bags=0),
                  label_row("b.jpg", backpacks=0, large_bags=1),
                  label_row("c.jpg", backpacks=1)]
        exports = [export_row("a.jpg", backpacks=1), export_row("b.jpg", backpacks=1),
                   export_row("c.jpg")]
        f = score(labels, exports)["fields"]["backpacks"]
        self.assertEqual((f["n"], f["prediction_blank"], f["mae"]), (2, 1, 1.0))
        self.assertEqual((f["presence"]["tp"], f["presence"]["fp"], f["presence"]["fn"]), (1, 1, 0))
        self.assertEqual(score(labels, exports)["fields"]["large_bags"]["n"], 0)

    def test_backpacks_support_v1_and_event_exports(self):
        v1 = evaluate([{"relative_path": "a.jpg", "backpacks": "2"}],
                      [{"relative_path": "a.jpg", "combined_backpacks": "1"}], prefix="combined_")
        self.assertEqual(v1["fields"]["backpacks"]["mae"], 1.0)
        events = evaluate([{"event_id": "e1", "backpacks": "2"}],
                          [{"event_id": "e1", "backpacks": "2"}], level="event")
        self.assertEqual(events["fields"]["backpacks"]["exact"], 1.0)

    def test_age_unclear_scores_against_age_unknown(self):
        report = score([label_row("a.jpg", age_unclear=2)], [export_row("a.jpg", age_unknown=2)])
        f = report["fields"]["age_unclear"]
        self.assertEqual((f["prediction_columns"], f["n"], f["exact"]), (["age_unknown"], 1, 1.0))

    def test_label_file_may_name_age_unknown(self):
        labels = [{"relative_path": "a.jpg", "age_unknown": "1"}]
        f = evaluate(labels, [export_row("a.jpg", age_unknown=0)], prediction_columns=V2_COLUMNS)
        self.assertEqual(f["fields"]["age_unclear"]["label_column"], "age_unknown")
        self.assertEqual(f["fields"]["age_unclear"]["mae"], 1.0)

    def test_v1_prefix_with_direction_aliases(self):
        columns = ["relative_path", "status", "combined_people_total", "combined_adults", "combined_children",
                   "combined_age_unknown", "combined_dogs"] + [
            f"combined_direction_{d}" for d in ("left", "right", "toward", "away", "unclear")]
        row = dict.fromkeys(columns, "0")
        row.update(relative_path="a.jpg", status="ok", combined_people_total="2", combined_age_unknown="2",
                   combined_direction_left="1", combined_direction_unclear="1")
        labels = [label_row("a.jpg", people_total=2, adults=1, children=1, age_unclear=0, dir_left=2,
                            dir_right=0, dir_toward=0, dir_away=0, dir_stationary=0, dir_unclear=0)]
        for prefix in ("combined_", "combined"):
            report = score(labels, [row], prefix=prefix, prediction_columns=columns)
            fields = report["fields"]
            self.assertEqual(fields["dir_left"]["prediction_columns"], ["combined_direction_left"])
            self.assertEqual((fields["dir_left"]["mae"], fields["people_total"]["exact"]), (1.0, 1.0))
            self.assertEqual(fields["dir_stationary"]["status"], "missing_prediction_column")
            self.assertEqual(fields["large_bags"]["status"], "missing_prediction_column")
            self.assertNotIn("large_bags_with_uncertain", fields)
            self.assertEqual(report["groups"]["direction"]["missing"], ["dir_stationary"])
            age = report["groups"]["age"]
            self.assertEqual((age["status"], age["n"], age["bucket_l1"]), ("ok", 1, 4.0))
            self.assertEqual(age["children_presence"]["fn"], 1)

    def test_v2_name_preferred_over_v1_alias(self):
        columns = V2_COLUMNS + ["direction_left"]
        row = export_row("a.jpg", dir_left=1)
        row["direction_left"] = "9"
        f = score([label_row("a.jpg", dir_left=1)], [row], prediction_columns=columns)["fields"]["dir_left"]
        self.assertEqual((f["prediction_columns"], f["exact"]), (["dir_left"], 1.0))

    def test_empty_prefix_does_not_pick_prefixed_columns(self):
        report = evaluate([label_row("a.jpg", dogs=1)], [{"relative_path": "a.jpg", "combined_dogs": "1"}])
        self.assertEqual(report["fields"]["dogs"]["status"], "missing_prediction_column")

    def test_age_group_and_children_presence(self):
        labels = [label_row("a.jpg", adults=2, children=1, age_unclear=0),
                  label_row("b.jpg", adults=1, children=0, age_unclear=0),
                  label_row("c.jpg", adults=1, children=1),  # incomplete group label
                  label_row("d.jpg", adults=0, children=2, age_unclear=0)]
        exports = [export_row("a.jpg", adults=2, children=0, age_unknown=1),
                   export_row("b.jpg", adults=1, children=0, age_unknown=0),
                   export_row("c.jpg", adults=1, children=1, age_unknown=0),
                   export_row("d.jpg", adults=0, children=1)]  # prediction bucket blank
        group = score(labels, exports)["groups"]["age"]
        self.assertEqual((group["labelled"], group["prediction_unavailable"], group["n"]), (3, 1, 2))
        self.assertEqual((group["bucket_mae"], group["bucket_l1"], group["exact"]), (0.3333, 1.0, 0.5))
        self.assertEqual(group["children_presence"]["fn"], 1)
        self.assertEqual(group["children_presence"]["recall"], 0.0)

    def test_direction_group(self):
        dirs = {f"dir_{d}": 0 for d in DIRECTIONS}
        labels = [label_row("a.jpg", **{**dirs, "dir_left": 1, "dir_stationary": 1})]
        exports = [export_row("a.jpg", **{**dirs, "dir_left": 1, "dir_unclear": 1})]
        group = score(labels, exports)["groups"]["direction"]
        self.assertEqual((group["n"], group["bucket_l1"], group["exact"]), (1, 2.0, 0.0))
        self.assertNotIn("children_presence", group)

    def test_uncertain_bags_derived_field(self):
        labels = [label_row("a.jpg", large_bags=1), label_row("b.jpg", large_bags=0)]
        exports = [export_row("a.jpg", large_bags=0, large_bags_uncertain=1),
                   export_row("b.jpg", large_bags=0, large_bags_uncertain="")]
        fields = score(labels, exports)["fields"]
        self.assertEqual((fields["large_bags"]["n"], fields["large_bags"]["exact"]), (2, 0.5))
        derived = fields["large_bags_with_uncertain"]
        self.assertEqual(derived["prediction_columns"], ["large_bags", "large_bags_uncertain"])
        self.assertEqual((derived["n"], derived["exact"], derived["prediction_blank"]), (1, 1.0, 1))

    def test_invalid_label_cells_reported_once_and_skipped(self):
        labels = [label_row("a.jpg", people_total="two", large_bags="-1", dogs=1)]
        report = score(labels, [export_row("a.jpg", people_total=2, large_bags=0, large_bags_uncertain=0, dogs=1)])
        self.assertEqual(sorted(c["column"] for c in report["invalid_label_cells"]), ["large_bags", "people_total"])
        self.assertEqual(report["fields"]["people_total"]["label_invalid"], 1)
        self.assertEqual(report["fields"]["people_total"]["n"], 0)
        self.assertEqual(report["fields"]["dogs"]["exact"], 1.0)

    def test_missing_label_column(self):
        report = evaluate([{"relative_path": "a.jpg", "people_total": "1"}], [export_row("a.jpg", people_total=1)],
                          prediction_columns=V2_COLUMNS)
        self.assertEqual(report["fields"]["dogs"]["status"], "missing_label_column")
        self.assertNotIn("labelled", report["fields"]["dogs"])
        self.assertEqual(report["groups"]["age"]["status"], "missing_label_column")
        self.assertIn("dogs", format_report(report))

    def test_largest_errors_sorted_and_capped(self):
        count = POLICY["largest_errors"] * 2
        labels = [label_row(f"{i:03}.jpg", people_total=0) for i in range(count)]
        exports = [export_row(f"{i:03}.jpg", people_total=i % 4) for i in range(count)]
        errors = score(labels, exports)["largest_errors"]
        self.assertEqual(len(errors), POLICY["largest_errors"])
        self.assertEqual(errors[0], {"key": "003.jpg", "label": 0, "prediction": 3, "error": 3})
        self.assertEqual([e["error"] for e in errors], sorted((e["error"] for e in errors), reverse=True))
        self.assertTrue(all(e["error"] for e in errors))

    def test_empty_inputs(self):
        report = evaluate([], [], label_columns=LABEL_COLUMNS, prediction_columns=V2_COLUMNS)
        self.assertEqual(report["matching"]["matched"], 0)
        self.assertEqual(report["fields"]["people_total"]["n"], 0)
        self.assertIsNone(report["fields"]["people_total"]["mae"])
        self.assertEqual(report["groups"]["age"]["n"], 0)
        self.assertEqual(report["largest_errors"], [])
        bare = evaluate([], [])
        self.assertEqual(bare["fields"]["people_total"]["status"], "missing_label_column")
        for r in (report, bare):
            json.dumps(r, allow_nan=False)
            self.assertIn("Image-level", format_report(r))

    def test_unknown_level(self):
        with self.assertRaises(ValueError):
            evaluate([], [], level="video")

    def test_deterministic_plain_json(self):
        labels = [label_row(f"{i}.jpg", people_total=i % 3, adults=i % 2, children=0, age_unclear=1,
                            dogs=i % 2) for i in range(12)]
        exports = [export_row(f"{i}.jpg", people_total=(i * 7) % 4, adults=1, children=i % 2, age_unknown=0,
                              dogs=0) for i in range(12)]
        first, second = score(labels, exports), score(list(labels), list(exports))
        self.assertEqual(json.dumps(first, sort_keys=True), json.dumps(second, sort_keys=True))
        self.assertTrue(plain_json(first))
        json.dumps(first, allow_nan=False)

    def test_event_level(self):
        columns = ["event_id", "people_unique", "people_max_frame", "adults", "children", "age_unknown"]
        events = [{"event_id": "e1", "people_unique": "3", "people_max_frame": "2", "adults": "2",
                   "children": "1", "age_unknown": "0"},
                  {"event_id": "e2", "people_unique": "1", "people_max_frame": "1", "adults": "0",
                   "children": "0", "age_unknown": "1"}]
        labels = [{"event_id": "e1", "people_unique": "2", "adults": "2", "children": "0", "age_unclear": "0"},
                  {"event_id": "e2", "people_unique": "1", "adults": "1", "children": "0", "age_unclear": "0"}]
        report = evaluate(labels, events, level="event", prediction_columns=columns)
        f = report["fields"]
        self.assertNotIn("people_total", f)
        self.assertEqual(list(f)[:len(EVENT_LABEL_FIELDS)], list(EVENT_LABEL_FIELDS))
        self.assertEqual((f["people_unique"]["mae"], f["people_unique"]["within1"]), (0.5, 1.0))
        self.assertEqual(report["groups"]["age"]["n"], 2)
        self.assertEqual(report["largest_errors"], [{"key": "e1", "label": 2, "prediction": 3, "error": 1}])
        self.assertIn("Event-level", format_report(report))


class CliTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.export = write(self.dir / "export.csv", V2_COLUMNS,
                            [export_row("cam1/a.jpg", people_total=2, adults=2, children=0, age_unknown=0),
                             export_row("cam1/b.jpg", people_total=0),
                             export_row("=odd.jpg", people_total=1)])
        # write() does not add the spreadsheet apostrophe; emulate the real export for the formula row.
        text = Path(self.export).read_text(encoding="utf-8-sig").replace("=odd.jpg", "'=odd.jpg")
        Path(self.export).write_text(text, encoding="utf-8-sig")
        self.labels = write(self.dir / "labels.csv", LABEL_COLUMNS,
                            [label_row("CAM1/A.JPG", people_total=3, adults=2, children=1, age_unclear=0),
                             label_row("b.jpg", people_total=0), label_row("zzz.jpg", people_total=1)])
        self.events = write(self.dir / "events.csv", ["event_id", "camera_id", "start", "people_unique"],
                            [{"event_id": "cam1__1", "camera_id": "cam1", "start": "2026-05-01T10:00:00",
                              "people_unique": "2"}])
        self.event_labels = write(self.dir / "event_labels.csv", ["event_id", "people_unique"],
                                  [{"event_id": "cam1__1", "people_unique": "3"}])

    def tearDown(self):
        self._tmp.cleanup()

    def test_read_csv_handles_bom_and_header_spaces(self):
        path = self.dir / "spaced.csv"
        path.write_text(" relative_path , people_total\na.jpg,1\n", encoding="utf-8-sig")
        columns, rows = read_csv(path)
        self.assertEqual(columns, ["relative_path", "people_total"])
        self.assertEqual(rows, [{"relative_path": "a.jpg", "people_total": "1"}])

    def test_scores_prints_table_and_writes_json(self):
        out_json = self.dir / "out.json"
        code, out, err = cli(["--labels", self.labels, "--export", self.export, "--json", str(out_json)])
        self.assertEqual((code, err), (0, ""))
        self.assertIn("Image-level evaluation", out)
        self.assertIn("people_total", out)
        self.assertIn("unmatched 1", out)
        report = json.loads(out_json.read_text(encoding="utf-8"))
        self.assertEqual(list(report), ["images"])
        self.assertEqual(report["images"]["matching"]["matched_by"], {"path": 1, "suffix": 0, "filename": 1})
        self.assertEqual(report["images"]["fields"]["people_total"]["mae"], 0.5)
        self.assertEqual(report["images"]["groups"]["age"]["children_presence"]["fn"], 1)

    def test_images_and_events_together(self):
        code, out, _ = cli(["--labels", self.labels, "--export", self.export,
                            "--events", self.events, "--event-labels", self.event_labels])
        self.assertEqual(code, 0)
        self.assertIn("Image-level", out)
        self.assertIn("Event-level", out)

    def test_event_labels_via_labels_flag(self):
        out_json = self.dir / "events.json"
        code, _, _ = cli(["--labels", self.event_labels, "--events", self.events, "--json", str(out_json)])
        self.assertEqual(code, 0)
        report = json.loads(out_json.read_text(encoding="utf-8"))
        self.assertEqual(list(report), ["events"])
        self.assertEqual(report["events"]["fields"]["people_unique"]["bias"], -1.0)

    def test_input_errors_return_2(self):
        keyless = write(self.dir / "keyless.csv", ["people_total"], [{"people_total": "1"}])
        cases = [["--labels", keyless, "--export", self.export],
                 ["--labels", self.labels],
                 ["--labels", self.labels, "--export", self.export, "--events", self.events],
                 ["--labels", self.event_labels, "--export", self.export],
                 ["--labels", self.event_labels, "--events", self.events, "--event-labels", self.event_labels],
                 ["--labels", self.labels, "--export", self.export, "--event-labels", self.labels,
                  "--events", self.events],
                 ["--labels", str(self.dir / "absent.csv"), "--export", self.export],
                 ["--labels", self.labels, "--export", self.export, "--json", self.labels]]
        for argv in cases:
            code, out, err = cli(argv)
            self.assertEqual(code, 2, argv)
            self.assertTrue(err.startswith("error:"), argv)
        self.assertIn("relative_path", Path(self.labels).read_text(encoding="utf-8-sig"))

    def test_argument_errors_exit(self):
        for argv in ([], ["--template", self.export], ["--out", "x.csv", "--labels", self.labels],
                     ["--template", self.export, "--out", "t.csv", "--labels", self.labels]):
            with self.assertRaises(SystemExit, msg=argv), redirect_stderr(io.StringIO()):
                main(argv)

    def test_image_template(self):
        out = self.dir / "template.csv"
        code, printed, _ = cli(["--template", self.export, "--out", str(out)])
        self.assertEqual(code, 0)
        self.assertIn("3 rows", printed)
        self.assertTrue(out.read_bytes().startswith(b"\xef\xbb\xbf"))
        columns, rows = read_csv(out)
        self.assertEqual(columns, LABEL_COLUMNS)
        self.assertIn("backpacks", columns)
        self.assertEqual([r["relative_path"] for r in rows], ["cam1/a.jpg", "cam1/b.jpg", "'=odd.jpg"])
        self.assertTrue(all(r[f] == "" for r in rows for f in LABEL_FIELDS))
        # A filled-in template scores against its own export, formula-like names included.
        rows[2]["people_total"] = "1"
        report = evaluate(rows, read_csv(self.export)[1], label_columns=columns, prediction_columns=V2_COLUMNS)
        self.assertEqual((report["matching"]["matched"], report["fields"]["people_total"]["exact"]), (3, 1.0))

    def test_template_refuses_to_overwrite(self):
        out = self.dir / "template.csv"
        out.write_text("manual labels", encoding="utf-8")
        code, _, err = cli(["--template", self.export, "--out", str(out)])
        self.assertEqual(code, 2)
        self.assertIn("--force", err)
        self.assertEqual(out.read_text(encoding="utf-8"), "manual labels")
        self.assertEqual(cli(["--template", self.export, "--out", str(out), "--force"])[0], 0)
        before = Path(self.export).read_bytes()
        self.assertEqual(cli(["--template", self.export, "--out", self.export, "--force"])[0], 2)
        self.assertEqual(Path(self.export).read_bytes(), before)

    def test_event_template(self):
        count, kind = write_template(self.events, self.dir / "event_template.csv")
        self.assertEqual((count, kind), (1, "event"))
        columns, rows = read_csv(self.dir / "event_template.csv")
        self.assertEqual(columns, ["event_id", "camera_id", "start"] + list(EVENT_LABEL_FIELDS) + ["notes"])
        self.assertEqual((rows[0]["event_id"], rows[0]["people_unique"]), ("cam1__1", ""))
        self.assertEqual(label_kind(columns), "event")

    def test_template_rejects_unknown_export(self):
        other = write(self.dir / "other.csv", ["name"], [{"name": "x"}])
        with self.assertRaises(ValueError):
            write_template(other, self.dir / "t.csv")
        empty = self.dir / "empty.csv"
        empty.write_text("", encoding="utf-8")
        with self.assertRaises(ValueError):
            write_template(empty, self.dir / "t.csv")

    def test_header_only_files(self):
        labels = write(self.dir / "empty_labels.csv", LABEL_COLUMNS, [])
        export = write(self.dir / "empty_export.csv", V2_COLUMNS, [])
        code, out, _ = cli(["--labels", labels, "--export", export])
        self.assertEqual(code, 0)
        self.assertIn("matched 0", out)
        self.assertEqual(write_template(export, self.dir / "t.csv"), (0, "image"))


def event_image(path, event_id, **counts):
    row = export_row(path, **counts)
    row["event_id"] = event_id
    return row


class AdversarialParseTests(unittest.TestCase):
    def test_huge_counts_are_invalid_not_overflowing(self):
        limit = POLICY["max_count"]
        self.assertEqual(parse_count(str(limit)), limit)
        for value in (str(limit + 1), "1" + "0" * 400, "9" * 5000, "1e300", "1e16"):
            with self.assertRaises(ValueError, msg=value[:12]):
                parse_count(value)

    def test_non_string_and_odd_inputs(self):
        self.assertEqual((parse_count(3), parse_count(2.0), parse_count(0), parse_count("+3"), parse_count("0.0")),
                         (3, 2, 0, 3, 0))
        for value in (float("nan"), float("inf"), -1, 1.5, True, [1], "1\x00", "\u200b", "0x1", "#N/A", "?"):
            with self.assertRaises(ValueError, msg=ascii(value)):
                parse_count(value)


class AdversarialMatchTests(unittest.TestCase):
    def test_exact_label_wins_whatever_the_label_order(self):
        exports = [export_row("cam1/IMAG0001.JPG", people_total=1)]
        cam2, cam1 = label_row("cam2/IMAG0001.JPG", people_total=5), label_row("cam1/IMAG0001.JPG", people_total=1)
        for labels in ([cam2, cam1], [cam1, cam2]):
            report = score(labels, exports)
            self.assertEqual(report["matching"]["unmatched"], ["cam2/IMAG0001.JPG"])
            self.assertEqual(report["matching"]["duplicate"], [])
            self.assertEqual(report["fields"]["people_total"]["exact"], 1.0)

    def test_bare_name_after_exact_label_is_duplicate(self):
        pairs, m = match_rows([label_row("a.jpg"), label_row("cam1/a.jpg")], [export_row("cam1/a.jpg")])
        self.assertEqual((m["matched_by"]["path"], m["duplicate"], pairs[0][0]), (1, ["a.jpg"], "cam1/a.jpg"))

    def test_bare_name_in_two_folders_is_ambiguous(self):
        exports = [export_row("cam1/a.jpg"), export_row("cam2/A.JPG")]
        pairs, m = match_rows([label_row("a.jpg"), label_row("cam2/a.jpg")], exports)
        self.assertEqual(m["ambiguous"], ["a.jpg"])
        self.assertEqual([p[2]["relative_path"] for p in pairs], ["cam2/A.JPG"])

    def test_suffix_matches_across_export_roots(self):
        exports = [export_row("site/cam1/a.jpg"), export_row("b.jpg")]
        labels = [label_row("cam1/a.jpg"), label_row("D:\\photos\\cam1\\B.JPG"), label_row("cam1//a.jpg"),
                  label_row("/"), label_row("cam2/a.jpg")]
        pairs, m = match_rows(labels, exports)
        self.assertEqual([p[2]["relative_path"] for p in pairs], ["site/cam1/a.jpg", "b.jpg"])
        self.assertEqual(m["matched_by"], {"path": 0, "suffix": 2, "filename": 0})
        self.assertEqual((m["duplicate"], m["unmatched"]), (["cam1//a.jpg"], ["/", "cam2/a.jpg"]))

    def test_unicode_normalisation(self):
        pairs, m = match_rows([label_row("caf\u00e9/IMG_1.JPG")], [export_row("cafe\u0301/IMG_1.JPG")])
        self.assertEqual(m["matched_by"]["path"], 1)

    def test_ragged_rows_and_none_values(self):
        labels = [{"relative_path": "a.jpg", "people_total": None, None: ["x", "y"]},
                  {"relative_path": None, "filename": "b.jpg", "people_total": "2"}]
        exports = [export_row("a.jpg", people_total=1), export_row("b.jpg", people_total=2),
                   {"relative_path": None, "people_total": "3"}]
        report = evaluate(labels, exports, prediction_columns=V2_COLUMNS)
        f = report["fields"]["people_total"]
        self.assertEqual((report["matching"]["matched"], f["labelled"], f["n"], f["exact"]), (2, 1, 1, 1.0))
        self.assertEqual(report["matching"]["unlabelled_prediction_rows"], 1)
        json.dumps(report, allow_nan=False)

    def test_generators_and_unknown_level(self):
        _, m = match_rows((r for r in [label_row("a.jpg")]), (r for r in [export_row("a.jpg")]))
        self.assertEqual(m["matched"], 1)
        with self.assertRaises(ValueError):
            match_rows([], [], "video")
        self.assertEqual(count_metrics(x for x in [(1, 1), (0, 2)])["n"], 2)
        self.assertEqual(presence_metrics(iter([(True, True), (False, True)]))["fp"], 1)
        self.assertEqual(group_metrics(iter([([1], [1])]))["exact"], 1.0)
        report = evaluate(iter([label_row("a.jpg", people_total=1)]), iter([export_row("a.jpg", people_total=1)]))
        self.assertEqual(report["fields"]["people_total"]["n"], 1)

    def test_event_id_edge_cases(self):
        _, m = match_rows([{"event_id": "e1"}], [{"event_id": "e1"}, {"event_id": " e1 "}], "event")
        self.assertEqual(m["ambiguous"], ["e1"])
        pairs, _ = match_rows([{"event_id": "-cam__1"}], [{"event_id": "'-cam__1"}], "event")
        self.assertEqual(len(pairs), 1)


class AdversarialEvaluateTests(unittest.TestCase):
    def test_huge_label_cell_is_listed_not_crashing(self):
        report = score([label_row("a.jpg", people_total="1" + "0" * 400, dogs="1e300")],
                       [export_row("a.jpg", people_total=1, dogs=0)])
        self.assertEqual(sorted(c["column"] for c in report["invalid_label_cells"]), ["dogs", "people_total"])
        json.dumps(report, allow_nan=False)
        format_report(report)

    def test_non_finite_and_fractional_predictions_are_invalid(self):
        values = ("nan", "inf", "-1", "1.5", "1" + "0" * 30, "1")
        f = score([label_row(f"{i}.jpg", people_total=1) for i in range(len(values))],
                  [export_row(f"{i}.jpg", people_total=v) for i, v in enumerate(values)])["fields"]["people_total"]
        self.assertEqual((f["prediction_invalid"], f["prediction_blank"], f["n"], f["exact"]), (5, 0, 1, 1.0))

    def test_extreme_valid_counts(self):
        top = POLICY["max_count"]
        f = score([label_row("a.jpg", people_total=0), label_row("b.jpg", people_total=top)],
                  [export_row("a.jpg", people_total=top), export_row("b.jpg", people_total=0)])["fields"]["people_total"]
        self.assertEqual((f["mae"], f["bias"], f["within1"]), (float(top), 0.0, 0.0))
        self.assertEqual(f["presence"]["f1"], 0.0)

    def test_repeated_columns_are_refused(self):
        with self.assertRaises(ValueError):  # csv.DictReader would keep only the last people_total
            score([label_row("a.jpg")], [export_row("a.jpg")], label_columns=LABEL_COLUMNS + ["people_total"])
        with self.assertRaises(ValueError):
            score([label_row("a.jpg")], [export_row("a.jpg")], prediction_columns=V2_COLUMNS + ["dogs"])
        # Repeats of unused or unnamed columns (spreadsheet padding) are harmless.
        score([label_row("a.jpg")], [export_row("a.jpg")], label_columns=LABEL_COLUMNS + ["notes", "", ""])

    def test_wrong_csv_kind_is_refused(self):
        with self.assertRaises(ValueError):  # events CSV passed as the images export
            evaluate([label_row("a.jpg")], [{"event_id": "e1", "people_unique": "1"}])
        with self.assertRaises(ValueError):  # images CSV passed as the events export
            evaluate([{"event_id": "e1"}], [event_image("a.jpg", "e1")], level="event")

    def test_event_people_max_frame_scored_only_when_labelled(self):
        events = [{"event_id": "e1", "people_unique": "2", "people_max_frame": "2"}]
        plain = evaluate([{"event_id": "e1", "people_unique": "2"}], events, level="event")
        self.assertNotIn("people_max_frame", plain["fields"])
        extra = evaluate([{"event_id": "e1", "people_unique": "2", "people_max_frame": "1"}], events, level="event")
        f = extra["fields"]["people_max_frame"]
        self.assertEqual((f["mae"], f["within1"]), (1.0, 1.0))

    def test_row_order_does_not_change_scores(self):
        rng = random.Random(7)
        labels = [label_row(f"cam{i % 3}/{i:04}.jpg", people_total=rng.randint(0, 4), adults=rng.randint(0, 2),
                            children=rng.randint(0, 1), age_unclear=0, dogs=rng.choice(["", "0", "1", "x"]))
                  for i in range(300)]
        exports = [export_row(f"cam{i % 3}/{i:04}.jpg", people_total=rng.choice(["", "0", "2", "3"]),
                              adults=rng.randint(0, 2), children=rng.randint(0, 1),
                              age_unknown=rng.choice(["", "0", "1"]), dogs=rng.randint(0, 1)) for i in range(300)]
        base = score(labels, exports)
        labels2, exports2 = labels[:], exports[:]
        rng.shuffle(labels2)
        rng.shuffle(exports2)
        other = score(labels2, exports2)
        for part in ("fields", "groups", "largest_errors"):
            self.assertEqual(json.dumps(base[part], sort_keys=True), json.dumps(other[part], sort_keys=True))
        self.assertTrue(plain_json(base))
        json.dumps(base, allow_nan=False)

    def test_large_input_is_fast(self):
        n = 20000
        labels = [label_row(f"cam{i % 9}/IMAG{i:05}.JPG" if i % 2 else f"IMAG{i:05}.JPG", people_total=i % 4,
                            dogs=i % 2) for i in range(n)]
        exports = [export_row(f"site/cam{i % 9}/IMAG{i:05}.JPG", people_total=i % 3, dogs=0) for i in range(n)]
        # Worst case for the fallback: one file name repeated in every folder.
        same_name = [export_row(f"cam{i}/IMAG0001.JPG") for i in range(n)]
        start = time.perf_counter()
        report = score(labels, exports)
        _, crowded = match_rows([label_row("IMAG0001.JPG")] * n, same_name)
        self.assertLess(time.perf_counter() - start, 8.0)
        self.assertEqual(report["matching"]["matched_by"], {"path": 0, "suffix": n // 2, "filename": n // 2})
        self.assertEqual(len(crowded["ambiguous"]), n)
        json.dumps(report, allow_nan=False)


class DerivedEventTests(unittest.TestCase):
    def setUp(self):
        self.images = [event_image(f"cam1/{i}.jpg", e) for i, e in enumerate(("e1", "e1", "e2", "e3", "e3", ""), 1)]
        self.labels = [label_row("cam1/1.jpg", people_total=2, dogs=1, bicycles=0),
                       label_row("cam1/2.jpg", people_total=3, dogs=0, bicycles=""),
                       label_row("cam1/3.jpg", people_total=0, dogs=0, bicycles=1),
                       label_row("cam1/4.jpg", people_total=1, dogs=1),  # e3 frame 5 is unlabelled
                       label_row("cam1/6.jpg", people_total=1)]  # not in any event

    def test_only_fully_labelled_events_and_fields(self):
        rows, info = derive_event_labels(self.labels, self.images)
        self.assertEqual(info, {"events_derived": 2, "events_incomplete": 1})
        self.assertEqual(rows, [{"event_id": "e1", "dogs": 1, "people_max_frame": 3, "people_unique_min": 3,
                                 "people_unique_max": 5},
                                {"event_id": "e2", "bicycles": 1, "dogs": 0, "people_max_frame": 0,
                                 "people_unique_min": 0, "people_unique_max": 0}])

    def test_events_csv_image_count_reveals_missing_frames(self):
        rows, info = derive_event_labels(self.labels, self.images,
                                         [{"event_id": "e2", "image_count": "2"}, {"event_id": "e1", "image_count": "x"},
                                          {"event_id": "e7", "image_count": "3"}])
        self.assertEqual(([r["event_id"] for r in rows], info["events_incomplete"]), (["e1"], 2))
        self.assertEqual(derive_event_labels([], [], []), ([], {"events_derived": 0, "events_incomplete": 0}))

    def test_unique_bounds(self):
        derived = [{"event_id": f"e{i}", "people_unique_min": 2, "people_unique_max": 4} for i in range(6)]
        events = [{"event_id": f"e{i}", "people_unique": v} for i, v in enumerate(("1", "2", "4", "5", "", "x"))]
        derived.append({"event_id": "e9"})  # people_total not labelled in every frame: no bounds
        events.append({"event_id": "e9", "people_unique": "1"})
        self.assertEqual(unique_bounds(derived, events),
                         {"n": 4, "within": 2, "below": 1, "above": 1, "prediction_blank": 1, "prediction_invalid": 1,
                          "within_rate": 0.5})
        self.assertIsNone(unique_bounds([], [])["within_rate"])

    def test_derived_report(self):
        events = [{"event_id": "e1", "image_count": "2", "people_unique": "4", "people_max_frame": "3", "dogs": "1",
                   "bicycles": "0", "large_bags": "0", "large_bags_uncertain": "0"},
                  {"event_id": "e2", "image_count": "1", "people_unique": "1", "people_max_frame": "1", "dogs": "0",
                   "bicycles": "1"}]
        report = evaluate_derived_events(self.labels, self.images, events)
        f = report["fields"]
        self.assertEqual(set(f), set(MAX_OVER_FRAMES) | {"people_max_frame", "large_bags_with_uncertain"})
        self.assertEqual((f["people_max_frame"]["n"], f["people_max_frame"]["mae"]), (2, 0.5))
        self.assertEqual((f["dogs"]["n"], f["dogs"]["exact"], f["bicycles"]["n"]), (2, 1.0, 1))
        self.assertEqual(f["strollers"]["status"], "missing_prediction_column")
        self.assertEqual(report["groups"], {})
        bounds = report["derivation"]["people_unique_bounds"]
        self.assertEqual((bounds["within"], bounds["above"]), (1, 1))
        text = format_report(report)
        self.assertIn("derived from image labels", text)
        self.assertIn("people_unique within", text)
        self.assertTrue(plain_json(report))
        json.dumps(report, allow_nan=False)


class BackpackEventTests(unittest.TestCase):
    def test_event_backpacks_are_maximum_and_incomplete_labels_abstain(self):
        images = [{"relative_path": "a.jpg", "event_id": "e1"},
                  {"relative_path": "b.jpg", "event_id": "e1"}]
        labels = [{"relative_path": "a.jpg", "backpacks": "2"},
                  {"relative_path": "b.jpg", "backpacks": "1"}]
        derived, _ = derive_event_labels(labels, images)
        self.assertEqual(derived, [{"event_id": "e1", "backpacks": 2}])
        labels[1]["backpacks"] = ""
        derived, _ = derive_event_labels(labels, images)
        self.assertNotIn("backpacks", derived[0])


class AdversarialCliTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.export = write(self.dir / "export.csv", V2_COLUMNS + ["event_id"],
                            [event_image("cam1/a.jpg", "cam1__1", people_total=2),
                             event_image("cam1/b.jpg", "cam1__1", people_total=0)])
        self.labels = write(self.dir / "labels.csv", LABEL_COLUMNS,
                            [label_row("CAM1/A.JPG", people_total=3), label_row("b.jpg", people_total=0)])
        self.events = write(self.dir / "events.csv", ["event_id", "image_count", "people_unique", "people_max_frame"],
                            [{"event_id": "cam1__1", "image_count": "2", "people_unique": "2", "people_max_frame": "2"}])

    def tearDown(self):
        self._tmp.cleanup()

    def test_events_with_image_labels_derives_event_labels(self):
        out_json = self.dir / "out.json"
        code, out, err = cli(["--labels", self.labels, "--export", self.export, "--events", self.events,
                              "--json", str(out_json)])
        self.assertEqual((code, err), (0, ""))
        self.assertIn("derived from image labels", out)
        report = json.loads(out_json.read_text(encoding="utf-8"))["events"]
        self.assertEqual((report["source"], report["derivation"]["events_derived"]), ("image_labels", 1))
        self.assertEqual(report["derivation"]["people_unique_bounds"]["below"], 1)
        self.assertEqual(report["fields"]["people_max_frame"]["bias"], -1.0)

    def test_non_utf8_and_ragged_files(self):
        bad = self.dir / "cp1255.csv"
        bad.write_bytes("relative_path,people_total\n".encode() + "\u05ea.jpg,1\n".encode("cp1255"))
        code, _, err = cli(["--labels", str(bad), "--export", self.export])
        self.assertEqual(code, 2)
        self.assertIn("UTF-8", err)
        self.assertIn("cp1255.csv", err)
        ragged = self.dir / "ragged.csv"
        ragged.write_text("relative_path,people_total\ncam1/a.jpg,2,extra,cells\nb.jpg\n\n,,\n", encoding="utf-8")
        code, out, _ = cli(["--labels", str(ragged), "--export", self.export])
        self.assertEqual(code, 0)
        self.assertIn("matched 2", out)
        self.assertIn("missing key 1", out)

    def test_repeated_label_column(self):
        dup = self.dir / "dup.csv"
        dup.write_text("relative_path,people_total,people_total\ncam1/a.jpg,5,2\n", encoding="utf-8")
        code, _, err = cli(["--labels", str(dup), "--export", self.export])
        self.assertEqual(code, 2)
        self.assertIn("people_total", err)

    def test_output_survives_narrow_console_encoding(self):
        raw = io.BytesIO()
        stream = io.TextIOWrapper(raw, encoding="cp1252", errors="strict")
        target = self.dir / "\u05ea\u05d1\u05e0\u05d9\u05ea.csv"
        with redirect_stdout(stream):
            template_code = main(["--template", self.export, "--out", str(target)])
            score_code = main(["--labels", self.labels, "--export", self.export, "--prefix", "\u05ea_"])
        stream.flush()
        self.assertEqual((template_code, score_code), (0, 0))
        self.assertTrue(target.exists())
        self.assertIn(b"\\u05ea", raw.getvalue())

    def test_force_requires_template(self):
        with self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
            main(["--labels", self.labels, "--export", self.export, "--force"])


if __name__ == "__main__":
    unittest.main()
