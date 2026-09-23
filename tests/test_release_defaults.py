"""Release-level checks for the conservative CLI and opt-in cached exports."""
import contextlib
import io
import json
from unittest.mock import patch

try:
    from . import test_pipeline as helpers
except ImportError:
    import test_pipeline as helpers


class ReleaseDefaultsTests(helpers.PipelineCase):
    def args(self, *extra):
        return helpers.entry.arguments([
            "--input", str(self.images), "--output", str(self.output),
            "--models", str(self.models), "--cache", str(self.cache),
            "--no-contact-sheet", *extra])

    def test_default_cli_exports_one_csv_and_native_expert_evidence(self):
        self.trail()
        with patch("trailcam.__main__.start_age_model", wraps=helpers.entry.start_age_model) as age:
            export = self.run_export()
        self.assertEqual(len(list(export.folder.glob("*.csv"))), 1)
        self.assertIsNone(export.summary["events_csv"])
        self.assertIsNone(export.summary["camera_days_csv"])
        self.assertEqual(export.summary["camera_calibrations"], {})
        self.assertFalse(export.summary["configuration"]["empty_frame_gate"])
        self.assertIsNone(age.call_args.args[0].age_model)
        self.vision_constructor.assert_called_once_with(self.models, "auto", 8, "standard", False)
        for row in export.rows:
            self.assertEqual((row["event_id"], row["large_bags"], row["large_bags_uncertain"]), ("", "", ""))
            self.assertEqual(row["age_unknown"], row["people_total"])
            self.assertEqual((row["adults"], row["children"]), ("0", "0"))
            self.assertEqual(row["people_near"], row["people_total"])
            self.assertEqual(row["direction_from_motion"], "0")
            self.assertEqual(row["large_bags_status"], "disabled")
            for person in json.loads(row["persons_json"]):
                self.assertIn("scores", person["attributes"])
                self.assertIn("pose", person)
                self.assertEqual(person["age"], "unknown")

    def test_experimental_options_reuse_raw_inference_and_can_be_disabled_again(self):
        self.trail()
        initial = self.run_export()
        calls = len(self.vision.calls)
        enabled = self.run_export("--events", "--geometry-age", "--large-bags", "--near-fraction", ".07")
        self.assertEqual(enabled.summary["cached_images"], len(helpers.TRAIL))
        self.assertEqual(len(self.vision.calls), calls)
        self.assertEqual(len(list(enabled.folder.glob("*.csv"))), 3)
        self.assertEqual(len(enabled.events), 3)
        self.assertEqual(enabled.row(helpers.GROUP)["children"], "1")
        self.assertEqual(enabled.row(helpers.GROUP)["adults"], "0")
        self.assertEqual(enabled.row(helpers.GROUP)["large_bags"], "0")
        again = self.run_export()
        self.assertEqual(len(self.vision.calls), calls)
        self.assertNotEqual(initial.folder, again.folder)
        self.assertEqual(len(list(again.folder.glob("*.csv"))), 1)
        self.assertEqual([helpers.without(row) for row in initial.rows],
                         [helpers.without(row) for row in again.rows])

    def test_experimental_settings_reject_non_booleans_before_inference(self):
        for key in ("events", "geometry_age", "large_bags"):
            with self.subTest(key=key):
                (self.root / "settings.json").write_text(json.dumps({key: "false"}), encoding="utf-8")
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    self.args()
        self.vision_constructor.assert_not_called()

    def test_contact_sheet_default_and_off_switch(self):
        defaults = helpers.entry.arguments([])
        self.assertEqual((defaults.contact_sheet, defaults.contact_sheet_size), (True, 20))
        self.assertFalse(self.args("--no-contact-sheet").contact_sheet)
