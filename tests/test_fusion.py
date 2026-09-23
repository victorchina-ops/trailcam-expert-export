"""Synthetic checks for per-image fusion (no models, no photos).

Covers counting-threshold filtering, cross-expert object verification
(corroboration, single-expert thresholds, vetoes, corroborate-only
MegaDetector), vehicle label merging, pose re-matching and ``fuse`` on a
synthetic v2 inference result (people acceptance, object counts, large bags,
facing counts and the review flag).
"""
from __future__ import annotations

import copy
import json
import unittest

from trailcam import fusion, roster
from trailcam.fusion import (COUNT_FIELDS, COUNTED, DIRECTIONS, OBJECT_POLICY, ORIENTATIONS, POSE_REMATCH,
                             _merge_vehicle_labels, fuse, rematch_pose, thresholded, verified_objects)

WIDTH, HEIGHT = 1000, 800


def det(label, confidence, box, index=None, **extra):
    d = {"label": label, "confidence": confidence, "xyxy": list(box), "bbox_xyxy": list(box)}
    if index is not None:
        d["detection_index"] = index
    d.update(extra)
    return d


def standing_keypoints(box, confidence=0.9):
    """17 COCO keypoints of an upright, fully visible person filling ``box``."""
    x1, y1, x2, y2 = box
    w, h, cx = x2 - x1, y2 - y1, (x1 + x2) / 2
    layout = [(0, .08), (-.05, .06), (.05, .06), (-.1, .07), (.1, .07), (-.2, .2), (.2, .2),
              (-.25, .35), (.25, .35), (-.25, .5), (.25, .5), (-.15, .5), (.15, .5),
              (-.15, .75), (.15, .75), (-.15, .97), (.15, .97)]
    return [[cx + dx * w, y1 + dy * h, confidence] for dx, dy in layout]


def evidence(orientation="unknown", facing="unclear"):
    return {"orientation": orientation, "facing_direction": facing}


def pose_det(confidence, box, orientation="unknown", facing="unclear", keypoints=True):
    return det("person", confidence, box, keypoints=standing_keypoints(box) if keypoints else None,
               orientation_evidence=evidence(orientation, facing))


def attributes(orientation="unknown", backpack=None):
    return {"status": "ok", "native_orientation_label": orientation, "backpack_presence": backpack,
            "native_age_label": "unknown"}


def make_result(yoloe=(), yolo26n=(), megadetector=(), pose=(), strip=None, width=WIDTH, height=HEIGHT):
    """A v2 inference result as ``VisionEngine.analyze`` + attributes would cache it."""
    experts = {}
    for name, dets in (("yolo26n", yolo26n), ("megadetector", megadetector), ("yoloe", yoloe), ("pose", pose)):
        dets = [dict(d) for d in dets]
        for index, d in enumerate(dets):
            d["detection_index"] = index
        experts[name] = {"detections": dets, "counts": {}, "device": "cpu"}
    strip = strip or {"top": 0, "bottom": 0}
    persons = []
    for c in roster.build_candidates(experts, strip, height):
        p = {"person_id": c["candidate_id"], **c, "keypoints": None, "pose_confidence": None,
             "orientation_evidence": None, "mask_polygon_xy": None, "appearance": None,
             "attributes": attributes()}
        if "pose" in c["members"]:
            source = experts["pose"]["detections"][c["members"]["pose"]["detection_index"]]
            p.update(keypoints=source.get("keypoints"), pose_confidence=source["confidence"],
                     orientation_evidence=source.get("orientation_evidence"))
        persons.append(p)
    return {"status": "ok", "width": width, "height": height, "data_strip": strip,
            "metadata": {"capture_time": None, "camera_make": None, "camera_model": None},
            "experts": experts, "persons": persons}


def person_at(result, x1):
    return next(p for p in result["persons"] if p["xyxy"][0] == x1)


def objects_of(**lists):
    return {name: list(dets) for name, dets in lists.items()}


class CountVoteTests(unittest.TestCase):
    def test_statuses(self):
        vote = fusion.count_vote({"yolo26n": 2, "yoloe": 2, "megadetector": 3})
        self.assertEqual((vote["value"], vote["status"], vote["support"], vote["model_count"]), (2, "majority", 2, 3))
        self.assertEqual((vote["spread"], vote["mean"]), (1, round(7 / 3, 6)))
        vote = fusion.count_vote({"yolo26n": 1, "yoloe": 2})
        self.assertEqual((vote["value"], vote["status"], vote["support"]), (None, "no_majority", 0))
        vote = fusion.count_vote({"yolo26n": None, "yoloe": 0})
        self.assertEqual((vote["value"], vote["status"], vote["mean"], vote["models"]),
                         (None, "single_expert", 0, ["yoloe"]))
        vote = fusion.count_vote({"yolo26n": None, "yoloe": None})
        self.assertEqual((vote["value"], vote["status"], vote["mean"], vote["spread"]),
                         (None, "no_eligible_output", None, None))

    def test_only_non_negative_ints_are_eligible(self):
        vote = fusion.count_vote({"a": True, "b": -1, "c": 1.0, "d": 1, "e": 1})
        self.assertEqual((vote["value"], vote["model_count"], vote["models"]), (1, 2, ["d", "e"]))


class OrientationFusionTests(unittest.TestCase):
    def test_table(self):
        cases = {("side", "side"): ("side", "attribute_and_pose"),
                 ("front", "back"): ("unknown", "conflict_abstention"),
                 ("front", "unknown"): ("front", "attribute_only"),
                 ("unknown", "back"): ("back", "pose_only"),
                 ("unknown", "unknown"): ("unknown", "no_usable_evidence"),
                 ("sideways", None): ("unknown", "no_usable_evidence")}
        for (attribute, pose), expected in cases.items():
            self.assertEqual(fusion.orientation_fusion(attribute, pose), expected, (attribute, pose))


class ThresholdedTests(unittest.TestCase):
    def test_counting_threshold_is_inclusive(self):
        t = OBJECT_POLICY["threshold"]
        result = make_result(yoloe=[det("dog", t, [0, 0, 10, 10]), det("dog", t - 1e-4, [50, 0, 60, 10])],
                             megadetector=[det("animal", .9, [0, 0, 10, 10]), det("animal", .1, [0, 0, 10, 10])])
        kept = thresholded(result)
        self.assertEqual(set(kept), {"yoloe", "yolo26n", "megadetector", "pose"})
        self.assertEqual([d["confidence"] for d in kept["yoloe"]], [t])
        self.assertEqual([d["confidence"] for d in kept["megadetector"]], [.9])

    def test_data_strip_detections_removed(self):
        result = make_result(yoloe=[det("bicycle", .9, [0, 770, 50, 800]), det("bicycle", .9, [0, 700, 50, 800])],
                             strip={"top": 0, "bottom": 40})
        kept = thresholded(result)["yoloe"]
        self.assertEqual([d["xyxy"] for d in kept], [[0, 700, 50, 800]])  # 40 % in the strip stays

    def test_semantic_vehicle_dedup_only_for_yoloe_and_yolo26n(self):
        box = [100, 100, 300, 250]
        pair = [det("car", .9, box), det("truck", .6, box)]
        result = make_result(yoloe=pair, yolo26n=pair, megadetector=pair)
        kept = thresholded(result)
        self.assertEqual([d["label"] for d in kept["yoloe"]], ["car"])
        self.assertEqual([d["label"] for d in kept["yolo26n"]], ["car"])
        self.assertEqual([d["label"] for d in kept["megadetector"]], ["car", "truck"])

    def test_vehicle_dedup_runs_inside_each_vehicle_group_only(self):
        # Trail utility vehicles retain their own label group; other competing
        # motor-vehicle labels are deduplicated together.
        # ATV/motorcycle and UTV/truck readings are settled by _merge_vehicle_labels.
        self.assertEqual(fusion.VEHICLE_GROUPS, ({"all-terrain vehicle", "utility terrain vehicle", "golf cart"},
                                                 {"car", "truck", "bus", "tractor", "motorcycle"}))
        box = [100, 100, 300, 250]
        cases = [(["truck", "utility terrain vehicle"], ["truck", "utility terrain vehicle"]),
                 (["bus", "golf cart", "car"], ["bus", "golf cart"]),
                 (["all-terrain vehicle", "golf cart", "tractor", "motorcycle"], ["all-terrain vehicle", "tractor"]),
                 (["car", "bicycle", "dog"], ["car", "bicycle", "dog"])]
        for labels, expected in cases:
            with self.subTest(labels=labels):
                detections = [det(label, .9 - .1 * n, box) for n, label in enumerate(labels)]
                kept = thresholded(make_result(yoloe=detections))["yoloe"]
                self.assertEqual([d["label"] for d in kept], expected)
                self.assertEqual([d["detection_index"] for d in kept],
                                 [labels.index(label) for label in expected])  # input order kept

    def test_input_is_not_mutated(self):
        result = make_result(yoloe=[det("car", .9, [0, 0, 10, 10]), det("truck", .6, [0, 0, 10, 10]),
                                    det("dog", .1, [0, 0, 5, 5])])
        before = copy.deepcopy(result)
        thresholded(result)
        self.assertEqual(result, before)

    def test_missing_parts(self):
        self.assertEqual(thresholded({}), {})
        self.assertEqual(thresholded({"experts": {"yoloe": {}}}), {"yoloe": []})


class VerifiedObjectsTests(unittest.TestCase):
    def test_two_experts_corroborate_one_object(self):
        found = verified_objects(objects_of(yoloe=[det("bicycle", .3, [0, 0, 100, 60])],
                                            yolo26n=[det("bicycle", .26, [2, 1, 101, 62])]), "bicycles")
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0], {"field": "bicycles", "status": "corroborated", "confidence": .3,
                                    "xyxy": [0, 0, 100, 60], "experts": ["yolo26n", "yoloe"], "label": "bicycle"})

    def test_below_match_iou_stays_separate(self):
        self.assertEqual(OBJECT_POLICY["match_iou"], .3)
        found = verified_objects(objects_of(yoloe=[det("bicycle", .9, [0, 0, 100, 100])],
                                            yolo26n=[det("bicycle", .9, [0, 0, 100, 29])]), "bicycles")
        self.assertEqual([o["status"] for o in found], ["single_expert_confident"] * 2)
        found = verified_objects(objects_of(yoloe=[det("bicycle", .9, [0, 0, 100, 100])],
                                            yolo26n=[det("bicycle", .9, [0, 0, 100, 30])]), "bicycles")
        self.assertEqual([o["status"] for o in found], ["corroborated"])

    def test_single_expert_thresholds_per_field(self):
        for field, spec in OBJECT_POLICY["fields"].items():
            label = spec["experts"]["yoloe"][0]
            single = spec["single"]
            with self.subTest(field=field):
                at = verified_objects(objects_of(yoloe=[det(label, single, [0, 0, 50, 50])]), field)
                self.assertEqual([o["status"] for o in at], ["single_expert_confident"])
                below = verified_objects(objects_of(yoloe=[det(label, single - 1e-3, [0, 0, 50, 50])]), field)
                self.assertEqual([o["status"] for o in below], ["unverified"])

    def test_policy_shape(self):
        fields = OBJECT_POLICY["fields"]
        self.assertEqual(set(fields), set(COUNT_FIELDS[1:]))
        # A lone dog must be very confident (crouching people); lone motorcycles more than bicycles.
        self.assertEqual(fields["dogs"]["single"], max(spec["single"] for spec in fields.values()))
        self.assertGreater(fields["motorcycles"]["single"], fields["bicycles"]["single"])
        for spec in fields.values():
            self.assertGreaterEqual(spec["single"], OBJECT_POLICY["threshold"])
        self.assertEqual(fields["dogs"]["veto"]["labels"], ("person",))
        self.assertEqual(fields["motorcycles"]["veto"]["labels"], ("bicycle",))

    def test_one_member_per_expert(self):
        found = verified_objects(objects_of(yoloe=[det("dog", .95, [0, 0, 50, 50]), det("dog", .94, [1, 0, 51, 50])]),
                                 "dogs")
        self.assertEqual(len(found), 2)
        self.assertTrue(all(o["experts"] == ["yoloe"] for o in found))

    def test_clusters_are_reported_by_descending_confidence(self):
        found = verified_objects(objects_of(yoloe=[det("backpack", .4, [0, 0, 10, 10]),
                                                   det("backpack", .8, [100, 0, 110, 10]),
                                                   det("backpack", .6, [200, 0, 210, 10])]), "backpacks")
        self.assertEqual([o["confidence"] for o in found], [.8, .6, .4])

    def test_labels_are_expert_specific(self):
        self.assertEqual(verified_objects(objects_of(yolo26n=[det("baby stroller", .9, [0, 0, 5, 5])]), "strollers"), [])
        tractor = [det("tractor", .9, [0, 0, 50, 50])]
        self.assertEqual(len(verified_objects(objects_of(yoloe=tractor), "other_vehicles")), 1)
        self.assertEqual(verified_objects(objects_of(yolo26n=tractor), "other_vehicles"), [])
        self.assertEqual(verified_objects(objects_of(yoloe=[det("golf cart", .5, [0, 0, 50, 50])]),
                                          "atv_utv")[0]["status"], "single_expert_confident")
        self.assertEqual(verified_objects(objects_of(yoloe=[det("dog", .99, [0, 0, 5, 5])]), "bicycles"), [])

    def test_generic_animal_never_corroborates_dog_species(self):
        animal = det("animal", .99, [0, 0, 50, 50])
        self.assertEqual(verified_objects(objects_of(megadetector=[animal]), "dogs"), [])
        found = verified_objects(objects_of(megadetector=[animal], yoloe=[det("dog", .3, [1, 1, 50, 51])]), "dogs")
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["status"], "unverified")
        self.assertEqual(found[0]["experts"], ["yoloe"])
        self.assertEqual(found[0]["confidence"], .3)             # creators only
        self.assertEqual(found[0]["xyxy"], [1, 1, 50, 51])       # anchor is the creator box
        self.assertEqual(found[0]["label"], "dog")

    def test_dog_on_person_is_vetoed(self):
        veto = OBJECT_POLICY["fields"]["dogs"]["veto"]
        box = [300, 600, 380, 700]  # height 100: a person box [300, 600, 380, 600 + 100 r] has IoU r
        dog = det("dog", .95, box)
        for expert in ("yolo26n", "megadetector"):
            with self.subTest(expert=expert):
                self.assertIn(expert, veto["experts"])
                found = verified_objects(objects_of(yoloe=[dog], **{expert: [det("person", veto["conf"], box)]}),
                                         "dogs")
                self.assertEqual([o["status"] for o in found], ["vetoed"])
        # YOLOE's own person box is not a veto source; low-confidence and low-overlap people do not veto.
        self.assertNotIn("yoloe", veto["experts"])
        self.assertEqual(verified_objects(objects_of(yoloe=[dog, det("person", .9, box)]), "dogs")[0]["status"],
                         "single_expert_confident")
        weak = det("person", veto["conf"] - .01, box)
        self.assertEqual(verified_objects(objects_of(yoloe=[dog], yolo26n=[weak]), "dogs")[0]["status"],
                         "single_expert_confident")
        low = det("person", .9, [300, 600, 380, 600 + 100 * (veto["iou"] - .02)])
        self.assertEqual(verified_objects(objects_of(yoloe=[dog], yolo26n=[low]), "dogs")[0]["status"],
                         "single_expert_confident")
        high = det("person", .9, [300, 600, 380, 600 + 100 * (veto["iou"] + .02)])
        self.assertEqual(verified_objects(objects_of(yoloe=[dog], yolo26n=[high]), "dogs")[0]["status"], "vetoed")

    def test_veto_beats_corroboration(self):
        box = [300, 600, 380, 700]
        found = verified_objects(objects_of(yoloe=[det("dog", .95, box)],
                                            yolo26n=[det("dog", .9, box), det("person", .8, box)]), "dogs")
        self.assertEqual([o["status"] for o in found], ["vetoed"])
        self.assertEqual(found[0]["experts"], ["yolo26n", "yoloe"])

    def test_motorcycle_on_bicycle_is_vetoed(self):
        box = [600, 550, 700, 650]
        detections = objects_of(yoloe=[det("motorcycle", .8, box), det("bicycle", .3, box)])
        self.assertEqual([o["status"] for o in verified_objects(detections, "motorcycles")], ["vetoed"])
        self.assertEqual([o["status"] for o in verified_objects(detections, "bicycles")],
                         ["single_expert_confident"])
        by_yolo26n = objects_of(yoloe=[det("motorcycle", .8, box)], yolo26n=[det("bicycle", .5, box)])
        self.assertEqual(verified_objects(by_yolo26n, "motorcycles")[0]["status"], "vetoed")
        veto = OBJECT_POLICY["fields"]["motorcycles"]["veto"]
        weak = objects_of(yoloe=[det("motorcycle", .8, box), det("bicycle", veto["conf"] - .01, box)])
        self.assertEqual(verified_objects(weak, "motorcycles")[0]["status"], "single_expert_confident")
        beside = objects_of(yoloe=[det("motorcycle", .8, box),
                                   det("bicycle", .9, [600, 550, 700, 550 + 100 * (veto["iou"] - .02)])])
        self.assertEqual(verified_objects(beside, "motorcycles")[0]["status"], "single_expert_confident")

    def test_veto_boxes_come_from_all_detections(self):
        box = [0, 0, 100, 100]
        counted = objects_of(yoloe=[det("dog", .95, box)])
        raw = objects_of(yoloe=[det("dog", .95, box)], yolo26n=[det("person", .45, box)])
        self.assertEqual(verified_objects(counted, "dogs")[0]["status"], "single_expert_confident")
        self.assertEqual(verified_objects(counted, "dogs", all_detections=raw)[0]["status"], "vetoed")

    def test_only_counted_statuses_count(self):
        self.assertEqual(COUNTED, ("corroborated", "single_expert_confident"))

    def test_output_is_json_serialisable_and_input_unchanged(self):
        detections = objects_of(yoloe=[det("dog", .95, [0, 0, 50, 50])], megadetector=[det("animal", .6, [0, 0, 50, 50])])
        before = copy.deepcopy(detections)
        found = verified_objects(detections, "dogs")
        self.assertEqual(detections, before)
        self.assertEqual(json.loads(json.dumps(found)), found)


class MergeVehicleLabelTests(unittest.TestCase):
    @staticmethod
    def obj(field, status, box):
        return {"field": field, "status": status, "confidence": .6, "xyxy": list(box), "experts": ["yoloe"],
                "label": {"atv_utv": "utility terrain vehicle", "motorcycles": "motorcycle"}.get(field, "truck")}

    def test_overlapping_other_vehicle_merges_into_atv(self):
        utv, truck = [100, 100, 300, 250], [100, 100, 300, 300]  # IoU .75
        objects = {"atv_utv": [self.obj("atv_utv", "single_expert_confident", utv)],
                   "other_vehicles": [self.obj("other_vehicles", "single_expert_confident", truck)],
                   "cars_trucks_buses": [self.obj("cars_trucks_buses", "corroborated", truck)]}
        counts = {"atv_utv": 1, "other_vehicles": 1, "cars_trucks_buses": 1}
        _merge_vehicle_labels(objects, counts, OBJECT_POLICY)
        self.assertEqual(objects["other_vehicles"][0]["status"], "merged_into_atv_utv")
        self.assertEqual(objects["cars_trucks_buses"][0]["status"], "merged_into_atv_utv")
        # The merge now also settles motorcycle-vs-ATV readings and recounts both fields.
        self.assertEqual(counts, {"atv_utv": 1, "motorcycles": 0, "other_vehicles": 0, "cars_trucks_buses": 0})

    def test_separate_or_unverified_vehicles_are_kept(self):
        objects = {"atv_utv": [self.obj("atv_utv", "unverified", [0, 0, 100, 100]),
                               self.obj("atv_utv", "single_expert_confident", [500, 0, 600, 100])],
                   "other_vehicles": [self.obj("other_vehicles", "single_expert_confident", [0, 0, 100, 100]),
                                      self.obj("other_vehicles", "unverified", [500, 0, 600, 100])],
                   "cars_trucks_buses": [self.obj("cars_trucks_buses", "corroborated", [800, 0, 900, 100])]}
        counts = {}
        _merge_vehicle_labels(objects, counts, OBJECT_POLICY)
        self.assertEqual([o["status"] for o in objects["other_vehicles"]], ["single_expert_confident", "unverified"])
        self.assertEqual(counts, {"atv_utv": 1, "motorcycles": 0, "other_vehicles": 1, "cars_trucks_buses": 1})

    def test_merge_iou_threshold_and_missing_fields(self):
        objects = {"atv_utv": [self.obj("atv_utv", "corroborated", [0, 0, 100, 100])],
                   "other_vehicles": [dict(self.obj("other_vehicles", "corroborated", [0, 0, 100, 39]), label="tractor")]}  # IoU .39
        counts = {}
        _merge_vehicle_labels(objects, counts, OBJECT_POLICY)
        self.assertEqual(counts, {"atv_utv": 1, "motorcycles": 0, "other_vehicles": 1, "cars_trucks_buses": 0})
        counts = {}
        _merge_vehicle_labels({}, counts, OBJECT_POLICY)
        self.assertEqual(counts, {"atv_utv": 0, "motorcycles": 0, "other_vehicles": 0, "cars_trucks_buses": 0})

    def test_quad_read_as_atv_and_motorcycle_counts_once(self):
        # Regression: a quad bike labelled both ATV and motorcycle was counted twice.
        quad = [100, 100, 300, 250]
        objects = {"atv_utv": [dict(self.obj("atv_utv", "single_expert_confident", quad), confidence=.6)],
                   "motorcycles": [dict(self.obj("motorcycles", "corroborated", quad), confidence=.55)]}
        counts = {}
        _merge_vehicle_labels(objects, counts, OBJECT_POLICY)
        self.assertEqual((counts["atv_utv"], counts["motorcycles"]), (1, 0))
        objects["motorcycles"][0].update(status="corroborated", confidence=.7)
        objects["atv_utv"][0]["status"] = "single_expert_confident"
        counts = {}
        _merge_vehicle_labels(objects, counts, OBJECT_POLICY)
        self.assertEqual((counts["atv_utv"], counts["motorcycles"]), (0, 1))


class RematchPoseTests(unittest.TestCase):
    @staticmethod
    def scene(poses, people):
        return {"experts": {"pose": {"detections": poses}}}, people

    @staticmethod
    def accepted(person_id, box, keypoints=None, members=None):
        return {"person_id": person_id, "xyxy": list(box), "keypoints": keypoints, "members": members or {}}

    def test_person_without_keypoints_gets_overlapping_pose(self):
        box, loose = [550, 250, 640, 520], [520, 200, 680, 560]
        pose = pose_det(.13, loose, "back", "away")
        result, people = self.scene([pose], [self.accepted("cand_0001", box)])
        rematch_pose(result, people)
        person = people[0]
        self.assertEqual(person["keypoints"], pose["keypoints"])
        self.assertEqual(person["pose_confidence"], .13)
        self.assertEqual(person["orientation_evidence"], evidence("back", "away"))
        self.assertEqual(person["pose_rematch"]["detection_index"], 0)
        self.assertAlmostEqual(person["pose_rematch"]["iou"], round(90 * 270 / (160 * 360), 4))

    def test_people_with_keypoints_or_used_poses_are_skipped(self):
        box = [0, 0, 100, 300]
        own = [[1, 1, 1]] * 17
        result, people = self.scene([pose_det(.9, box)], [self.accepted("cand_0001", box, keypoints=own)])
        rematch_pose(result, people)
        self.assertIs(people[0]["keypoints"], own)
        self.assertNotIn("pose_rematch", people[0])
        # A pose already linked to an accepted person is not handed out again.
        holder = self.accepted("cand_0001", box, keypoints=own, members={"pose": {"detection_index": 0}})
        result, people = self.scene([pose_det(.9, box)], [holder, self.accepted("cand_0002", [5, 0, 105, 300])])
        rematch_pose(result, people)
        self.assertIsNone(people[1]["keypoints"])

    def test_gates(self):
        box = [0, 0, 100, 300]  # a pose box [0, 0, 100, 300 r] has IoU r
        cases = {"low_confidence": pose_det(POSE_REMATCH["min_confidence"] - 1e-3, box),
                 "no_keypoints": pose_det(.9, box, keypoints=False),
                 "low_iou": pose_det(.9, [0, 0, 100, 300 * (POSE_REMATCH["min_iou"] - .01)])}
        for name, pose in cases.items():
            with self.subTest(case=name):
                result, people = self.scene([pose], [self.accepted("cand_0001", box)])
                rematch_pose(result, people)
                self.assertIsNone(people[0]["keypoints"])
        just_above = [0, 0, 100, 300 * (POSE_REMATCH["min_iou"] + .005)]
        result, people = self.scene([pose_det(POSE_REMATCH["min_confidence"], just_above)],
                                    [self.accepted("cand_0001", box)])
        rematch_pose(result, people)  # confidence at the floor is inclusive
        self.assertIsNotNone(people[0]["keypoints"])

    def test_best_overlap_wins_and_each_pose_is_used_once(self):
        pose_box = [0, 0, 100, 300]
        a = self.accepted("cand_0001", [0, 0, 100, 250])    # IoU .83
        b = self.accepted("cand_0002", [10, 0, 110, 300])   # IoU .82
        result, people = self.scene([pose_det(.8, pose_box)], [b, a])
        rematch_pose(result, people)
        self.assertIsNotNone(a["keypoints"])
        self.assertIsNone(b["keypoints"])
        # With two poses each person receives its own best one.
        poses = [pose_det(.8, [200, 0, 300, 300]), pose_det(.8, pose_box)]
        a, b = self.accepted("cand_0001", [0, 0, 100, 300]), self.accepted("cand_0002", [205, 0, 305, 300])
        result, people = self.scene(poses, [a, b])
        rematch_pose(result, people)
        self.assertEqual(a["pose_rematch"]["detection_index"], 1)
        self.assertEqual(b["pose_rematch"]["detection_index"], 0)

    def test_sparse_raw_pose_ids_are_not_reused_or_replaced_by_list_positions(self):
        box = [0, 0, 100, 300]
        pose = dict(pose_det(.9, box), detection_index=4)
        holder = self.accepted("cand_0001", box, keypoints=pose["keypoints"],
                               members={"pose": {"detection_index": 4}})
        waiting = self.accepted("cand_0002", [2, 0, 102, 300])
        result, people = self.scene([pose], [holder, waiting])
        rematch_pose(result, people)
        self.assertIsNone(waiting["keypoints"])
        # Without a holder the recorded assignment keeps the raw ID, and that
        # same ID prevents a second call from lending the pose out again.
        rematch_pose(result, [waiting])
        self.assertEqual(waiting["pose_rematch"]["detection_index"], 4)
        next_person = self.accepted("cand_0003", box)
        rematch_pose(result, [waiting, next_person])
        self.assertIsNone(next_person["keypoints"])

    def test_no_pose_expert(self):
        people = [self.accepted("cand_0001", [0, 0, 10, 10])]
        rematch_pose({"experts": {}}, people)
        self.assertIsNone(people[0]["keypoints"])

    def test_rematch_is_idempotent(self):
        # v2: a pose handed out by an earlier rematch counts as used, so running
        # rematch_pose again never gives the same pose to a second person.
        pose_box = [0, 0, 100, 300]
        first = self.accepted("cand_0001", [0, 0, 100, 290])
        result, people = self.scene([pose_det(.8, pose_box)], [first])
        rematch_pose(result, people)
        self.assertEqual(first["pose_rematch"]["detection_index"], 0)
        snapshot = copy.deepcopy(first)
        second = self.accepted("cand_0002", [2, 0, 102, 300])  # would otherwise take pose 0
        people.append(second)
        rematch_pose(result, people)
        rematch_pose(result, people)
        self.assertEqual(first, snapshot)
        self.assertIsNone(second["keypoints"])
        self.assertNotIn("pose_rematch", second)


# The shared synthetic scene (1000 x 800, data strip 40 px at the bottom).
P1 = [100, 200, 200, 500]   # strong YOLOE + pose, side-left, large pack
P2 = [400, 200, 500, 500]   # weak YOLOE + weak pose (pair rule), front, uncertain pack
P3 = [700, 50, 760, 180]    # weak lone yolo26n -> rejected
P4 = [0, 770, 60, 800]      # inside the data strip -> not a candidate
P5 = [550, 250, 640, 520]   # YOLOE only, back; its loose pose box forms a rejected candidate
P5_POSE = [520, 200, 680, 560]
P6 = [800, 150, 880, 420]   # attribute front vs pose back -> conflict
BIKE = [600, 550, 700, 650]
CROUCH = [300, 600, 380, 700]
MOTO_LONE = [20, 20, 80, 80]
SUITCASE = [900, 20, 980, 120]
ANIMAL = [850, 600, 950, 700]


def scene_result():
    result = make_result(
        yoloe=[det("person", .85, P1), det("person", .15, P2), det("person", .9, P4), det("person", .7, P5),
               det("person", .6, P6),
               det("backpack", .6, [120, 230, 180, 380]),   # tall, above the shoulders -> large
               det("backpack", .5, [420, 260, 480, 380]),   # mid-size -> uncertain
               det("bicycle", .6, BIKE), det("motorcycle", .7, BIKE), det("motorcycle", .4, MOTO_LONE),
               det("dog", .92, CROUCH), det("suitcase", .5, SUITCASE)],
        yolo26n=[det("person", .3, P3), det("bicycle", .5, [602, 552, 701, 652])],
        megadetector=[det("person", .6, CROUCH), det("animal", .99, ANIMAL)],
        pose=[pose_det(.8, P1, "side", "left"), pose_det(.14, [402, 198, 501, 502]),
              pose_det(.13, P5_POSE, "back", "away"), pose_det(.5, P6, "back", "away")],
        strip={"top": 0, "bottom": 40})
    person_at(result, P1[0])["attributes"] = attributes("side", True)
    person_at(result, P2[0])["attributes"] = attributes("front", None)
    person_at(result, P5[0])["attributes"] = attributes("back", False)
    person_at(result, P6[0])["attributes"] = attributes("front", None)
    return result


class FuseSceneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = fuse(scene_result())
        cls.combined = cls.result["combined"]

    def test_candidates_and_acceptance(self):
        persons = {p["xyxy"][0]: p for p in self.result["persons"]}
        self.assertNotIn(P4[0], persons)  # strip person never became a candidate
        self.assertEqual(persons[P1[0]]["acceptance"], "single_expert_confident:yoloe+pose")
        self.assertEqual(persons[P2[0]]["acceptance"], "multi_expert_agreement:yoloe+pose")
        self.assertTrue(persons[P5[0]]["accepted"])
        self.assertTrue(persons[P6[0]]["accepted"])
        self.assertEqual(persons[P3[0]]["accepted"], False)
        self.assertEqual(persons[P3[0]]["acceptance"], "insufficient_confidence_or_support")
        self.assertEqual(persons[P5_POSE[0]]["accepted"], False)  # the loose pose-only candidate
        self.assertEqual(self.combined["counts"]["people_total"], 4)
        self.assertEqual(self.combined["count_sources"]["people_total"], roster.POLICY["version"])
        self.assertEqual(self.combined["count_sources"]["bicycles"], OBJECT_POLICY["version"])

    def test_pose_rematched_to_accepted_person(self):
        p5 = person_at(self.result, P5[0])
        self.assertEqual(p5["pose_rematch"]["detection_index"], 2)
        self.assertEqual(p5["pose_confidence"], .13)
        self.assertIsNotNone(p5["keypoints"])
        self.assertEqual(p5["combined"]["orientation_source"], "attribute_and_pose")

    def test_object_counts(self):
        counts = self.combined["counts"]
        self.assertEqual(counts["bicycles"], 1)
        self.assertEqual(counts["motorcycles"], 0)
        self.assertEqual(counts["dogs"], 0)
        self.assertEqual(counts["backpacks"], 2)
        for field in ("strollers", "atv_utv", "other_vehicles", "cars_trucks_buses", "kick_scooters"):
            self.assertEqual(counts[field], 0, field)
        self.assertEqual(set(counts), set(COUNT_FIELDS))
        objects = self.result["objects"]
        self.assertEqual([o["status"] for o in objects["bicycles"]], ["corroborated"])
        self.assertEqual(sorted(o["status"] for o in objects["motorcycles"]), ["unverified", "vetoed"])
        self.assertEqual([o["status"] for o in objects["dogs"]], ["vetoed"])  # MegaDetector person under the "dog"
        self.assertEqual(self.combined["unverified_counts"]["motorcycles"], 1)
        self.assertEqual(self.combined["unverified_counts"]["dogs"], 0)

    def test_expert_counts_use_counting_threshold_and_strip(self):
        experts = self.result["experts"]
        self.assertEqual(experts["yoloe"]["counts"]["people_total"], 3)       # .15 below, strip excluded
        self.assertEqual(experts["yolo26n"]["counts"]["people_total"], 1)
        self.assertEqual(experts["megadetector"]["counts"]["people_total"], 1)
        self.assertIsNone(experts["yolo26n"]["counts"]["strollers"])           # unsupported, not zero
        self.assertEqual(experts["yoloe"]["counts"]["strollers"], 0)
        votes = self.result["votes"]
        self.assertEqual(set(votes), set(COUNT_FIELDS))
        self.assertEqual(votes["people_total"]["status"], "majority")   # yolo26n and MegaDetector agree
        self.assertEqual(votes["people_total"]["value"], 1)             # votes are diagnostics only
        self.assertEqual(votes["motorcycles"]["status"], "no_majority")
        self.assertEqual(votes["strollers"]["status"], "single_expert")

    def test_large_bags(self):
        p1, p2 = person_at(self.result, P1[0]), person_at(self.result, P2[0])
        self.assertIs(p1["combined"]["large_bag"], True)
        self.assertIsNone(p2["combined"]["large_bag"])
        self.assertTrue(p2["combined"]["bag_assessed"])
        self.assertEqual(self.combined["loose_luggage"], 1)
        self.assertEqual(self.combined["large_bags"], 2)            # carried pack + loose suitcase
        self.assertEqual(self.combined["large_bags_uncertain"], 1)
        self.assertIsNone(person_at(self.result, P6[0])["combined"]["large_bag"])  # no bag, not assessed
        self.assertFalse(person_at(self.result, P6[0])["combined"]["bag_assessed"])

    def test_backpack_carrying(self):
        self.assertEqual(person_at(self.result, P1[0])["combined"]["backpack_source"],
                         "detector_spatial_support_and_attribute")
        self.assertEqual(person_at(self.result, P2[0])["combined"]["backpack_source"], "spatial_support_only_unverified")
        self.assertTrue(person_at(self.result, P2[0])["combined"]["associated_backpack"])
        self.assertEqual(person_at(self.result, P5[0])["combined"]["backpack_source"], "attribute_only")
        self.assertEqual(self.combined["carrying_backpack_counts"], {"yes": 1, "no": 1, "unknown": 2})

    def test_facing_and_orientation(self):
        facing = {x: person_at(self.result, x)["combined"]["facing"] for x in (P1[0], P2[0], P5[0], P6[0])}
        self.assertEqual(facing, {P1[0]: "left", P2[0]: "toward", P5[0]: "away", P6[0]: "unclear"})
        self.assertEqual(self.combined["facing_counts"], {"left": 1, "right": 0, "toward": 1, "away": 1, "unclear": 1})
        self.assertEqual(list(self.combined["facing_counts"]), list(DIRECTIONS))
        self.assertEqual(self.combined["orientation_counts"], {"front": 1, "back": 1, "side": 1, "unknown": 1})
        self.assertEqual(list(self.combined["orientation_counts"]), list(ORIENTATIONS))
        self.assertEqual(self.combined["orientation_conflicts"], 1)
        self.assertEqual(person_at(self.result, P6[0])["combined"]["orientation_source"], "conflict_abstention")

    def test_age_and_direction_pending_postprocess(self):
        self.assertIsNone(self.combined["adults"])
        self.assertIsNone(self.combined["children"])
        self.assertEqual(self.combined["age_unknown"], 4)
        self.assertIsNone(self.combined["direction_counts"])
        self.assertEqual(self.combined["age_status"], "pending_postprocess")
        for x in (P1[0], P2[0], P5[0], P6[0]):
            c = person_at(self.result, x)["combined"]
            self.assertEqual((c["age"], c["age_method"], c["direction_source"]), ("unknown", "pending", "pending"))

    def test_needs_review(self):
        self.assertTrue(self.result["needs_review"])

    def test_json_serialisable(self):
        encoded = json.dumps(self.result, allow_nan=False)
        self.assertIsInstance(encoded, str)

    def test_fuse_is_deterministic(self):
        self.assertEqual(fuse(scene_result()), fuse(scene_result()))


class FuseTargetedTests(unittest.TestCase):
    def test_clean_image_does_not_need_review(self):
        box = [400, 200, 500, 500]
        result = fuse(make_result(yoloe=[det("person", .9, box)], yolo26n=[det("person", .9, box)],
                                  megadetector=[det("person", .9, box)]))
        self.assertEqual(result["combined"]["counts"]["people_total"], 1)
        self.assertEqual(result["votes"]["people_total"]["status"], "majority")
        self.assertEqual(result["votes"]["people_total"]["value"], 1)
        self.assertFalse(result["needs_review"])
        person = result["persons"][0]
        self.assertEqual(person["combined"]["facing"], "unclear")
        self.assertEqual(person["combined"]["orientation_source"], "no_usable_evidence")
        self.assertIsNone(person["combined"]["large_bag"])
        self.assertEqual(result["combined"]["large_bags"], 0)
        self.assertEqual(result["combined"]["large_bags_uncertain"], 0)

    def test_empty_image_counts_are_zero_not_blank(self):
        result = fuse(make_result())
        counts = result["combined"]["counts"]
        self.assertEqual(counts, dict.fromkeys(COUNT_FIELDS, 0))
        self.assertEqual(result["combined"]["age_unknown"], 0)
        self.assertEqual(result["combined"]["facing_counts"], dict.fromkeys(DIRECTIONS, 0))
        self.assertFalse(result["needs_review"])

    def test_each_review_trigger(self):
        box = [400, 200, 500, 500]

        def scene(yoloe=(), yolo26n=(), pose=()):
            people = [det("person", .9, box)]
            return make_result(yoloe=people + list(yoloe), yolo26n=people + list(yolo26n),
                               megadetector=people, pose=pose)

        corroborated = fuse(scene(yoloe=[det("motorcycle", .3, [0, 0, 50, 50])],
                                  yolo26n=[det("motorcycle", .3, [0, 0, 50, 50])]))
        self.assertEqual(corroborated["combined"]["counts"]["motorcycles"], 1)
        self.assertFalse(corroborated["needs_review"])
        below = fuse(scene(yoloe=[det("kick scooter", .2, [0, 0, 50, 50])]))
        self.assertEqual(below["combined"]["unverified_counts"]["kick_scooters"], 0)
        self.assertFalse(below["needs_review"])  # below the counting threshold: not even unverified
        # Unverified objects alone (the per-expert vote agrees: one each).
        unverified = fuse(scene(yoloe=[det("motorcycle", .4, [0, 0, 50, 50])],
                                yolo26n=[det("motorcycle", .4, [900, 0, 950, 50])]))
        self.assertEqual(unverified["votes"]["motorcycles"]["status"], "majority")
        self.assertEqual(unverified["combined"]["unverified_counts"]["motorcycles"], 2)
        self.assertTrue(unverified["needs_review"])
        # Expert count disagreement alone (the dog itself is confident).
        disagreement = fuse(scene(yoloe=[det("dog", .95, [0, 0, 50, 50])]))
        self.assertEqual(disagreement["votes"]["dogs"]["status"], "no_majority")
        self.assertEqual(disagreement["combined"]["unverified_counts"]["dogs"], 0)
        self.assertTrue(disagreement["needs_review"])
        # Uncertain large-bag reading alone.
        pack = [420, 260, 480, 380]
        uncertain = fuse(scene(yoloe=[det("backpack", .5, pack)], yolo26n=[det("backpack", .5, pack)],
                               pose=[pose_det(.9, box)]))
        self.assertEqual(uncertain["votes"]["backpacks"]["status"], "majority")
        self.assertEqual(uncertain["combined"]["large_bags_uncertain"], 1)
        self.assertTrue(uncertain["needs_review"])
        # Orientation conflict alone.
        conflict = scene(pose=[pose_det(.9, box, "back", "away")])
        conflict["persons"][0]["attributes"] = attributes("front")
        conflict = fuse(conflict)
        self.assertEqual(conflict["combined"]["orientation_conflicts"], 1)
        self.assertTrue(conflict["needs_review"])

    def test_vehicle_labels_merge_inside_fuse(self):
        utv, truck = [100, 100, 300, 250], [100, 100, 300, 300]  # IoU .75: both survive the .8 dedup
        result = fuse(make_result(yoloe=[det("utility terrain vehicle", .5, utv), det("truck", .6, truck)]))
        counts = result["combined"]["counts"]
        self.assertEqual((counts["atv_utv"], counts["other_vehicles"], counts["cars_trucks_buses"]), (1, 0, 0))
        self.assertEqual(result["objects"]["other_vehicles"][0]["status"], "merged_into_atv_utv")

    def test_utv_and_truck_on_the_same_box_count_as_one_atv_utv(self):
        # v2: label de-duplication runs inside vehicle groups only, so a UTV label
        # is no longer deleted in favour of an overlapping, more confident truck
        # (v1 counted this as other_vehicles=1, atv_utv=0). The truck object is
        # then merged into the ATV/UTV one: one physical vehicle, counted once.
        box = [100, 100, 300, 300]
        result = fuse(make_result(yoloe=[det("utility terrain vehicle", .5, box), det("truck", .6, box)]))
        counts = result["combined"]["counts"]
        self.assertEqual((counts["atv_utv"], counts["other_vehicles"], counts["cars_trucks_buses"]), (1, 0, 0))
        self.assertEqual([o["status"] for o in result["objects"]["atv_utv"]], ["single_expert_confident"])
        self.assertEqual([o["status"] for o in result["objects"]["other_vehicles"]], ["merged_into_atv_utv"])
        self.assertEqual([o["status"] for o in result["objects"]["cars_trucks_buses"]], ["merged_into_atv_utv"])
        self.assertEqual(sum(result["combined"]["counts"][f] for f in ("atv_utv", "other_vehicles")), 1)

    def test_competing_car_motorcycle_labels_count_once_within_and_across_experts(self):
        box = [100, 100, 300, 300]
        cases = [("yoloe", "yoloe"), ("yoloe", "yolo26n"), ("yolo26n", "yoloe")]
        for car_expert, motorcycle_expert in cases:
            with self.subTest(car=car_expert, motorcycle=motorcycle_expert):
                inputs = {"yoloe": [], "yolo26n": []}
                inputs[car_expert].append(det("car", .8, box))
                inputs[motorcycle_expert].append(det("motorcycle", .7, box))
                result = fuse(make_result(**inputs))
                counts = result["combined"]["counts"]
                self.assertEqual((counts["motorcycles"], counts["other_vehicles"], counts["cars_trucks_buses"]), (0, 1, 1))

    def test_vehicle_subset_does_not_survive_its_parent_merging_into_utv(self):
        result = fuse(make_result(
            yoloe=[det("utility terrain vehicle", .7, [0, 0, 100, 100]),
                   det("tractor", .9, [0, 0, 100, 100])],
            yolo26n=[det("car", .8, [0, 0, 35, 100])]))
        counts = result["combined"]["counts"]
        self.assertEqual((counts["atv_utv"], counts["other_vehicles"], counts["cars_trucks_buses"]), (1, 0, 0))
        # A real separate car is retained, including its subset membership.
        separate = fuse(make_result(yoloe=[det("tractor", .8, [0, 0, 100, 100]),
                                                  det("car", .9, [200, 0, 300, 100])]))
        counts = separate["combined"]["counts"]
        self.assertEqual((counts["other_vehicles"], counts["cars_trucks_buses"]), (2, 1))

    def test_backpack_spatial_association_alone_does_not_establish_carrying(self):
        box, bag = [100, 100, 200, 400], [120, 150, 180, 250]
        for attribute, expected, source in ((None, None, "spatial_support_only_unverified"),
                                             (False, None, "attribute_detector_conflict"),
                                             (True, True, "detector_spatial_support_and_attribute")):
            with self.subTest(attribute=attribute):
                raw = make_result(yoloe=[det("person", .9, box), det("backpack", .8, bag)])
                raw["persons"][0]["attributes"] = attributes(backpack=attribute)
                result = fuse(raw)
                person = result["persons"][0]["combined"]
                self.assertIs(person["carrying_backpack"], expected)
                self.assertTrue(person["associated_backpack"])
                self.assertEqual(person["backpack_source"], source)
                self.assertEqual(result["combined"]["counts"]["backpacks"], 1)

    def test_disabled_bag_size_keeps_backpack_detection_without_calling_geometry(self):
        from unittest.mock import patch
        box, bag = [100, 100, 200, 400], [120, 150, 180, 250]
        raw = make_result(yoloe=[det("person", .9, box), det("backpack", .8, bag)])
        with patch("trailcam.bags.classify_person_bags", side_effect=AssertionError("size expert disabled")), \
                patch("trailcam.bags.count_large_bags", side_effect=AssertionError("size expert disabled")):
            result = fuse(raw, large_bags=False)
        self.assertIsNone(result["combined"]["large_bags"])
        self.assertIsNone(result["combined"]["large_bags_uncertain"])
        self.assertEqual(result["combined"]["large_bags_status"], "disabled")
        self.assertEqual(result["combined"]["counts"]["backpacks"], 1)
        person = result["persons"][0]["combined"]
        self.assertIsNone(person["large_bag"])
        self.assertTrue(person["associated_backpack"])
        self.assertFalse(person["bag_assessed"])
        self.assertEqual(person["bag_reasons"], ["large_bag_size_disabled"])

    def test_experts_skipped_by_the_empty_frame_gate_report_no_counts(self):
        # v2: a skipped expert assessed nothing, so its counts are blank (None), not zero.
        result = make_result(yolo26n=[det("person", .15, [0, 0, 50, 150])])
        for name in ("yoloe", "pose"):
            result["experts"][name].update(skipped="empty_frame_gate",
                                           counts={"people_total": 0, "strollers": 0})  # as cached by vision
        fuse(result)
        for name in ("yoloe", "pose"):
            with self.subTest(expert=name):
                self.assertEqual(result["experts"][name]["counts"], dict.fromkeys(COUNT_FIELDS))
        self.assertEqual(result["experts"]["yolo26n"]["counts"]["people_total"], 0)  # ran, saw nothing countable
        self.assertEqual(result["experts"]["megadetector"]["counts"]["people_total"], 0)
        self.assertEqual(result["votes"]["strollers"]["status"], "no_eligible_output")
        self.assertEqual(result["votes"]["people_total"]["models"], ["yolo26n", "megadetector"])
        self.assertEqual(result["combined"]["counts"]["people_total"], 0)

    def test_loose_luggage_needs_confidence_and_is_counted_once(self):
        # v2: large_bags come from bags.count_large_bags; loose luggage must reach
        # luggage_confidence (.35), although it passes the .25 counting threshold.
        suitcase = [700, 300, 780, 420]
        weak = fuse(make_result(yoloe=[det("suitcase", .30, suitcase)]))
        self.assertEqual((weak["combined"]["large_bags"], weak["combined"]["large_bags_uncertain"]), (0, 0))
        at_threshold = fuse(make_result(yoloe=[det("duffel bag", .35, suitcase)]))
        self.assertEqual(at_threshold["combined"]["large_bags"], 1)
        # Two labels (or experts' boxes) on one physical bag count once.
        twins = fuse(make_result(yoloe=[det("suitcase", .6, suitcase), det("duffel bag", .5, [702, 301, 781, 421]),
                                        det("suitcase", .5, [100, 300, 180, 420])]))
        self.assertEqual(twins["combined"]["large_bags"], 2)

    def test_bag_results_are_returned_per_person(self):
        box = [400, 200, 500, 500]
        result = make_result(yoloe=[det("person", .9, box), det("backpack", .6, [420, 260, 480, 380])],
                             pose=[pose_det(.9, box)])
        people = [p for p in result["persons"] if roster.accept(p)[0]]
        detections = thresholded(result)
        bag_results = fusion._people(result, people, detections, OBJECT_POLICY)
        self.assertEqual(len(bag_results), 1)
        self.assertEqual(set(bag_results[0]), {"large_bag", "reasons", "features"})
        self.assertIsNone(bag_results[0]["large_bag"])  # mid-size pack: uncertain
        self.assertEqual(bag_results[0]["features"]["uncertain_bag_count"], 1)
        self.assertEqual(people[0]["combined"]["large_bag"], bag_results[0]["large_bag"])
        self.assertEqual(people[0]["combined"]["bag_detection_indices"], [1])

    def test_generic_animal_support_keeps_weak_dog_unverified_and_retains_evidence(self):
        box = [100, 500, 200, 600]
        # Wildlife can be confidently detected as a generic animal while a
        # dog-only classifier gives the same animal a weaker dog label.
        for confidence in (.3, .62, .86, .899):
            with self.subTest(confidence=confidence):
                raw = make_result(yoloe=[det("dog", confidence, box)],
                                  megadetector=[det("animal", .95, [102, 498, 201, 603])])
                result = fuse(raw)
                self.assertEqual(result["combined"]["counts"]["dogs"], 0)
                self.assertEqual(result["combined"]["unverified_counts"]["dogs"], 1)
                self.assertEqual(result["objects"]["dogs"][0]["experts"], ["yoloe"])
                self.assertEqual(result["experts"]["megadetector"]["counts"]["animals_total"], 1)
                self.assertEqual(result["experts"]["megadetector"]["detections"][0]["label"], "animal")
                self.assertTrue(result["needs_review"])

    def test_dog_needs_two_species_predictions_or_the_unchanged_single_threshold(self):
        box = [100, 500, 200, 600]
        corroborated = fuse(make_result(yoloe=[det("dog", .4, box)],
                                         yolo26n=[det("dog", .3, [102, 498, 201, 603])]))
        self.assertEqual(corroborated["combined"]["counts"]["dogs"], 1)
        self.assertEqual(corroborated["objects"]["dogs"][0]["status"], "corroborated")
        confident = fuse(make_result(yoloe=[det("dog", .9, box)]))
        self.assertEqual(confident["combined"]["counts"]["dogs"], 1)
        self.assertEqual(confident["objects"]["dogs"][0]["status"], "single_expert_confident")

    def test_luggage_carried_by_a_person_is_not_also_loose(self):
        box = [400, 200, 500, 500]
        suitcase = [410, 380, 490, 490]
        result = make_result(yoloe=[det("person", .9, box), det("suitcase", .6, suitcase)])
        result = fuse(result)
        person = result["persons"][0]
        self.assertIs(person["combined"]["large_bag"], True)
        self.assertEqual(result["combined"]["loose_luggage"], 0)
        self.assertEqual(result["combined"]["large_bags"], 1)

    def test_daypack_is_not_large(self):
        box = [400, 200, 500, 500]
        result = make_result(yoloe=[det("person", .9, box), det("backpack", .6, [420, 266, 480, 350])],
                             pose=[pose_det(.9, box, "back", "away")])
        result = fuse(result)
        combined = result["persons"][0]["combined"]
        self.assertIs(combined["large_bag"], False)
        self.assertEqual(result["combined"]["large_bags"], 0)
        self.assertEqual(result["combined"]["large_bags_uncertain"], 0)

    def test_rejected_people_do_not_count(self):
        result = fuse(make_result(yolo26n=[det("person", .3, [0, 0, 50, 150])], pose=[pose_det(.3, [200, 0, 250, 150])]))
        self.assertEqual(result["combined"]["counts"]["people_total"], 0)
        self.assertEqual(result["combined"]["facing_counts"], dict.fromkeys(DIRECTIONS, 0))
        self.assertTrue(all(p["accepted"] is False for p in result["persons"]))
        self.assertTrue(all("combined" not in p for p in result["persons"]))

    def test_side_facing_needs_pose_side(self):
        box = [400, 200, 500, 500]
        result = make_result(yoloe=[det("person", .9, box)], pose=[pose_det(.9, box, "side", "right")])
        result["persons"][0]["attributes"] = attributes("side")
        self.assertEqual(fuse(result)["persons"][0]["combined"]["facing"], "right")
        result = make_result(yoloe=[det("person", .9, box)])
        result["persons"][0]["attributes"] = attributes("side")
        fused = fuse(result)["persons"][0]["combined"]
        self.assertEqual((fused["orientation"], fused["facing"]), ("side", "unclear"))

    def test_module_constants(self):
        self.assertIs(fusion.fuse, fuse)
        self.assertEqual(fusion.TRAVEL_DIRECTIONS, ("left", "right", "toward", "away", "stationary", "unclear"))


if __name__ == "__main__":
    unittest.main()
