"""Synthetic checks for the multi-expert person roster (no models, no photos).

Covers candidate clustering (one member per expert, representative box,
confidence, ordering and ids), data-strip exclusion, input validation, the
post-processing acceptance rule (per-expert single thresholds and the
two-expert pair rule) and determinism.
"""
from __future__ import annotations

import copy
import json
import math
import random
import unittest

from trailcam import roster
from trailcam.roster import EXPERT_ORDER, POLICY, accept, box_iou, build_candidates, strip_fraction


def det(label, confidence, box, index=None):
    d = {"label": label, "confidence": confidence, "xyxy": list(box)}
    if index is not None:
        d["detection_index"] = index
    return d


def person(confidence, box, index=None):
    return det("person", confidence, box, index)


def experts(**lists):
    return {name: {"detections": list(dets)} for name, dets in lists.items()}


def member(confidence, box=(0, 0, 10, 20), index=0):
    return {"detection_index": index, "confidence": confidence, "xyxy": list(box)}


def candidate(**confidences):
    return {"members": {name: member(conf) for name, conf in confidences.items()}}


class BoxIouTests(unittest.TestCase):
    def test_identical_disjoint_and_partial(self):
        self.assertEqual(box_iou([0, 0, 10, 10], [0, 0, 10, 10]), 1.0)
        self.assertEqual(box_iou([0, 0, 10, 10], [20, 20, 30, 30]), 0.0)
        self.assertEqual(box_iou([0, 0, 10, 10], [10, 0, 20, 10]), 0.0)  # touching edge
        self.assertAlmostEqual(box_iou([0, 0, 10, 10], [5, 0, 15, 10]), 50 / 150)

    def test_degenerate_boxes_give_zero(self):
        self.assertEqual(box_iou([0, 0, 0, 0], [0, 0, 0, 0]), 0.0)

    def test_symmetric(self):
        a, b = [3, 4, 50, 60], [10, 1, 40, 90]
        self.assertEqual(box_iou(a, b), box_iou(b, a))


class StripFractionTests(unittest.TestCase):
    def test_no_strip(self):
        for strip in (None, {}, {"top": 0, "bottom": 0}):
            self.assertEqual(strip_fraction([0, 900, 10, 1000], strip, 1000), 0.0)

    def test_bottom_strip_share(self):
        strip = {"top": 0, "bottom": 50}
        self.assertEqual(strip_fraction([0, 960, 10, 1000], strip, 1000), 1.0)
        self.assertAlmostEqual(strip_fraction([0, 900, 10, 1000], strip, 1000), 0.5)
        self.assertAlmostEqual(strip_fraction([0, 850, 10, 1000], strip, 1000), 50 / 150)
        self.assertEqual(strip_fraction([0, 100, 10, 900], strip, 1000), 0.0)

    def test_top_strip_share(self):
        strip = {"top": 40, "bottom": 0}
        self.assertAlmostEqual(strip_fraction([0, 0, 10, 80], strip, 1000), 0.5)
        self.assertEqual(strip_fraction([0, 50, 10, 80], strip, 1000), 0.0)

    def test_zero_area_box(self):
        self.assertEqual(strip_fraction([5, 5, 5, 50], {"bottom": 50}, 100), 0.0)


class BuildCandidatesTests(unittest.TestCase):
    def test_one_candidate_per_person_across_experts(self):
        yoloe_box, pose_box, n_box = [100, 100, 200, 400], [102, 98, 203, 405], [98, 102, 198, 398]
        candidates = build_candidates(experts(yolo26n=[person(.5, n_box)], pose=[person(.6, pose_box)],
                                              yoloe=[person(.8, yoloe_box)]))
        self.assertEqual(len(candidates), 1)
        c = candidates[0]
        self.assertEqual(list(c["members"]), ["yoloe", "pose", "yolo26n"])  # EXPERT_ORDER
        self.assertEqual(c["xyxy"], yoloe_box)          # YOLOE is the representative (masks)
        self.assertEqual(c["bbox_xyxy"], yoloe_box)
        self.assertIsNot(c["xyxy"], c["bbox_xyxy"])
        self.assertEqual(c["confidence"], .8)
        self.assertEqual(c["candidate_id"], "cand_0001")
        self.assertEqual(c["members"]["pose"], {"detection_index": 0, "confidence": .6, "xyxy": pose_box})
        self.assertEqual(c["members"]["yolo26n"]["xyxy"], n_box)

    def test_representative_follows_expert_order_and_confidence_is_max(self):
        pose_box, n_box = [100, 100, 200, 400], [101, 101, 199, 401]
        c = build_candidates(experts(pose=[person(.3, pose_box)], yolo26n=[person(.95, n_box)]))[0]
        self.assertEqual(c["xyxy"], pose_box)   # pose (keypoints) preferred over yolo26n
        self.assertEqual(c["confidence"], .95)
        c = build_candidates(experts(yolo26n=[person(.7, n_box)]))[0]
        self.assertEqual(c["xyxy"], n_box)
        self.assertEqual(list(c["members"]), ["yolo26n"])

    def test_megadetector_and_other_labels_are_not_people(self):
        box = [100, 100, 200, 400]
        self.assertNotIn("megadetector", EXPERT_ORDER)
        self.assertEqual(build_candidates(experts(megadetector=[person(.99, box)])), [])
        self.assertEqual(build_candidates(experts(yoloe=[det("dog", .99, box), det("backpack", .9, box)])), [])
        candidates = build_candidates(experts(yoloe=[person(.4, box)], megadetector=[person(.99, box)]))
        self.assertEqual(len(candidates), 1)
        self.assertEqual(list(candidates[0]["members"]), ["yoloe"])
        self.assertEqual(candidates[0]["confidence"], .4)

    def test_raw_confidence_floor(self):
        floor = POLICY["raw_floor"]
        kept = build_candidates(experts(yoloe=[person(floor, [0, 0, 10, 20])]))
        self.assertEqual(len(kept), 1)
        self.assertEqual(build_candidates(experts(yoloe=[person(floor - 1e-4, [0, 0, 10, 20])])), [])

    def test_non_finite_and_non_numeric_confidences_are_skipped(self):
        # v2: NaN passed the raw-floor comparison (NaN < floor is False) and
        # could reorder or join clusters; non-finite confidences are now skipped.
        box = [100, 100, 200, 400]
        bad = [person(math.nan, box), person(math.inf, box), person(-math.inf, box), person(None, box),
               person("0.9", box)]
        self.assertEqual(build_candidates(experts(yoloe=bad, pose=[person(math.nan, box)])), [])
        kept = build_candidates(experts(yoloe=bad + [person(.5, box)], pose=[person(math.nan, box), person(.6, box)]))
        self.assertEqual(len(kept), 1)
        self.assertEqual({name: m["detection_index"] for name, m in kept[0]["members"].items()},
                         {"yoloe": len(bad), "pose": 1})
        self.assertEqual(kept[0]["confidence"], .6)
        json.dumps(kept, allow_nan=False)

    def test_invalid_boxes_are_skipped(self):
        bad = [[math.nan, 0, 10, 10], [0, math.inf, 10, 10], [10, 0, 10, 20], [0, 20, 10, 20],
               [10, 0, 5, 20], [0, 30, 10, 20]]
        self.assertEqual(build_candidates(experts(yoloe=[person(.9, b) for b in bad])), [])
        kept = build_candidates(experts(yoloe=[person(.9, b) for b in bad] + [person(.9, [0, 0, 10, 20])]))
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["members"]["yoloe"]["detection_index"], len(bad))

    def test_one_member_per_expert_per_cluster(self):
        a, b = [100, 100, 200, 400], [105, 100, 205, 400]  # IoU ~0.9: two people side by side
        pose_box = [106, 100, 206, 400]                    # closer to b
        candidates = build_candidates(experts(yoloe=[person(.9, a), person(.8, b)], pose=[person(.7, pose_box)]))
        self.assertEqual(len(candidates), 2)
        by_conf = {c["confidence"]: c for c in candidates}
        self.assertEqual(list(by_conf[.9]["members"]), ["yoloe"])
        self.assertEqual(list(by_conf[.8]["members"]), ["yoloe", "pose"])
        self.assertEqual(by_conf[.8]["members"]["yoloe"]["xyxy"], b)

    def test_duplicate_box_from_same_expert_starts_new_candidate(self):
        box = [100, 100, 200, 400]
        candidates = build_candidates(experts(yoloe=[person(.9, box)],
                                              pose=[person(.8, box), person(.7, [101, 100, 201, 400])]))
        self.assertEqual(len(candidates), 2)
        self.assertEqual(list(candidates[0]["members"]), ["yoloe", "pose"])
        self.assertEqual(candidates[0]["members"]["pose"]["confidence"], .8)
        self.assertEqual(list(candidates[1]["members"]), ["pose"])
        self.assertEqual(candidates[1]["confidence"], .7)

    def test_match_iou_threshold_is_inclusive(self):
        self.assertEqual(POLICY["match_iou"], .45)
        base = [0, 0, 100, 100]
        joined = build_candidates(experts(yoloe=[person(.9, base)], pose=[person(.8, [0, 0, 100, 45])]))
        self.assertEqual(len(joined), 1)  # IoU exactly 0.45
        split = build_candidates(experts(yoloe=[person(.9, base)], pose=[person(.8, [0, 0, 100, 44])]))
        self.assertEqual(len(split), 2)   # IoU 0.44

    def test_joins_best_overlapping_cluster(self):
        a, b = [0, 0, 100, 200], [300, 0, 400, 200]
        pose_box = [290, 0, 400, 200]
        candidates = build_candidates(experts(yoloe=[person(.9, a), person(.85, b)], pose=[person(.6, pose_box)]))
        self.assertEqual(len(candidates), 2)
        second = next(c for c in candidates if c["xyxy"] == b)
        self.assertIn("pose", second["members"])

    def test_data_strip_exclusion(self):
        strip, height = {"top": 0, "bottom": 60}, 1000
        inside = person(.95, [10, 950, 50, 1000])       # entirely in the strip rows (940..1000)
        mostly = person(.95, [100, 900, 150, 1000])     # 60 % in the strip
        half = person(.95, [200, 880, 250, 1000])       # exactly 50 % in the strip
        partly = person(.95, [300, 800, 350, 1000])     # 30 % in the strip
        scene = person(.95, [400, 100, 450, 400])
        candidates = build_candidates(experts(yoloe=[inside, mostly, half, partly, scene]), strip, height)
        self.assertEqual(sorted(c["xyxy"][0] for c in candidates), [300, 400])
        top = build_candidates(experts(yoloe=[person(.9, [0, 0, 50, 30])]), {"top": 40, "bottom": 0}, height)
        self.assertEqual(top, [])
        # A different strip_fraction policy moves the cut.
        loose = build_candidates(experts(yoloe=[half, partly]), strip, height, {**POLICY, "strip_fraction": .6})
        self.assertEqual(sorted(c["xyxy"][0] for c in loose), [200, 300])

    def test_detection_index_explicit_or_list_position(self):
        c = build_candidates(experts(yoloe=[person(.9, [0, 0, 10, 20], index=7)]))[0]
        self.assertEqual(c["members"]["yoloe"]["detection_index"], 7)
        cs = build_candidates(experts(pose=[det("dog", .9, [0, 0, 5, 5]), person(.9, [50, 0, 60, 20])]))
        self.assertEqual(cs[0]["members"]["pose"]["detection_index"], 1)

    def test_candidates_sorted_by_confidence_with_sequential_ids(self):
        dets = [person(.5, [0, 0, 10, 20]), person(.9, [100, 0, 110, 20]), person(.7, [200, 0, 210, 20]),
                person(.7, [150, 0, 160, 20])]
        candidates = build_candidates(experts(yoloe=dets))
        self.assertEqual([c["confidence"] for c in candidates], [.9, .7, .7, .5])
        self.assertEqual([c["xyxy"][0] for c in candidates], [100, 150, 200, 0])  # ties by x1
        self.assertEqual([c["candidate_id"] for c in candidates], ["cand_0001", "cand_0002", "cand_0003", "cand_0004"])

    def test_empty_and_missing_experts(self):
        self.assertEqual(build_candidates({}), [])
        self.assertEqual(build_candidates({"yoloe": {}}), [])
        self.assertEqual(build_candidates(experts(yoloe=[], pose=[], yolo26n=[])), [])

    def test_custom_policy(self):
        box_a, box_b = [0, 0, 100, 100], [0, 0, 100, 80]  # IoU .8
        dets = experts(yoloe=[person(.3, box_a)], pose=[person(.3, box_b)])
        self.assertEqual(len(build_candidates(dets)), 1)
        self.assertEqual(len(build_candidates(dets, policy={**POLICY, "match_iou": .9})), 2)
        self.assertEqual(build_candidates(dets, policy={**POLICY, "raw_floor": .5}), [])

    def test_input_is_not_mutated_and_outputs_are_copies(self):
        source = experts(yoloe=[person(.9, [0, 0, 10, 20], 0)], pose=[person(.8, [0, 0, 10, 21], 0)])
        before = copy.deepcopy(source)
        candidates = build_candidates(source)
        self.assertEqual(source, before)
        candidates[0]["xyxy"][0] = 999
        candidates[0]["members"]["pose"]["xyxy"][0] = 999
        self.assertEqual(source, before)

    def test_deterministic_under_input_order_and_json_serialisable(self):
        rng = random.Random(42)
        lists = {name: [] for name in EXPERT_ORDER}
        for n in range(12):
            x = 120 * n
            for name in EXPERT_ORDER:
                if rng.random() < .8:
                    jitter = [rng.uniform(-6, 6) for _ in range(4)]
                    box = [x + jitter[0], 50 + jitter[1], x + 80 + jitter[2], 300 + jitter[3]]
                    lists[name].append(person(round(rng.uniform(.1, .95), 3), box, index=len(lists[name])))
        lists["pose"].append(person(.5, [5, 50, 85, 300], index=len(lists["pose"])))  # a duplicate pose
        reference = build_candidates(experts(**lists), {"top": 0, "bottom": 30}, 1000)
        self.assertEqual(json.loads(json.dumps(reference, allow_nan=False)), reference)
        for _ in range(5):
            shuffled = {name: rng.sample(dets, len(dets)) for name, dets in lists.items()}
            order = list(shuffled)
            rng.shuffle(order)
            self.assertEqual(build_candidates({name: {"detections": shuffled[name]} for name in order},
                                              {"top": 0, "bottom": 30}, 1000), reference)

    def test_every_candidate_has_at_most_one_member_per_expert(self):
        rng = random.Random(7)

        def random_box():
            x, y = rng.uniform(0, 500), rng.uniform(0, 300)
            return [x, y, x + rng.uniform(20, 80), y + rng.uniform(60, 200)]

        lists = {name: [person(round(rng.uniform(.12, 1), 3), random_box(), index=i) for i in range(15)]
                 for name in EXPERT_ORDER}
        candidates = build_candidates(experts(**lists))
        used = {name: [] for name in EXPERT_ORDER}
        for c in candidates:
            self.assertTrue(set(c["members"]) <= set(EXPERT_ORDER))
            self.assertEqual(c["confidence"], max(m["confidence"] for m in c["members"].values()))
            for name, m in c["members"].items():
                used[name].append(m["detection_index"])
        for name in EXPERT_ORDER:  # every detection lands in exactly one candidate
            self.assertEqual(sorted(used[name]), list(range(15)))


class AcceptTests(unittest.TestCase):
    def test_policy_documents_per_expert_single_thresholds(self):
        single = POLICY["accept_single"]
        self.assertEqual(set(single), set(EXPERT_ORDER))
        self.assertLess(single["yoloe"], single["pose"])     # YOLOE is the primary counter
        self.assertLessEqual(POLICY["accept_pair"], min(single.values()))
        self.assertGreaterEqual(POLICY["accept_pair"], POLICY["raw_floor"])

    def test_single_expert_threshold_is_inclusive_per_expert(self):
        for name, threshold in POLICY["accept_single"].items():
            with self.subTest(expert=name):
                self.assertEqual(accept(candidate(**{name: threshold})),
                                 (True, "single_expert_confident:" + name))
                self.assertEqual(accept(candidate(**{name: threshold - 1e-3})),
                                 (False, "insufficient_confidence_or_support"))

    def test_same_confidence_differs_by_expert(self):
        value = POLICY["accept_single"]["yoloe"]
        self.assertTrue(accept(candidate(yoloe=value))[0])
        self.assertFalse(accept(candidate(pose=value))[0])
        self.assertFalse(accept(candidate(yolo26n=value))[0])

    def test_pair_rule(self):
        pair = POLICY["accept_pair"]
        self.assertEqual(accept(candidate(yoloe=pair, pose=pair)), (True, "multi_expert_agreement:yoloe+pose"))
        self.assertEqual(accept(candidate(pose=pair, yolo26n=pair)), (True, "multi_expert_agreement:pose+yolo26n"))
        self.assertEqual(accept(candidate(yoloe=pair, pose=pair, yolo26n=pair)),
                         (True, "multi_expert_agreement:yoloe+pose+yolo26n"))
        self.assertEqual(accept(candidate(yoloe=pair, pose=pair - 1e-3)),
                         (False, "insufficient_confidence_or_support"))
        # Only the experts at >= accept_pair are named.
        self.assertEqual(accept(candidate(yoloe=pair, pose=pair, yolo26n=pair - 1e-3)),
                         (True, "multi_expert_agreement:yoloe+pose"))

    def test_confident_single_takes_precedence_and_lists_all_strong(self):
        single = POLICY["accept_single"]
        self.assertEqual(accept(candidate(yoloe=.9, pose=POLICY["accept_pair"])),
                         (True, "single_expert_confident:yoloe"))
        self.assertEqual(accept(candidate(yoloe=single["yoloe"], pose=single["pose"], yolo26n=.2)),
                         (True, "single_expert_confident:yoloe+pose"))

    def test_expert_without_threshold_never_accepts_or_pairs(self):
        self.assertEqual(accept(candidate(megadetector=.99)), (False, "insufficient_confidence_or_support"))
        self.assertFalse(accept(candidate(megadetector=.99, yoloe=POLICY["accept_pair"]))[0])

    def test_empty_candidates_are_rejected(self):
        self.assertEqual(accept({"members": {}}), (False, "insufficient_confidence_or_support"))
        self.assertEqual(accept({}), (False, "insufficient_confidence_or_support"))

    def test_scalar_single_threshold_and_custom_pair_experts(self):
        policy = {**POLICY, "accept_single": .5}
        self.assertTrue(accept(candidate(yolo26n=.5), policy)[0])
        self.assertFalse(accept(candidate(yoloe=.49), policy)[0])
        policy = {**POLICY, "pair_experts": ("yoloe", "pose")}
        self.assertFalse(accept(candidate(pose=.2, yolo26n=.2), policy)[0])
        self.assertTrue(accept(candidate(yoloe=.13, pose=.2), policy)[0])

    def test_build_then_accept_scene(self):
        strip, height = {"top": 0, "bottom": 40}, 800
        scene = experts(
            yoloe=[person(.85, [100, 100, 200, 400]),        # strong YOLOE alone
                   person(.15, [300, 100, 400, 400]),        # weak, but pose agrees
                   person(.99, [600, 770, 650, 800])],       # inside the data strip
            pose=[person(.14, [302, 98, 401, 402]),
                  person(.30, [500, 100, 560, 300])],        # weak lone pose
            yolo26n=[person(.2, [700, 100, 760, 300])],      # weak lone yolo26n
            megadetector=[person(.95, [700, 100, 760, 300])])
        candidates = build_candidates(scene, strip, height)
        decisions = {c["xyxy"][0]: accept(c) for c in candidates}
        self.assertEqual(set(decisions), {100, 300, 500, 700})
        self.assertEqual(decisions[100], (True, "single_expert_confident:yoloe"))
        self.assertEqual(decisions[300], (True, "multi_expert_agreement:yoloe+pose"))
        self.assertFalse(decisions[500][0])
        self.assertFalse(decisions[700][0])  # MegaDetector corroboration does not create people

    def test_module_reexports(self):
        self.assertIs(roster.accept, accept)
        self.assertEqual(POLICY["version"], "roster_v2")


if __name__ == "__main__":
    unittest.main()
