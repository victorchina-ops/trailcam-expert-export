"""Synthetic invariants, not accuracy evaluation of orientation on real people."""
import unittest

from trailcam.pose_geometry import match_canonical, orientation_from_keypoints


def skeleton(back=False):
    points = [[100, 100, 0.1] for _ in range(17)]
    points[0] = [100, 35, .95]
    points[1] = [110, 30, .95]
    points[2] = [90, 30, .95]
    points[5] = [140, 80, .95]
    points[6] = [60, 80, .95]
    points[11] = [125, 180, .95]
    points[12] = [75, 180, .95]
    if back:
        points[5], points[6] = points[6], points[5]
        points[11], points[12] = points[12], points[11]
        for i in range(5):
            points[i][2] = .1
    return points


def predict(points, **kwargs):
    return orientation_from_keypoints(points, [20, 10, 180, 300], 400, 400, **kwargs)


class GeometryTests(unittest.TestCase):
    def test_front_requires_face_and_compatible_torso(self):
        self.assertEqual(predict(skeleton())["orientation"], "front")
        p = skeleton()
        for i in range(5):
            p[i][2] = .1
        self.assertEqual(predict(p)["orientation"], "unknown")

    def test_face_absence_does_not_alone_mean_back(self):
        p = skeleton()
        for i in range(5):
            p[i][2] = .1
        self.assertEqual(predict(p)["orientation"], "unknown")
        self.assertEqual(predict(skeleton(back=True))["orientation"], "back")

    def test_torso_label_disagreement_abstains(self):
        p = skeleton()
        p[11], p[12] = p[12], p[11]
        self.assertEqual(predict(p)["orientation"], "unknown")

    def test_rear_torso_front_face_abstains(self):
        p = skeleton(back=True)
        p[:3] = skeleton()[:3]
        self.assertEqual(predict(p)["orientation"], "unknown")

    def test_low_confidence_and_small_box_abstain(self):
        self.assertEqual(predict(skeleton(), detection_confidence=.2)["orientation"], "unknown")
        p = skeleton()
        p[6][2] = .2
        self.assertEqual(predict(p)["orientation"], "unknown")
        self.assertEqual(predict(skeleton(), imgsz=48)["orientation"], "unknown")

    def test_profile_direction_and_horizontal_flip(self):
        p = skeleton()
        p[5], p[6] = [110, 80, .95], [90, 80, .95]
        p[11], p[12] = [110, 180, .95], [90, 180, .95]
        p[0], p[1], p[3] = [130, 38, .95], [120, 30, .95], [100, 32, .95]
        p[2][2] = .1
        self.assertEqual(predict(p)["orientation"], "side")
        self.assertEqual(predict(p)["facing_direction"], "right")
        p = [[200-x, y, c] for x, y, c in p]
        self.assertEqual(predict(p)["facing_direction"], "left")

    def test_no_pose_and_ambiguous_pose_association_abstain(self):
        canonical = [{"person_id": "p1", "xyxy": [0, 0, 10, 10]}]
        self.assertEqual(match_canonical(canonical, [])[0]["match_status"], "unmatched")
        poses = [{"xyxy": [0, 0, 10, 10], "orientation_evidence": {}} for _ in range(2)]
        self.assertEqual(match_canonical(canonical, poses)[0]["match_status"], "ambiguous")

    def test_no_many_to_one_match(self):
        canonical = [{"person_id": "p1", "xyxy": [0, 0, 10, 10]}, {"person_id": "p2", "xyxy": [0, 0, 10, 10]}]
        poses = [{"xyxy": [0, 0, 10, 10], "orientation_evidence": {}}]
        self.assertTrue(all(r["match_status"] == "ambiguous" for r in match_canonical(canonical, poses)))


if __name__ == "__main__":
    unittest.main()
