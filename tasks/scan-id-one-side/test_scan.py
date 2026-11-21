import unittest
from unittest.mock import patch

import numpy as np

import task
from labkit import PartState


def resting(position, quaternion=(0, 0, 0, 1)):
    return PartState(pos=np.array(position), quat_xyzw=np.array(quaternion), lin_vel=np.zeros(3),
                     ang_vel=np.zeros(3), low=np.zeros(3), high=np.zeros(3))


class ScannerTests(unittest.TestCase):
    def test_scene_is_one_chassis_and_the_card_stand(self):
        self.assertEqual([part.name for part in task.OBJECTS],
                         ["all_in_one", "card_stand", "card", "paper", "ink_front", "ink_back"])
        self.assertEqual({j.name for j in task.ALL_IN_ONE.joints}, {"lid", "lamp", "latch", "button", "roller"})
        np.testing.assert_allclose(task.CARD_PART.boxes[0][1], (.08560, .05398, .00076))
        np.testing.assert_allclose(task.PAPER_PART.boxes[0][1], (.297, .210, .00010))

    def test_either_face_can_be_the_one_that_is_copied(self):
        scanner = resting((task.SX, task.SY, task.TOP))
        center = scanner.pos + (0, 0, task.SEAT_Z)
        for quat, expected in [((0, 0, 0, 1), "back"), ((1, 0, 0, 0), "front")]:
            with self.subTest(face=expected):
                face, position = task._card_on_glass(resting(center, quat), scanner)
                self.assertEqual(face, expected)
                np.testing.assert_allclose(position, [0, 0])
        edge_up = resting(center, (2 ** -0.5, 0, 0, 2 ** -0.5))
        self.assertIsNone(task._card_on_glass(edge_up, scanner))

    def test_sheet_starts_leaning_in_the_rear_slit(self):
        self.assertEqual(task.PAPER_PART.rot_xyzw, task.STAND_Q)
        self.assertTrue(task._in_slit(resting(task.PAPER_START, task.STAND_Q)))
        self.assertFalse(task._in_slit(resting(task.PAPER_START, (2 ** -0.5, 0, 2 ** -0.5, 0))))
        lifted = np.array(task.PAPER_START) + (0, 0, .03)
        self.assertFalse(task._in_slit(resting(lifted, task.STAND_Q)))
        self.assertTrue(task._fully_out(resting((*task.PAPER_XY0, task.TOP + task.TRAY_Z))))

    def test_lid_is_passive(self):
        lid = task.ALL_IN_ONE.joints[0]
        self.assertEqual(lid.name, "lid")
        self.assertEqual(lid.stiffness, 0.0)
        self.assertGreater(lid.friction, 0.0)
        self.assertTrue(task._lid_down(task.LID_SHUT))
        self.assertFalse(task._lid_down(task.LID_OPEN))

    def test_partial_credit_is_only_a_finished_scan_then_the_copy(self):
        objects = {"paper": resting((0, 0, 0)), "card": resting((0, 0, 0)),
                   "ink_front": resting((0, 0, 0)), "ink_back": resting((0, 0, 0))}
        with patch.dict(task._S, task._fresh(), clear=True):
            task._S["lamp_dir"] = 1
            self.assertEqual(task.score(objects), 0)
            task._S["pending"]["front"] = (0.0, 0.05)
            self.assertEqual(task.score(objects), 0.25)


if __name__ == "__main__":
    unittest.main()
