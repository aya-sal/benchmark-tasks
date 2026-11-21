"""Geometry and scoring checks that do not need Isaac Sim."""

import math
import unittest

import scoring

I = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
YAW = ((-1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0))
# Thin axis (column 2) points toward the robot, along the glass normal in the world.
FACE = ((0.0, 0.0, -1.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0))
FAR = ((0.0, -0.4, 1.3), (0.0, -0.5, 1.3))


def panel_pose():
    ox, oy, oz = scoring.ORIGIN
    return (ox, oy, oz + scoring.HEIGHT), YAW


def base_pose(shift=(0.0, 0.0, 0.0)):
    return tuple(scoring.ORIGIN[i] + shift[i] for i in range(3)), YAW


def mark_world(index, panel_pos, panel_R):
    return scoring._add(panel_pos, scoring._mul(panel_R, scoring.MARKS[index]))


def obs(base=None, panel=None, cloth_pos=None, cloth_R=None, held=True, tips=FAR):
    bpos, bR = base or base_pose()
    ppos, pR = panel or panel_pose()
    return {
        "base_pos": bpos, "base_R": bR, "panel_pos": ppos, "panel_R": pR,
        "cloth_pos": cloth_pos if cloth_pos is not None else (0.0, 0.0, 1.0),
        "cloth_R": cloth_R or FACE, "tips": tips, "held": held,
    }


class Layout(unittest.TestCase):
    def test_foot_starts_on_the_desk_with_a_short_left_margin(self):
        margins = []
        for corner in scoring.FOOT:
            x, y, z = scoring.yaw180(corner)
            self.assertGreaterEqual(x, scoring.DESK_X[0])
            self.assertLessEqual(x, scoring.DESK_X[1])
            self.assertGreaterEqual(y, scoring.DESK_Y[0])
            self.assertLess(y, scoring.DESK_Y[1])
            self.assertAlmostEqual(z, scoring.TOP + 0.002, places=3)
            margins.append(scoring.DESK_Y[1] - y)
        self.assertAlmostEqual(min(margins), 0.030, places=3)
        self.assertGreater(max(margins), 0.25)

    def test_glass_overhangs_the_left_edge_and_marks_stay_on_the_glass(self):
        overhang = False
        for mark in scoring.MARKS:
            x, y, z = mark
            self.assertAlmostEqual(x, 0.0545, places=4)
            self.assertLess(abs(y), 0.2956)
            self.assertTrue(0.2364 < z < 0.5732)
            if scoring.yaw180(mark, scoring.HEIGHT)[1] > scoring.DESK_Y[1]:
                overhang = True
        self.assertTrue(overhang)

    def test_cloth_does_not_start_under_the_foot(self):
        cx, cy, cz = scoring.CLOTH_POS
        self.assertAlmostEqual(cz, scoring.TOP + scoring.CLOTH_SIZE[2] / 2, places=4)
        for corner in scoring.FOOT:
            fx, fy, _ = scoring.yaw180(corner)
            self.assertGreater(math.hypot(cx - fx, cy - fy), 0.05)


class Score(unittest.TestCase):
    def test_a_stroke_clears_one_mark_and_four_marks_score_a_quarter(self):
        state = scoring.new_state()
        panel = panel_pose()
        for i in range(4):
            mark = mark_world(i, *panel)
            scoring.step(state, obs(panel=panel, cloth_pos=mark, held=True))
            nudged = (mark[0], mark[1], mark[2] - scoring.SLIDE_M)
            scoring.step(state, obs(panel=panel, cloth_pos=nudged, held=True))
        self.assertEqual(sum(state["clean"]), 4)
        self.assertEqual(state["score"], 0.25)

    def test_a_poke_or_a_loose_pad_does_not_count(self):
        state = scoring.new_state()
        panel = panel_pose()
        mark = mark_world(0, *panel)
        scoring.step(state, obs(panel=panel, cloth_pos=mark, held=True))
        scoring.step(state, obs(panel=panel, cloth_pos=mark, held=True))
        self.assertFalse(state["clean"][0])
        scoring.step(state, obs(panel=panel, cloth_pos=mark, held=False))
        scoring.step(state, obs(panel=panel, cloth_pos=(mark[0], mark[1], mark[2] - 0.04), held=False))
        self.assertFalse(state["clean"][0])

    def test_the_pad_must_face_the_glass(self):
        state = scoring.new_state()
        panel = panel_pose()
        mark = mark_world(0, *panel)
        scoring.step(state, obs(panel=panel, cloth_pos=mark, cloth_R=I, held=True))
        scoring.step(state, obs(panel=panel, cloth_pos=(mark[0], mark[1], mark[2] - 0.04), cloth_R=I, held=True))
        self.assertFalse(state["clean"][0])

    def test_full_score_needs_every_mark_hands_clear_and_a_two_second_hold(self):
        state = scoring.new_state()
        panel = panel_pose()
        on_the_stand = (scoring.ORIGIN,) * 2
        for i in range(len(scoring.MARKS)):
            mark = mark_world(i, *panel)
            scoring.step(state, obs(panel=panel, cloth_pos=mark, tips=on_the_stand))
            scoring.step(state, obs(
                panel=panel, cloth_pos=(mark[0], mark[1], mark[2] - scoring.SLIDE_M), tips=on_the_stand,
            ))
        self.assertEqual(state["score"], 0.5)
        for _ in range(scoring.HOLD_STEPS - 1):
            scoring.step(state, obs(panel=panel, cloth_pos=(0.2, 0.0, 0.9)))
            self.assertFalse(state["success"])
        scoring.step(state, obs(panel=panel, cloth_pos=(0.2, 0.0, 0.9)))
        self.assertTrue(state["success"])
        self.assertEqual(state["score"], 1.0)

    def test_a_hand_left_on_the_monitor_blocks_full_score(self):
        state = scoring.new_state()
        panel = panel_pose()
        for i in range(len(scoring.MARKS)):
            mark = mark_world(i, *panel)
            scoring.step(state, obs(panel=panel, cloth_pos=mark, tips=FAR))
            scoring.step(state, obs(panel=panel, cloth_pos=(mark[0], mark[1], mark[2] - scoring.SLIDE_M), tips=FAR))
        on_foot = (scoring.ORIGIN[0], scoring.ORIGIN[1], scoring.ORIGIN[2] + 0.05)
        for _ in range(scoring.HOLD_STEPS):
            scoring.step(state, obs(panel=panel, cloth_pos=(0.2, 0.0, 0.9), tips=(on_foot, FAR[1])))
        self.assertEqual(state["score"], 0.5)
        self.assertFalse(state["success"])

    def test_leaving_the_desk_locks_the_score_at_zero(self):
        state = scoring.new_state()
        panel = panel_pose()
        mark = mark_world(0, *panel)
        scoring.step(state, obs(panel=panel, cloth_pos=mark))
        scoring.step(state, obs(panel=panel, cloth_pos=(mark[0], mark[1], mark[2] - scoring.SLIDE_M)))
        self.assertEqual(state["score"], 0.0)
        shoved = base_pose((0.0, 0.05, 0.0))  # 30 mm of margin, plus 50 mm, puts a corner past the edge
        scoring.step(state, obs(base=shoved, panel=panel, cloth_pos=(0.2, 0.0, 0.9)))
        self.assertTrue(state["failed"])
        self.assertEqual(state["score"], 0.0)
        for i in range(len(scoring.MARKS)):
            m = mark_world(i, *panel)
            scoring.step(state, obs(base=base_pose(), panel=panel, cloth_pos=m))
            scoring.step(state, obs(base=base_pose(), panel=panel, cloth_pos=(m[0], m[1], m[2] - scoring.SLIDE_M)))
        self.assertEqual(state["score"], 0.0)
        self.assertFalse(state["success"])


if __name__ == "__main__":
    unittest.main()
