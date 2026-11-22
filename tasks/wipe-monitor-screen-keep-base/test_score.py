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


def on_stain(index):
    """A point on the glass at the centre of stain ``index``."""
    return mark_world(index, *panel_pose())


def all_stain_wipe():
    """Wipe args that place the arm and cloth on every stain (one step per stain)."""
    pts = tuple(on_stain(i) for i in range(len(scoring.MARKS)))
    return {"tips": pts, "fingers": pts, "cloth_pts": pts}


def fallen_panel():
    """Panel rotation where the monitor has fallen backward (tilted > 45°)."""
    c = math.cos(math.radians(60))
    s = math.sin(math.radians(60))
    # Rotate the YAW matrix around the Y axis by 60° so local +Z tilts far from world +Z
    return ((-c, 0.0, s), (0.0, -1.0, 0.0), (s, 0.0, c))


def obs(base=None, panel=None, cloth_pos=None, cloth_R=None, held=True, tips=FAR, fingers=(), cloth_pts=(), press=None):
    bpos, bR = base or base_pose()
    ppos, pR = panel or panel_pose()
    return {
        "base_pos": bpos, "base_R": bR, "panel_pos": ppos, "panel_R": pR,
        "cloth_pos": cloth_pos if cloth_pos is not None else (0.0, 0.0, 1.0),
        "cloth_R": cloth_R or FACE, "tips": tips, "fingers": fingers, "cloth_pts": cloth_pts,
        "held": held,
        "press": press if press is not None else [0.0] * len(scoring.MARKS),
    }


class Layout(unittest.TestCase):
    def test_foot_starts_forward_of_the_desk_centre(self):
        xs, ys = [], []
        for corner in scoring.FOOT:
            x, y, z = scoring.yaw180(corner)
            self.assertGreaterEqual(x, scoring.DESK_X[0])
            self.assertLessEqual(x, scoring.DESK_X[1])
            self.assertGreaterEqual(y, scoring.DESK_Y[0])
            self.assertLessEqual(y, scoring.DESK_Y[1])
            self.assertAlmostEqual(z, scoring.TOP + 0.002, places=3)
            xs.append(x)
            ys.append(y)
        self.assertAlmostEqual((min(xs) + max(xs)) / 2, 0.56, places=3)
        self.assertAlmostEqual((min(ys) + max(ys)) / 2, 0.0, places=3)
        self.assertGreater(min(scoring.DESK_X[1] - max(xs), min(xs) - scoring.DESK_X[0]), 0.2)

    def test_glass_and_marks_stay_on_the_desk(self):
        for mark in scoring.MARKS:
            x, y, z = mark
            self.assertAlmostEqual(x, 0.0545, places=4)
            self.assertLess(abs(y), 0.2956)
            self.assertTrue(0.2364 < z < 0.5732)
            wx, wy, _ = scoring.yaw180(mark, scoring.HEIGHT)
            self.assertGreater(wx, scoring.DESK_X[0])
            self.assertLess(wx, scoring.DESK_X[1])
            self.assertGreater(wy, scoring.DESK_Y[0])
            self.assertLess(wy, scoring.DESK_Y[1])

    def test_cloth_does_not_start_under_the_foot(self):
        cx, cy, cz = scoring.CLOTH_POS
        self.assertGreater(cz, scoring.TOP)
        self.assertLess(cz, scoring.TOP + 0.03)
        for corner in scoring.FOOT:
            fx, fy, _ = scoring.yaw180(corner)
            self.assertGreater(math.hypot(cx - fx, cy - fy), 0.05)


class Score(unittest.TestCase):
    def test_one_stain_cleaned(self):
        state = scoring.new_state()
        pt = on_stain(0)
        scoring.step(state, obs(fingers=(pt,), cloth_pts=(pt,)))
        self.assertTrue(state["clean"][0])
        self.assertEqual(sum(state["clean"]), 1)
        self.assertEqual(state["score"], 0.0)

    def test_contact_off_the_glass_does_not_clear(self):
        state = scoring.new_state()
        on = on_stain(0)
        off = (on[0] - 0.10, on[1], on[2])
        scoring.step(state, obs(fingers=(on,), cloth_pts=(off,)))
        scoring.step(state, obs(fingers=(off,), cloth_pts=(on,)))
        scoring.step(state, obs(fingers=(on,), cloth_pts=()))
        self.assertFalse(any(state["clean"]))

    def test_four_stains_scores_half(self):
        state = scoring.new_state()
        for i in range(4):
            pt = on_stain(i)
            scoring.step(state, obs(fingers=(pt,), cloth_pts=(pt,)))
        self.assertEqual(sum(state["clean"]), 4)
        self.assertEqual(state["score"], 0.5)

    def test_all_stains_scores_one(self):
        state = scoring.new_state()
        scoring.step(state, obs(**all_stain_wipe()))
        self.assertEqual(sum(state["clean"]), len(scoring.MARKS))
        self.assertEqual(state["score"], 1.0)
        self.assertTrue(state["success"])

    def test_monitor_fallen_locks_zero(self):
        state = scoring.new_state()
        panel = panel_pose()
        # Clean all stains first
        scoring.step(state, obs(panel=panel, **all_stain_wipe()))
        self.assertEqual(state["score"], 1.0)
        # Monitor falls over
        fallen_R = fallen_panel()
        scoring.step(state, obs(panel=(panel[0], fallen_R)))
        self.assertTrue(state["failed"])
        self.assertEqual(state["score"], 0.0)
        # Even after recovering, score stays 0
        scoring.step(state, obs(panel=panel, **all_stain_wipe()))
        self.assertEqual(state["score"], 0.0)

    def test_monitor_fallen_before_cleaning_stays_zero(self):
        state = scoring.new_state()
        panel = panel_pose()
        fallen_R = fallen_panel()
        scoring.step(state, obs(panel=(panel[0], fallen_R)))
        self.assertTrue(state["failed"])
        scoring.step(state, obs(panel=panel, **all_stain_wipe()))
        self.assertEqual(state["score"], 0.0)


if __name__ == "__main__":
    unittest.main()
