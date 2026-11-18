"""Stateful scorer for "Stick one price label from the price gun in the white price panel on top of each
box." From the labels reported by task.Dynamics, the measured poses of the boxes and the gun, and the
robot's measured TCPs."""
from __future__ import annotations

import math

import numpy as np

import gun as G

HZ = 20
HOLD_S = 1.0
MOVED_MM = 20.0                      # a box left further than this from where it started was shoved
TILT_DEG = 10.0                      # a box leaning more than this is not upright
GUN_DOWN_MM = 25.0                   # the gun's lowest point this near the bench: on the bench or its rest
HAND_OFF_GUN = 0.10                  # no gripper nearer than this to the gun's handle
HAND_OFF_BOX = 0.05                  # nor to either box


class TrialScorer:
    def __init__(self):
        self.held = 0
        self.step = 0

    def update(self, step, objects, root_poses, robot, mechanics=None, **_):
        if step != self.step + 1:
            raise ValueError("scorer needs one update per control step")
        if mechanics is None:
            raise ValueError("label measurements missing")
        self.step = step
        m = mechanics
        on = {n: m[n]["labels"] for n in G.BOX_NAMES}
        priced = {n: m[n]["in_panel"] >= 1 for n in G.BOX_NAMES}
        # exactly one label on each box and none anywhere else: two labels gone from the gun, both on boxes
        exact = all(on[n] == 1 for n in G.BOX_NAMES) and m["fed"] == 2
        moved = {n: 1000 * math.dist(np.asarray(root_poses[n])[:2], G.box_pos(n)[:2]) for n in G.BOX_NAMES}
        tilt = {n: G.box_tilt(root_poses[n]) for n in G.BOX_NAMES}
        boxes_ok = all(moved[n] <= MOVED_MM and tilt[n] <= TILT_DEG for n in G.BOX_NAMES)
        tcps = [np.asarray(robot[s]["tcp"], float)[:3] for s in ("left", "right")]
        gun_low = 1000 * G.gun_lowest(root_poses["gun"], root_poses["roller"])
        handle = G.handle_centre(root_poses["gun"])
        gun_down = gun_low <= GUN_DOWN_MM and all(math.dist(t, handle) > HAND_OFF_GUN for t in tcps)
        clear = all(G.box_distance(n, root_poses[n], t)[0] > HAND_OFF_BOX for n in G.BOX_NAMES for t in tcps)
        score = 0.0
        if sum(on.values()) >= 1:
            score = 0.25
        if any(priced.values()):
            score = 0.5
        if all(priced.values()):
            score = 0.75
        full = score == 0.75 and exact and boxes_ok and gun_down and clear
        self.held = self.held + 1 if full else 0
        if full and self.held >= HOLD_S * HZ:
            score = 1.0
        return {"score": score, "success": score == 1.0,
                "labels_on": on, "in_panel": {n: bool(v) for n, v in priced.items()},
                "labels_fed": m["fed"], "one_each_none_spare": bool(exact),
                "box_moved_mm": {n: round(v, 1) for n, v in moved.items()},
                "box_tilt_deg": {n: round(v, 1) for n, v in tilt.items()}, "boxes_in_place": bool(boxes_ok),
                "gun_lowest_mm": round(gun_low, 1), "gun_put_down": bool(gun_down), "hands_clear": bool(clear),
                "held_frames": self.held}
