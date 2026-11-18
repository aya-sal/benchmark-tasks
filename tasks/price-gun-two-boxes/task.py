"""Pricing bench: "Stick one price label from the price gun in the white price panel on top of each box."

Two light retail boxes, each with a white price panel printed on its top face, and a hand price-label
gun lying on a rest block. The gun is taken by its handle and squeeze lever; squeezing the lever home
feeds one label to the nose, touching the nose roller on a panel sticks it there, and the lever has to
be let out and squeezed again for the next label. The gun is put back down afterwards.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_here = Path(__file__).resolve().parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

import simkit  # noqa: E402
import gun as G  # noqa: E402
from scoring import TrialScorer  # noqa: E402

from yam_arena.constants import FINGER_OPEN, PAD_TIP_PAST_TCP as PAD_TIP  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402

NAME = "price-gun-two-boxes"
INSTRUCTION = json.loads((_here / "task.json").read_text())["instruction"]
HZ = 20
MAX_STEPS = 1600
OBJECT_NAMES = ("box_a", "box_b", "gun", "trigger", "roller")

DOWN = simkit.patch_demo()


def make_environment(embodiment, device, control_hz=HZ):
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene

    simkit.install()
    from yam_arena import scene as yscene

    objects = yscene.task_room("station", embodiment.rig.top_cam)
    specs = (G.box_object("box_a"), G.box_object("box_b"), G.rest_object(), G.gun_object(), G.trigger_object(),
             G.roller_object())
    objects += [yscene.task_object(s) for s in specs]
    cfg = IsaacLabArenaEnvironment(name="yam_price_gun_two_boxes", scene=Scene(objects), embodiment=embodiment,
                                   env_cfg_callback=yscene.env_cfg_callback(control_hz))
    return ArenaEnvBuilder(cfg, ArenaEnvBuilderCfg(num_envs=1, solve_relations=False, device=device)).make_registered()


class Dynamics:
    """Price labels, from measured poses only (a disclosed stand-in for a label feed and adhesive).

    The gun holds N_LABELS labels. Its squeeze lever is a real sprung hinge; its measured angle drives
    the feed. Squeezed to 85 % of its travel, the gun feeds one label to the nose (drawn hanging in
    front of the roller); the lever must come back out to 30 % before it can feed again. While a label
    is at the nose, the roller coming within 2 mm of a surface (a box face, the bench or the rest
    block) sticks the whole label there, centred under the roller with its long edge along the
    roller, and it stays on that body. Squeezing again with a label still at the nose pushes that
    label off as waste. There is no adhesion force, peel or wipe.
    """

    def __init__(self, env):
        import omni.usd
        self.base = env.unwrapped
        stage = omni.usd.get_context().get_stage()
        first = {**G.BOX_LABEL0, G.REST_NAME: G.REST_LABEL0}
        self.label_prims = {name: [None] * G.N_LABELS for name in first}
        self.nose = None
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            if path.endswith(f"/gun/piece{G.PRESENT_PIECE}"):
                self.nose = prim
            for name, k0 in first.items():
                for k in range(G.N_LABELS):
                    if path.endswith(f"/{name}/piece{k0 + k}"):
                        self.label_prims[name][k] = prim
        if self.nose is None or any(p is None for prims in self.label_prims.values() for p in prims):
            raise RuntimeError("label pieces not found on the stage")

    @staticmethod
    def _place(prim, pos, rot):
        """Draw a label piece at ``pos`` with rotation ``rot`` in its body's frame."""
        from pxr import Gf, UsdGeom
        ops = {op.GetOpType(): op.GetAttr() for op in UsdGeom.Xformable(prim).GetOrderedXformOps()}
        attr = ops[UsdGeom.XformOp.TypeTranslate]
        vec = Gf.Vec3f if str(attr.GetTypeName()) == "float3" else Gf.Vec3d
        attr.Set(vec(*map(float, pos)))
        attr = ops[UsdGeom.XformOp.TypeOrient]
        x, y, z, w = G.matrix_quat(np.asarray(rot, float))
        quat = Gf.Quatf if str(attr.GetTypeName()) == "quatf" else Gf.Quatd
        attr.Set(quat(float(w), float(x), float(y), float(z)))
        UsdGeom.Imageable(prim).MakeVisible()

    def reset(self):
        from pxr import UsdGeom
        for prims in self.label_prims.values():
            for prim in prims:
                UsdGeom.Imageable(prim).MakeInvisible()
        UsdGeom.Imageable(self.nose).MakeInvisible()
        self.fed = 0                 # labels that have left the roll
        self.at_nose = False         # one is fed and waiting at the nose
        self.armed = False           # the lever has come back out since the last feed
        self.wasted = 0
        self.labels = []             # stuck labels: surface, point, normal, rotation (in that surface's frame)

    def before_step(self, step, action):
        pass

    def _stick(self, surface, point, normal, axis):
        from pxr import UsdGeom
        rot = G.label_frame(normal, axis)
        k = len(self.labels)
        holder = surface if surface in G.BOX_NAMES else G.REST_NAME          # bench labels are drawn by the rest block
        local = np.asarray(point, float) + 0.0005 * np.asarray(normal, float)
        if holder == G.REST_NAME:
            local = local - G.rest_pos()
        self._place(self.label_prims[holder][k], local, rot)
        self.labels.append({"on": surface, "point": np.asarray(point, float), "normal": np.asarray(normal, float),
                            "rot": rot})
        self.at_nose = False
        UsdGeom.Imageable(self.nose).MakeInvisible()

    def after_step(self, step, objects, root_poses, robot):
        from pxr import UsdGeom
        frac = G.trigger_fraction(root_poses["gun"], root_poses["trigger"])
        if frac <= G.REARM_AT:
            self.armed = True
        elif self.armed and frac >= G.FEED_AT:
            self.armed = False
            if self.fed < G.N_LABELS:
                if self.at_nose:
                    self.wasted += 1                           # the waiting label is pushed off by the next one
                self.fed += 1
                self.at_nose = True
                UsdGeom.Imageable(self.nose).MakeVisible()
        if self.at_nose:
            touch = G.roller_touch(root_poses)
            if touch is not None:
                self._stick(*touch)
        out = {"trigger": round(frac, 3), "fed": self.fed, "labels_left": G.N_LABELS - self.fed,
               "at_nose": bool(self.at_nose), "wasted": self.wasted,
               "stray": sum(1 for lab in self.labels if lab["on"] not in G.BOX_NAMES), "labels": []}
        for name in G.BOX_NAMES:
            out[name] = {"labels": 0, "in_panel": 0, "margin_mm": None}
        for lab in self.labels:
            row = {"on": lab["on"], "at_mm": [round(1000 * float(v), 1) for v in lab["point"]]}
            if lab["on"] in G.BOX_NAMES:
                box = out[lab["on"]]
                box["labels"] += 1
                margin = G.panel_margin(lab["on"], lab["point"], lab["normal"], lab["rot"])
                if margin is not None:
                    row["panel_margin_mm"] = round(1000 * margin, 1)
                    box["margin_mm"] = row["panel_margin_mm"] if box["margin_mm"] is None else max(box["margin_mm"],
                                                                                                    row["panel_margin_mm"])
                    if margin >= 0:
                        box["in_panel"] += 1
            out["labels"].append(row)
        return out


def make_dynamics(env):
    return Dynamics(env)


def make_scorer():
    return TrialScorer()


# -- demo ----------------------------------------------------------------------------------

REST = {"left": (0.16, 0.24, G.TOP + 0.20), "right": (0.16, -0.24, G.TOP + 0.20)}
TCP_U = G.GRIP_U + PAD_TIP                                # the TCP above the gun's body centre: pad tips on the handle
F_WIDE = -0.030                                           # jaws 60 mm apart, to come down round handle and lever
F_SQUEEZE = -(G.SPAN_SHUT - 0.0075) / 2                   # lever squeezed home, about 19 N a side
F_HOLD = -(G.SPAN_OPEN - 0.0020) / 2                      # lever let right out, the gun still held by its spring
HOVER = 0.030                                             # roller above a panel before it comes down
NEAR = 0.006                                              # it pauses this far above the panel, to come down square
PRESS = 0.003                                             # and is then sent this far into it
# The arms hold position by joint stiffness, so the gun's weight lowers its nose a little, by an amount
# that depends on the reach. Measured on the settled arm, and added to the heights the plan asks for.
SAG = {"box_a": 0.0065, "box_b": 0.0090, "rest": 0.0040}
# The gun also hangs a degree or two off square in the jaws, and the loaded arm stops a few millimetres
# short: where the roller lands relative to where the plan sends it (x, y), measured and aimed off.
AIM = {"box_a": (0.0057, 0.0024), "box_b": (0.0056, 0.0036)}


def _hand(seconds, gun_xyz, finger, label, shut=True):
    """TCP holding the gun level with its body centre at ``gun_xyz``, nose toward +y."""
    f = G.GRIP_F_SHUT if shut else G.GRIP_F_OPEN
    tcp = np.asarray(gun_xyz, float) + G.L(f, 0.0, TCP_U)
    return Waypoint(seconds, tuple(map(float, tcp)), finger, label, DOWN)


def _gun_over(point, height):
    """Gun body centre that puts the roller's bottom ``height`` above ``point``."""
    return np.asarray(point, float) - G.L(G.ROLLER_AT[0], 0.0, G.ROLLER_AT[1] - G.ROLLER[0]) + np.array([0.0, 0.0, height])


def demo():
    up = np.array([0.0, 0.0, 1.0])
    g0 = G.gun_pos()
    # squeezing shifts the gun forward in the hand by half the lever's travel
    g1 = g0 + G.L(G.GRIP_F_OPEN - G.GRIP_F_SHUT, 0.0, 0.0)
    pa = G.panel_world("box_a") + np.array([*AIM["box_a"], 0.0])
    pb = G.panel_world("box_b") + np.array([*AIM["box_b"], 0.0])
    sa, sb, sr = SAG["box_a"], SAG["box_b"], SAG["rest"]
    carry = pb[2] - pa[2] + HOVER                         # over box A, high enough to clear box B's top

    def tap(panel, sag, which):
        return [
            _hand(1.0, _gun_over(panel, NEAR + sag), F_SQUEEZE, f"nose down to just over the {which} panel"),
            _hand(0.5, _gun_over(panel, NEAR + sag), F_SQUEEZE, "steady"),
            _hand(0.8, _gun_over(panel, sag - PRESS), F_SQUEEZE, "touch the roller on the panel: the label sticks"),
            _hand(0.3, _gun_over(panel, sag - PRESS), F_SQUEEZE, "press it on"),
        ]

    right = [
        Waypoint(1.5, REST["right"], FINGER_OPEN, "look", DOWN),
        _hand(2.0, g0 + 0.06 * up, F_WIDE, "over the price gun's handle", shut=False),
        _hand(1.2, g0, F_WIDE, "down round handle and lever", shut=False),
        _hand(0.8, g0, F_SQUEEZE, "squeeze: take it, and a label feeds to the nose", shut=False),
        _hand(1.2, g1 + 0.05 * up, F_SQUEEZE, "lift it off its rest"),
        _hand(2.2, _gun_over(pa, HOVER + sa), F_SQUEEZE, "over the first box's price panel"),
        *tap(pa, sa, "first"),
        _hand(1.2, _gun_over(pa, carry + sa), F_SQUEEZE, "lift off, high enough for the taller box"),
        _hand(0.7, _gun_over(pa, carry + sa), F_HOLD, "let the lever out"),
        _hand(0.4, _gun_over(pa, carry + sa), F_HOLD, "lever out"),
        _hand(0.7, _gun_over(pa, carry + sa), F_SQUEEZE, "squeeze again: the next label feeds"),
        _hand(2.2, _gun_over(pb, HOVER + sb), F_SQUEEZE, "over the second box's price panel"),
        *tap(pb, sb, "second"),
        _hand(1.0, _gun_over(pb, HOVER + sb), F_SQUEEZE, "lift off"),
        _hand(2.5, g1 + (0.05 + sr) * up, F_SQUEEZE, "back over the rest"),
        _hand(1.2, g1 + (0.005 + sr) * up, F_SQUEEZE, "down to just over the rest"),
        _hand(0.4, g1 + (0.005 + sr) * up, F_SQUEEZE, "steady"),
        _hand(0.6, g1 + (sr - 0.001) * up, F_SQUEEZE, "lay it on the rest"),
        _hand(0.8, g0 + (sr - 0.001) * up, F_WIDE, "let go", shut=False),
        _hand(0.8, g0 + 0.06 * up, F_WIDE, "lift off", shut=False),
        Waypoint(1.5, REST["right"], FINGER_OPEN, "stand clear", DOWN),
    ]
    # Each waypoint is its own move, eased to a stop: simkit's spline runs through consecutive waypoints
    # that share a finger target, and would dip below a point the plan dwells on (the touch on a panel).
    for i, w in enumerate(right):
        w.finger += (i % 2) * 2e-6
    left = [Waypoint(1.5, REST["left"], FINGER_OPEN, "stand by", DOWN)]
    t_r = sum(w.duration for w in right)
    t_l = sum(w.duration for w in left)
    left.append(Waypoint(t_r - t_l + 1.5, REST["left"], FINGER_OPEN, "done", DOWN))
    right.append(Waypoint(1.5, REST["right"], FINGER_OPEN, "done", DOWN))
    return {"left": left, "right": right}
