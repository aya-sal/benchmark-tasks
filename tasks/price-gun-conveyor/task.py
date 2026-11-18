"""Pricing line: "Stick one price label from the price gun in the white price panel on top of each box as
the boxes ride past on the belt."

Two light retail boxes, each with a white price panel printed on its top face, ride steel carriers on
a slow belt conveyor past the arms, and a hand price-label gun lies on a rest block beside the belt.
The gun is taken by its handle and squeeze lever; squeezing the lever home feeds one label to the
nose, touching the nose roller on a panel sticks it there, and the lever has to be let out and
squeezed again for the next label. The belt does not stop: each label goes on while its box is
moving, and the gun is put back down before the first box reaches the end of the belt.
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
import conveyor as C  # noqa: E402
import gun as G  # noqa: E402
from scoring import TrialScorer  # noqa: E402

from yam_arena.constants import FINGER_OPEN, PAD_TIP_PAST_TCP as PAD_TIP  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402

NAME = "price-gun-conveyor"
INSTRUCTION = json.loads((_here / "task.json").read_text())["instruction"]
HZ = 20
MAX_STEPS = 960
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
    specs = (C.belt_object(), C.frame_object(), *C.rail_objects(), C.stripes_object(),
             *G.outline_objects(), G.box_object("box_a"), G.box_object("box_b"),
             G.rest_object(), G.gun_object(), G.trigger_object(), G.roller_object())
    objects += [C.disable_gravity(yscene.task_object(s)) for s in G.carrier_objects()]
    objects += [yscene.task_object(s) for s in specs]
    cfg = IsaacLabArenaEnvironment(name="yam_price_gun_conveyor", scene=Scene(objects), embodiment=embodiment,
                                   env_cfg_callback=yscene.env_cfg_callback(control_hz))
    return ArenaEnvBuilder(cfg, ArenaEnvBuilderCfg(num_envs=1, solve_relations=False, device=device)).make_registered()


class Dynamics:
    """The belt drive, and price labels from measured poses only (a disclosed stand-in for a label feed
    and adhesive).

    Steel carriers under the boxes are driven exactly along the belt (pose and velocity written before
    every physics step). The belt stands still while the scene settles and runs from the first control
    step. Its travel moves the marks drawn on it and tells the scorer where each box's place has got to.

    The gun holds N_LABELS labels. Its squeeze lever is a real sprung hinge; its measured angle drives
    the feed. Squeezed to 85 % of its travel, the gun feeds one label to the nose (drawn hanging in
    front of the roller); the lever must come back out to 30 % before it can feed again. While a label
    is at the nose, the roller coming within 2 mm of a surface (a box face, a carrier, the bench or the
    rest block) sticks the whole label there, centred under the roller with its long edge along the
    roller, and it stays on that body (one on a carrier rides away with it). Squeezing again with a
    label still at the nose pushes that label off as waste. There is no adhesion force, peel or wipe.
    """

    def __init__(self, env):
        import omni.usd
        self.base = env.unwrapped
        stage = omni.usd.get_context().get_stage()
        first = {**G.BOX_LABEL0, G.REST_NAME: G.REST_LABEL0}
        self.label_prims = {name: [None] * G.N_LABELS for name in first}
        self.nose = None
        moving = {C.STRIPES_NAME: None, **{C.outline_name(n): None for n in G.BOX_NAMES}}
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            if path.endswith(f"/gun/piece{G.PRESENT_PIECE}"):
                self.nose = prim
            for name in moving:
                if path.endswith(f"/{name}"):
                    moving[name] = prim
            for name, k0 in first.items():
                for k in range(G.N_LABELS):
                    if path.endswith(f"/{name}/piece{k0 + k}"):
                        self.label_prims[name][k] = prim
        if self.nose is None or any(p is None for prims in self.label_prims.values() for p in prims):
            raise RuntimeError("label pieces not found on the stage")
        if any(p is None for p in moving.values()):
            raise RuntimeError("belt marks not found on the stage")
        self.stripes = moving[C.STRIPES_NAME]
        self.outlines = {n: moving[C.outline_name(n)] for n in G.BOX_NAMES}
        centres = {n: b["centre"] for n, b in G.BOXES.items()}
        self.clock, self._drive = C.install_drive(env, centres, HZ)

    @staticmethod
    def _move(prim, pos):
        """Slide a drawn mark to ``pos`` (world)."""
        from pxr import Gf, UsdGeom
        ops = {op.GetOpType(): op.GetAttr() for op in UsdGeom.Xformable(prim).GetOrderedXformOps()}
        attr = ops[UsdGeom.XformOp.TypeTranslate]
        vec = Gf.Vec3f if str(attr.GetTypeName()) == "float3" else Gf.Vec3d
        attr.Set(vec(*map(float, pos)))

    def _draw_belt(self, travel):
        """The marks printed on the belt, ``travel`` metres on; an outline goes under at the end stop."""
        from pxr import UsdGeom
        self._move(self.stripes, C.stripes_pos(travel))
        for name, prim in self.outlines.items():
            b = G.BOXES[name]
            pos = C.outline_pos(b["centre"], travel)
            self._move(prim, pos)
            show = pos[1] + b["size"][1] / 2 + C.MARK_GAP < C.Y1
            (UsdGeom.Imageable(prim).MakeVisible if show else UsdGeom.Imageable(prim).MakeInvisible)()

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
        self.travel = 0.0            # how far the belt has run
        self.clock["t"] = 0.0
        self.clock["running"] = False
        self._drive()
        self._draw_belt(0.0)

    def before_step(self, step, action):
        # The belt clock starts with the first control step (settling leaves carriers still).
        if step >= 1:
            self.clock["running"] = True

    def _stick(self, surface, point, normal, axis):
        from pxr import UsdGeom
        rot = G.label_frame(normal, axis)
        k = len(self.labels)
        holder = surface if surface in G.BOX_NAMES else G.REST_NAME   # bench and belt labels are drawn by the rest block
        local = np.asarray(point, float) + 0.0005 * np.asarray(normal, float)
        if holder == G.REST_NAME:
            local = local - G.rest_pos()
        self._place(self.label_prims[holder][k], local, rot)
        self.labels.append({"on": surface, "point": np.asarray(point, float), "normal": np.asarray(normal, float),
                            "rot": rot, "prim": self.label_prims[holder][k], "local": local, "travel": self.travel})
        self.at_nose = False
        UsdGeom.Imageable(self.nose).MakeInvisible()

    def after_step(self, step, objects, root_poses, robot):
        from pxr import UsdGeom
        self.travel = C.travel_at(step / HZ)
        self._draw_belt(self.travel)

        for lab in self.labels:                                # a label stuck on the belt rides away with it
            if lab["on"] == "belt":
                ridden = self.travel - lab["travel"]
                if lab["point"][1] + ridden < C.Y1 - G.LABEL[1]:
                    self._place(lab["prim"], lab["local"] + np.array([0.0, ridden, 0.0]), lab["rot"])
                else:
                    UsdGeom.Imageable(lab["prim"]).MakeInvisible()
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
        out = {"belt_travel_mm": round(1000 * self.travel, 3),
               "trigger": round(frac, 3), "fed": self.fed, "labels_left": G.N_LABELS - self.fed,
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
CLEAR = 0.020                                             # roller above the leading box's top while it is carried or waits
HOVER = 0.015                                             # roller above the flat box's top before it comes down
NEAR = 0.006                                              # it pauses this far above the panel, to come down square
PRESS = 0.003                                             # and is then sent this far into it
T_B, T_A = 19.5, 26.0                                     # when the roller is down on each box's panel (s)
# The arms hold position by joint stiffness, so the gun's weight lowers its nose a little, by an amount
# that depends on the reach. Measured on the settled arm, and added to the heights the plan asks for.
SAG = {"box_a": 0.0070, "box_b": 0.0075, "rest": 0.0040}
# The gun also hangs a degree or two off square in the jaws, and the loaded arm stops a few millimetres
# short: where the roller lands relative to where the plan sends it (x, y), measured and aimed off.
AIM = {"box_a": (0.0057, 0.0030), "box_b": (0.0057, 0.0030)}


def _hand(seconds, gun_xyz, finger, label, shut=True):
    """TCP holding the gun level with its body centre at ``gun_xyz``, nose toward +y."""
    f = G.GRIP_F_SHUT if shut else G.GRIP_F_OPEN
    tcp = np.asarray(gun_xyz, float) + G.L(f, 0.0, TCP_U)
    return Waypoint(seconds, tuple(map(float, tcp)), finger, label, DOWN)


def _gun_over(point, height):
    """Gun body centre that puts the roller's bottom ``height`` above ``point``."""
    return np.asarray(point, float) - G.L(G.ROLLER_AT[0], 0.0, G.ROLLER_AT[1] - G.ROLLER[0]) + np.array([0.0, 0.0, height])


def _panel(name, t):
    """Where the plan sends the roller for the box's price panel at time ``t``: the belt has carried it on."""
    return G.panel_world(name, C.travel_at(t)) + np.array([*AIM[name], 0.0])


def demo():
    up = np.array([0.0, 0.0, 1.0])
    g0 = G.gun_pos()
    # squeezing shifts the gun forward in the hand by half the lever's travel
    g1 = g0 + G.L(G.GRIP_F_OPEN - G.GRIP_F_SHUT, 0.0, 0.0)
    sa, sb, sr = SAG["box_a"], SAG["box_b"], SAG["rest"]
    top_b = G.panel_world("box_b")[2]
    carry = top_b + CLEAR + sb                              # roller's bottom while the gun is carried over the belt
    lift = carry - (g1[2] + G.ROLLER_AT[1] - G.ROLLER[0])   # that, as a lift of the gun off its rest
    plan = []                                              # (seconds, gun body centre at the end, finger, label, shut)

    def add(seconds, gun_xyz, finger, label, shut=True):
        plan.append((seconds, gun_xyz, finger, label, shut))
        return sum(p[0] for p in plan)

    def until(t, gun_at, finger, label):
        """A move that ends at time ``t``, at ``gun_at(t)``."""
        now = sum(p[0] for p in plan)
        if t - now < 0.25:
            raise ValueError(f"no time for '{label}': {t - now:.2f} s")
        return add(t - now, gun_at(t), finger, label)

    def over(name, height):
        return lambda t: _gun_over(_panel(name, t), height)

    add(1.5, None, FINGER_OPEN, "look")
    add(2.0, g0 + 0.06 * up, F_WIDE, "over the price gun's handle", shut=False)
    add(1.2, g0, F_WIDE, "down round handle and lever", shut=False)
    add(0.8, g0, F_SQUEEZE, "squeeze: take it, and a label feeds to the nose", shut=False)
    add(1.5, g1 + lift * up, F_SQUEEZE, "lift it clear of the tall box's top")
    # the tall box leads: meet its panel from above and come down with it
    until(T_B - 1.7, over("box_b", CLEAR + sb), F_SQUEEZE, "over the belt, above the tall box's price panel")
    until(T_B - 0.8, over("box_b", NEAR + sb), F_SQUEEZE, "nose down to just over the panel, moving with it")
    until(T_B, over("box_b", sb - PRESS), F_SQUEEZE, "touch the roller on the panel: the label sticks")
    until(T_B + 0.4, over("box_b", sb - PRESS), F_SQUEEZE, "press it on, moving with the box")
    t = until(T_B + 1.0, over("box_b", CLEAR + sb), F_SQUEEZE, "lift off")
    wait = _gun_over(_panel("box_b", t), CLEAR + sb)
    add(0.7, wait, F_HOLD, "let the lever out as the tall box rides on")
    add(0.4, wait, F_HOLD, "lever out")
    add(0.7, wait, F_SQUEEZE, "squeeze again: the next label feeds")
    # the flat box follows 30 mm behind it
    until(T_A - 1.3, over("box_a", HOVER + sa), F_SQUEEZE, "down behind the tall box, over the flat box's price panel")
    until(T_A - 0.8, over("box_a", NEAR + sa), F_SQUEEZE, "nose down to just over the panel, moving with it")
    until(T_A, over("box_a", sa - PRESS), F_SQUEEZE, "touch the roller on the panel: the label sticks")
    until(T_A + 0.4, over("box_a", sa - PRESS), F_SQUEEZE, "press it on, moving with the box")
    until(T_A + 1.2, over("box_a", 2 * HOVER + sa), F_SQUEEZE, "lift off")
    add(2.5, g1 + (0.05 + sr) * up, F_SQUEEZE, "back over the rest")
    add(1.2, g1 + (0.005 + sr) * up, F_SQUEEZE, "down to just over the rest")
    add(0.4, g1 + (0.005 + sr) * up, F_SQUEEZE, "steady")
    add(0.6, g1 + (sr - 0.001) * up, F_SQUEEZE, "lay it on the rest")
    add(0.8, g0 + (sr - 0.001) * up, F_WIDE, "let go", shut=False)
    add(0.8, g0 + 0.06 * up, F_WIDE, "lift off", shut=False)
    add(1.5, None, FINGER_OPEN, "stand clear")

    right = [Waypoint(s, REST["right"], f, label, DOWN) if xyz is None else _hand(s, xyz, f, label, shut)
             for s, xyz, f, label, shut in plan]
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
