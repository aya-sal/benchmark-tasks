"""Sort the supplies and tools into the open toolbox, then fold and pack the glasses."""

import itertools
import math
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from propkit import install_trial_reset, prop, rot_matrix
from yam_arena.task_file import TABLE_TOP

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0
INSTRUCTION = ("Put the grease bottle and spray can upright on the left half of the open toolbox with the "
               "sealant tube. Put the hand tools and duct tape on the right half. Fold both temples of "
               "the safety glasses, then put them anywhere inside the toolbox last.")

SIZE_INCREASE = 1.05
BOX_SCALE = tuple(axis * SIZE_INCREASE for axis in (1.42, 1.15, .43))
BOX_XY = (.380, 0.0)
BOX_YAW = math.pi
BOX_BOUNDS = ((-.26802, -.26531, .00005), (.14695, .26531, .47299))
BOX = prop("box", HERE / "assets/prepped/toolbox_pack_open.usda", BOX_BOUNDS, BOX_XY,
           table_top=TOP + BOX_BOUNDS[0][2] * BOX_SCALE[2], yaw=BOX_YAW,
           scale=BOX_SCALE, fixed=True)
BOX_Z = TOP + .001
BOX_FLOOR = BOX_Z + BOX_SCALE[2] * .01565
BOX_RIM = BOX_Z + BOX_SCALE[2] * .1742
# Toolbox asset frame. The base is a continuous cavity and y=0 is the scoring boundary.
INTERIOR = ((-.112, -.229), (.112, .229))
LIMITS = {"tools": ((-.112, 0), (.112, .229)),
          "supplies": ((-.112, -.229), (.112, 0)),
          "anywhere": INTERIOR}
TEMPLE_LEFT = "joint_templeLeft_01_joint_00"
TEMPLE_RIGHT = "joint_templeRight_01_joint_00"
TEMPLE_JOINTS = (TEMPLE_LEFT, TEMPLE_RIGHT)
HINGES = {TEMPLE_LEFT: (0.019738588, 0.081444114, 0.054177),
          TEMPLE_RIGHT: (0.019744325, -0.08144133, 0.05417777)}
FOLD_MIN = math.radians(75)

# USD, local visual bounds, station origin XY, yaw in degrees, uniform scale, mass in kg.
ITEMS = {
    "lubricant": ("prepped/lubricant_flat.usda", ((-.0413,-.0413,.0004),(.0413,.0413,.2466)), (.275,.425), 0, .62, .30),
    "spray": ("prepped/spray_flat.usda", ((-.031,-.031,0),(.031,.031,.1864)), (.355,.465), 0, .78, .28),
    "sealant": ("prepped/sealant.usda", ((-.0794,-.0179,.0001),(.0351,.0182,.0233)), (.365,.355), 90, .65, .07),
    "hammer": ("hammer/simready_usd/hammer.usda", ((-.0651,-.1651,0),(.0651,.1651,.0292)), (.23,-.42), 90, .55, .24),
    "pliers": ("prepped/pliers.usda", ((-.1108,-.0345,0),(.0811,.0342,.0259)), (.20,-.34), 0, .65, .10),
    "wrench": ("prepped/wrench.usda", ((-.0922,-.0302,.0003),(.1174,.0334,.0142)), (.40,-.37), 0, .55, .11),
    "screwdriver": ("screwdriver/simready_usd/screwdriver.usda", ((-.1029,-.0166,-.0035),(.1027,.0166,.0319)),
                    (.406,-.46), 0, .55, .07),
    "tape": ("prepped/tape.usda", ((-.0681,-.0681,.0002),(.0697,.0697,.0459)), (.098,-.24), 0, .55, .07),
    "glasses": ("prepped/glasses_articulated.usda", ((-.0825,-.0845,.0001),(.0845,.0845,.0596)),
                (.064,0), 0, .85, None),
}
SPEC = {name: prop(name, HERE / "assets" / path, bounds, xy, table_top=TOP,
                   yaw=math.radians(yaw), scale=scale * SIZE_INCREASE, mass=mass,
                   joints=TEMPLE_JOINTS if name == "glasses" else ())
        for name, (path, bounds, xy, yaw, scale, mass) in ITEMS.items()}
OBJECTS = [BOX, *SPEC.values()]
SUPPLIES = ("sealant", "lubricant", "spray")
TOOLS = ("hammer", "pliers", "wrench", "screwdriver", "tape")
ORDER = (*SUPPLIES, *TOOLS, "glasses")

# Station XY for the item bounding-box centre, and target item yaw in degrees.
SLOTS = {
    "lubricant": (.334, .105, 0), "spray": (.344, .229, 0), "sealant": (.260, .148, 0),
    "hammer": (.396, -.105, 90), "pliers": (.267, -.088, 90), "wrench": (.354, -.038, 0),
    "screwdriver": (.229, -.144, 90), "tape": (.439, -.210, 0), "glasses": (.302, .025, 90),
}

FRAME_BOUNDS = ((.0167,-.0845,.0145),(.0845,.0845,.0596))
TEMPLE_BOUNDS = {
    TEMPLE_LEFT: ((-.0825,.0738,.0001),(.0210,.0845,.0583)),
    TEMPLE_RIGHT: ((-.0825,-.0845,.0001),(.0210,-.0738,.0583)),
}


def rz(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]])


def corners(bounds):
    return np.array(list(itertools.product(*zip(*bounds))))


def local_points(name, joints):
    spec = SPEC[name]
    if name != "glasses":
        return corners((spec.bmin, spec.bmax)) * spec.scale
    parts = [corners(FRAME_BOUNDS)]
    for joint, bounds in TEMPLE_BOUNDS.items():
        pivot = np.asarray(HINGES[joint])
        parts.append((corners(bounds) - pivot) @ rz(joints[joint]).T + pivot)
    return np.concatenate(parts) * spec.scale


def world_points(name, state):
    return local_points(name, state.joints) @ rot_matrix(state.quat_xyzw).T + state.origin


def folded(state):
    return state.joints[TEMPLE_LEFT] >= FOLD_MIN and state.joints[TEMPLE_RIGHT] <= -FOLD_MIN


def inside(name, state, side):
    points = world_points(name, state)
    box_points = (points - (*BOX_XY, BOX_Z)) @ rz(BOX_YAW) / np.asarray(BOX_SCALE)
    low, high = map(np.asarray, LIMITS[side])
    bottom = points[:, 2].min()
    return (bool(np.all(box_points[:, :2] >= low - .002) and np.all(box_points[:, :2] <= high + .002))
            and BOX_FLOOR - .005 <= bottom <= BOX_FLOOR + .030)


def item_ok(name, state):
    side = "anywhere" if name == "glasses" else "supplies" if name in SUPPLIES else "tools"
    if not inside(name, state, side):
        return False
    if name in ("lubricant", "spray"):
        return rot_matrix(state.quat_xyzw)[2,2] >= math.cos(math.radians(15))
    return name != "glasses" or folded(state)


def settled(state):
    return np.linalg.norm(state.lin_vel) < .035 and np.linalg.norm(state.ang_vel) < .4


class PackingTrial:
    def __init__(self):
        self.glasses_invalid = False
        self.glasses_entered = False

    def evaluate(self, states):
        valid = {name: item_ok(name, states[name]) for name in ORDER}
        if not self.glasses_entered and inside("glasses", states["glasses"], "anywhere") and settled(states["glasses"]):
            self.glasses_entered = True
            self.glasses_invalid = (not folded(states["glasses"])
                                    or not all(valid[name] and settled(states[name]) for name in ORDER[:-1]))
        if self.glasses_invalid:
            valid["glasses"] = False
        return valid


_trial = PackingTrial()


def reset_trial():
    global _trial
    _trial = PackingTrial()


def score(states):
    return sum(_trial.evaluate(states).values()) / len(ORDER)


def success(states):
    return all(_trial.evaluate(states).values()) and all(settled(states[name]) for name in ORDER)


def demo_script(ctx):
    from packing_motion import pack
    yield from pack(ctx, ctx.emb.task.module)


install_trial_reset()
