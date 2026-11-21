"""Wipe eight fingerprints off an LG 27GX790A without the foot leaving the desk."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

import scoring
from lgasset import Monitor, hide_marks, on_reset, on_step
from yam_arena.task_file import Box

INSTRUCTION = (
    "Wipe every fingerprint off the monitor glass with the cloth. "
    "Leave the monitor standing on the desk, and let go."
)
SILVER = (0.55, 0.57, 0.60)
CLOTH_COLOR = (0.78, 0.80, 0.82)

MONITOR = Monitor(
    "monitor",
    pos=scoring.ORIGIN,
    size=(0.250, 0.606, 0.581),
    mass=9.3,
    color=(0.15, 0.15, 0.16),
    quat_xyzw=(0.0, 0.0, 1.0, 0.0),
    marks=scoring.MARKS,
)
CLOTH = Box("cloth", pos=scoring.CLOTH_POS, size=scoring.CLOTH_SIZE, mass=0.045, color=CLOTH_COLOR)
PEDESTALS = [
    Box(name=f"{side}_robot_pedestal", pos=(0.0, y, scoring.TOP + 0.175), size=(0.20, 0.20, 0.35), mass=None, color=SILVER)
    for side, y in (("left", 0.305), ("right", -0.305))
]
OBJECTS = [*PEDESTALS, MONITOR, CLOTH]
_STATE = scoring.new_state()


def _rot(xyzw):
    x, y, z, w = (float(v) for v in xyzw)
    return (
        (1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
        (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
        (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)),
    )


def _np(value):
    value = getattr(value, "torch", value)
    return value.detach().cpu().numpy()


def _link(asset, link):
    names = list(asset.body_names)
    for i, name in enumerate(names):
        leaf = name.rsplit("/", 1)[-1]
        if leaf == link:
            return i
    raise KeyError(f"{link} not in {names}")


def _observe(environment, cloth):
    from yam_arena.constants import FINGER_JOINTS
    from yam_arena.runtime import tcp_world

    base = environment.unwrapped
    origin = _np(base.scene.env_origins)[0]
    scene = base.scene
    asset = scene.articulations["monitor"] if "monitor" in getattr(scene, "articulations", {}) else scene["monitor"]
    poses = _np(asset.data.body_link_pose_w)[0]
    links = {}
    for link in ("base", "monitor"):
        pose = poses[_link(asset, link)]
        links[link] = (tuple(float(v) for v in pose[:3] - origin), _rot(pose[3:7]))
    tips, held = [], False
    cloth_pos = tuple(float(v) for v in cloth.pos)
    for side in ("left", "right"):
        robot = scene[f"{side}_robot"]
        tcp = tcp_world(robot) - origin
        tips.append(tuple(float(v) for v in tcp))
        ids = robot.find_joints(list(FINGER_JOINTS), preserve_order=True)[0]
        opening = float(abs(_np(robot.data.joint_pos)[0, ids]).sum())
        reach = sum((tcp[i] - cloth_pos[i]) ** 2 for i in range(3)) ** 0.5
        held = held or (opening < scoring.HOLD_OPENING and reach < scoring.HOLD_REACH)
    return {
        "base_pos": links["base"][0],
        "base_R": links["base"][1],
        "panel_pos": links["monitor"][0],
        "panel_R": links["monitor"][1],
        "cloth_pos": cloth_pos,
        "cloth_R": _rot(cloth.quat_xyzw),
        "tips": tips,
        "held": held,
    }


@on_reset
def _reset(environment=None):
    global _STATE
    _STATE = scoring.new_state()
    hide_marks(_STATE["clean"])


@on_step
def _tick(environment):
    _tick.env = environment
    _tick.n += 1


_tick.env = None
_tick.n = 0


def _update(objects):
    """Apply this control step once. ``score`` and ``success`` both call in."""
    if _tick.env is None or _STATE.get("seen") == _tick.n:
        return
    _STATE["seen"] = _tick.n
    scoring.step(_STATE, _observe(_tick.env, objects["cloth"]))
    hide_marks(_STATE["clean"])


def score(objects):
    _update(objects)
    return _STATE["score"]


def success(objects):
    _update(objects)
    return _STATE["success"]
