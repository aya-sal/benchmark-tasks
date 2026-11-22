"""Wipe eight fingerprints off an LG 27GX790A without the foot leaving the desk."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

import scoring
from lgasset import Cloth, Monitor, hide_marks, on_reset, on_step

INSTRUCTION = (
    "Wipe every fingerprint off the monitor glass with the cloth. "
    "Leave the monitor standing on the desk, and let go."
)
CLOTH_COLOR = (0.55, 0.57, 0.60)#(0.78, 0.80, 0.82)

MONITOR = Monitor(
    "monitor",
    pos=scoring.ORIGIN,
    size=(0.250, 0.606, 0.581),
    mass=9.3,
    color=(0.15, 0.15, 0.16),
    quat_xyzw=(0.0, 0.0, 1.0, 0.0),
    marks=scoring.MARKS,
)
CLOTH = Cloth("cloth", pos=scoring.CLOTH_POS, size=scoring.CLOTH_SIZE, mass=0.060, color=CLOTH_COLOR)
OBJECTS = [MONITOR, CLOTH]
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


def _sheet_rotation(nodes):
    """Axes of a soft sheet. Column 2 is the thin direction, the face normal."""
    import numpy as np

    centered = nodes - nodes.mean(axis=0)
    _, _, axes = np.linalg.svd(centered, full_matrices=False)
    normal = axes[2]
    if normal[2] < 0.0:
        normal = -normal
    helper = np.array([1.0, 0.0, 0.0]) if abs(float(normal[0])) < 0.9 else np.array([0.0, 1.0, 0.0])
    x_axis = np.cross(helper, normal)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(normal, x_axis)
    rotation = np.column_stack([x_axis, y_axis, normal])
    return tuple(tuple(float(v) for v in row) for row in rotation)


def _observe(environment, cloth):
    import numpy as np

    from yam_arena.constants import FINGER_JOINTS
    from yam_arena.runtime import pad_tips_world, tcp_world

    base = environment.unwrapped
    origin = _np(base.scene.env_origins)[0]
    scene = base.scene
    asset = scene.articulations["monitor"] if "monitor" in getattr(scene, "articulations", {}) else scene["monitor"]
    poses = _np(asset.data.body_link_pose_w)[0]
    links = {}
    for link in ("base", "monitor"):
        pose = poses[_link(asset, link)]
        links[link] = (tuple(float(v) for v in pose[:3] - origin), _rot(pose[3:7]))
    soft = getattr(scene, "deformable_objects", {}).get("cloth")
    if soft is not None:
        nodes = _np(soft.data.nodal_pos_w)[0] - origin
        cloth_pos = tuple(float(v) for v in nodes.mean(axis=0))
        cloth_R = _sheet_rotation(nodes)
    else:
        nodes = None
        cloth_pos = tuple(float(v) for v in cloth.pos)
        cloth_R = _rot(cloth.quat_xyzw)
    tips, fingers, held, extras = [], [], False, []
    for side in ("left", "right"):
        robot = scene[f"{side}_robot"]
        tcp = tcp_world(robot) - origin
        tips.append(tuple(float(v) for v in tcp))
        fingers.append(tuple(float(v) for v in (pad_tips_world(robot) - origin)))
        ids = robot.find_joints(list(FINGER_JOINTS), preserve_order=True)[0]
        opening = float(abs(_np(robot.data.joint_pos)[0, ids]).sum())
        if nodes is None:
            reach = sum((tcp[i] - cloth_pos[i]) ** 2 for i in range(3)) ** 0.5
        else:
            reach = float(np.linalg.norm(nodes - tcp, axis=1).min())
        closed = opening < scoring.HOLD_OPENING
        held = held or (closed and reach < scoring.HOLD_REACH)
        if closed:
            extras.append(pad_tips_world(robot) - origin)
    points = nodes if nodes is not None else np.zeros((0, 3))
    if extras:
        points = np.vstack([points, *extras]) if len(points) else np.vstack(extras)
    press = _press(points, links["monitor"][0], links["monitor"][1])
    return {
        "base_pos": links["base"][0],
        "base_R": links["base"][1],
        "panel_pos": links["monitor"][0],
        "panel_R": links["monitor"][1],
        "cloth_pos": cloth_pos,
        "cloth_R": cloth_R,
        "tips": tips,
        "fingers": fingers,
        "cloth_pts": () if nodes is None else tuple(tuple(float(v) for v in point) for point in nodes),
        "held": held,
        "press": press,
    }


def _press(points, panel_pos, panel_R):
    """Spring force, in newtons, on each stain from cloth nodes and a closed gripper's fingertip.

    Nothing in the scene reports the cloth's contact force. A point within 4 cm of a stain and
    3 cm of the glass pushes with 250 N/m times how far it is inside that gap. One closed
    fingertip on a stain is enough to pass the 1.5 N line.
    """
    import numpy as np

    points = np.atleast_2d(np.asarray(points, dtype=float))
    if points.size == 0:
        return [0.0] * len(scoring.MARKS)
    rotation = np.asarray(panel_R, dtype=float)
    normal = rotation[:, 0]
    origin = np.asarray(panel_pos, dtype=float)
    presses = []
    for local in scoring.MARKS:
        mark = origin + rotation @ np.asarray(local, dtype=float)
        delta = points - mark
        along = delta @ normal
        planar = np.linalg.norm(delta - along[:, None] * normal, axis=1)
        close = (planar < 0.04) & (along < 0.030) & (along > -0.008)
        depth = np.clip(0.030 - along[close], 0.0, 0.038)
        presses.append(float(250.0 * depth.sum()) if depth.size else 0.0)
    return presses


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
_was_held = False


def _update(objects):
    """Apply this control step once. ``score`` and ``success`` both call in."""
    global _was_held
    if _tick.env is None or _STATE.get("seen") == _tick.n:
        return
    _STATE["seen"] = _tick.n
    obs = _observe(_tick.env, objects["cloth"])
    if obs["held"]:
        _was_held = True
    elif _was_held:
        cx, cy, cz = obs["cloth_pos"]
        print(f"[step {_tick.n}] CLOTH DROPPED at ({cx:.3f}, {cy:.3f}, {cz:.3f})")
        _was_held = False
    scoring.step(_STATE, obs)
    hide_marks(_STATE["clean"])


def score(objects):
    _update(objects)
    return _STATE["score"]


def success(objects):
    _update(objects)
    return _STATE["success"]


def _hold_target(self, t):
    """Like ``ArmPlan.target``, but a waypoint may carry the ready rotation in ``rot``."""
    import numpy as np

    from yam_arena import ik
    from yam_arena.demo import grip_rotation

    prev_pos, prev_rot, prev_finger = self.start_pos, self.start_rot, self.finger_start
    for waypoint in self.waypoints:
        goal = np.asarray(waypoint.pos, dtype=float)
        rot = getattr(waypoint, "rot", None)
        if rot is None:
            rot = grip_rotation(waypoint.yaw)
        if t < waypoint.duration:
            s = ik.smoothstep(t / waypoint.duration) if waypoint.duration else 1.0
            return (
                prev_pos + s * (goal - prev_pos),
                ik.slerp_matrix(prev_rot, rot, s),
                prev_finger + s * (waypoint.finger - prev_finger),
                waypoint.label,
            )
        t -= waypoint.duration
        prev_pos, prev_rot, prev_finger = goal, rot, waypoint.finger
    return prev_pos, prev_rot, prev_finger, "done"


def _wipe_rotation():
    """Cloth face toward the glass: the down-grip pitched up until the sheet is nearly vertical."""
    import numpy as np

    from yam_arena.demo import grip_rotation
    from yam_arena.urdf import axis_angle_matrix

    down = grip_rotation(0.0)
    return axis_angle_matrix(np.array([0.0, 1.0, 0.0]), np.radians(-80.0)) @ down


def _rub_rotation(world_y):
    """Wrist for a rub. The far-left column is only reachable with the hand turned."""
    import numpy as np

    from yam_arena.demo import grip_rotation
    from yam_arena.urdf import axis_angle_matrix

    down = grip_rotation(0.0)
    pitch = -80.0 if world_y < 0.2 else -90.0
    pitched = axis_angle_matrix(np.array([0.0, 1.0, 0.0]), np.radians(pitch)) @ down
    if world_y < 0.2:
        return pitched
    return axis_angle_matrix(np.array([0.0, 0.0, 1.0]), np.radians(45.0)) @ pitched


def _fingertip_tcp(world_y, world_z, rotation):
    """TCP that puts the fingertip on this point of the glass."""
    import numpy as np

    from yam_arena.constants import PAD_TIP_PAST_TCP

    stain_x = scoring.ORIGIN[0] - 0.0545
    point = np.array([stain_x, world_y, world_z], dtype=float)
    return point - np.asarray(rotation)[:, 2] * PAD_TIP_PAST_TCP


def _stain_rubs_right():
    """Scrub the 6 stains reachable by the right arm (world y ≤ +0.08)."""
    import numpy as np

    points = []
    for local in scoring.MARKS:
        world = scoring.yaw180(local, scoring.HEIGHT)
        if world[1] <= 0.09:
            points.append((world[1], world[2]))
    points.sort()
    moves = []
    span = 0.02
    for index, (y, z) in enumerate(points):
        rot = _rub_rotation(y)
        if index and abs(points[index - 1][0] - y) > 0.01:
            py, pz = points[index - 1]
            moves.append((0.5, np.array([0.40, py, pz]), _rub_rotation(py), "back off"))
        center = _fingertip_tcp(y, z, rot)
        moves.append((1.2, center, rot, "onto the stain"))
        moves.append((0.5, center, rot, "settle on the stain"))
        moves.append((0.8, _fingertip_tcp(y - span, z, rot), rot, "rub across"))
        moves.append((1.0, _fingertip_tcp(y + span, z, rot), rot, "rub across the stain"))
        moves.append((0.6, center, rot, "back on the stain"))
    return moves


def _stain_rubs_left():
    """Scrub the 2 far-left stains (world y ≈ +0.24) with the left arm."""
    import numpy as np

    points = []
    for local in scoring.MARKS:
        world = scoring.yaw180(local, scoring.HEIGHT)
        if world[1] > 0.09:
            points.append((world[1], world[2]))
    points.sort()
    rot = _wipe_rotation()
    moves = []
    span = 0.02
    for index, (y, z) in enumerate(points):
        center = _fingertip_tcp(y, z, rot)
        moves.append((1.2, center, rot, "onto the stain"))
        moves.append((0.5, center, rot, "settle on the stain"))
        moves.append((0.8, _fingertip_tcp(y - span, z, rot), rot, "rub across"))
        moves.append((1.0, _fingertip_tcp(y + span, z, rot), rot, "rub across the stain"))
        moves.append((0.6, center, rot, "back on the stain"))
    return moves


def _waypoints(moves):
    from yam_arena.demo import Waypoint

    waypoints = []
    for duration, pos, rot, finger, label, yaw in moves:
        waypoint = Waypoint(duration, tuple(float(v) for v in pos), finger, label, yaw)
        if rot is not None:
            waypoint.rot = rot
        waypoints.append(waypoint)
    return waypoints


def demo():
    """Two-arm wipe: left crinkles, right wipes 6 stains and drops cloth, left picks up and wipes the last 2."""
    import numpy as np

    from yam_arena import ik, station
    from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN, PAD_TIP_PAST_TCP, READY
    from yam_arena.demo import ArmPlan, grip_rotation

    ArmPlan.target = _hold_target
    x, y, _ = scoring.CLOTH_POS
    z = scoring.TOP + PAD_TIP_PAST_TCP
    half = scoring.CLOTH_SIZE[1] / 2

    left_ready_pos, left_ready_rot = ik.tcp_pose(np.array(READY, float), np.asarray(station.BASES["left"], float))
    right_ready_pos, right_ready_rot = ik.tcp_pose(np.array(READY, float), np.asarray(station.BASES["right"], float))

    # ── Phase 1: Left arm crinkles the cloth ──────────────────────────
    edge_y = y + half + 0.02
    stop_y = y
    left_phase1 = [
        (1.6, (x, edge_y, z + 0.12), None, FINGER_CLOSED, "above the left edge", 0.0),
        (1.0, (x, edge_y, z), None, FINGER_CLOSED, "down on the edge", 0.0),
        (2.2, (x, stop_y, z), None, FINGER_CLOSED, "push the edge right", 0.0),
        (0.8, (x, stop_y, z + 0.14), None, FINGER_CLOSED, "lift clear", 0.0),
        (0.8, (x, edge_y + 0.04, z + 0.14), None, FINGER_CLOSED, "back off", 0.0),
        (0.8, tuple(float(v) for v in left_ready_pos), left_ready_rot, FINGER_OPEN, "ready", 0.0),
    ]
    phase1_t = sum(m[0] for m in left_phase1)

    # ── Phase 2: Right arm picks up cloth and wipes 6 stains ─────────
    bunch = (x, stop_y - 0.03, z)
    face = _wipe_rotation()
    right_all = [
        (phase1_t, tuple(float(v) for v in right_ready_pos), right_ready_rot, FINGER_OPEN, "wait for crinkle", 0.0),
        (1.6, (bunch[0], bunch[1], bunch[2] + 0.12), None, FINGER_OPEN, "above the bunch", 0.0),
        (1.2, bunch, None, FINGER_OPEN, "down onto the bunch", 0.0),
        (0.8, bunch, None, FINGER_OPEN, "open on the bunch", 0.0),
        (0.8, bunch, None, FINGER_CLOSED, "close", 0.0),
        (0.6, bunch, None, FINGER_CLOSED, "hold the pinch", 0.0),
        (1.0, (bunch[0], bunch[1], bunch[2] + 0.18), None, FINGER_CLOSED, "lift the cloth", 0.0),
        (1.6, (0.36, -0.10, 1.00), face, FINGER_CLOSED, "turn toward the glass", 0.0),
    ]
    for duration, pos, rot, label in _stain_rubs_right():
        right_all.append((duration, tuple(float(v) for v in pos), rot, FINGER_CLOSED, label, 0.0))

    # ── Phase 3: Right arm drops cloth and goes to ready ──────────────
    right_all += [
        (0.8, (0.36, 0.00, 1.00), face, FINGER_CLOSED, "back from glass", 0.0),
        (0.4, (0.36, 0.00, 1.00), face, FINGER_OPEN, "release cloth", 0.0),
        (1.0, tuple(float(v) for v in right_ready_pos), right_ready_rot, FINGER_OPEN, "right ready", 0.0),
    ]
    right_t = sum(m[0] for m in right_all)

    # ── Phase 4: Left arm picks up cloth at drop zone and wipes 2 stains ─
    left_face = _wipe_rotation()
    drop = (0.36, 0.00, z)  # cloth falls near where the right arm released it
    left_phase2 = [
        (right_t - phase1_t + 2.0, tuple(float(v) for v in left_ready_pos), left_ready_rot, FINGER_OPEN, "wait for right arm", 0.0),
        (1.4, (drop[0], drop[1], drop[2] + 0.12), None, FINGER_OPEN, "above drop zone", 0.0),
        (1.0, drop, None, FINGER_OPEN, "down onto cloth", 0.0),
        (0.6, drop, None, FINGER_OPEN, "settle", 0.0),
        (0.6, drop, None, FINGER_CLOSED, "grasp cloth", 0.0),
        (0.6, drop, None, FINGER_CLOSED, "hold", 0.0),
        (1.0, (drop[0], drop[1], drop[2] + 0.18), None, FINGER_CLOSED, "lift cloth", 0.0),
        (1.4, (0.36, 0.20, 1.00), left_face, FINGER_CLOSED, "turn toward glass", 0.0),
    ]
    for duration, pos, rot, label in _stain_rubs_left():
        left_phase2.append((duration, tuple(float(v) for v in pos), rot, FINGER_CLOSED, label, 0.0))
    left_phase2.append((1.0, tuple(float(v) for v in left_ready_pos), left_ready_rot, FINGER_OPEN, "left done", 0.0))

    return {"left": _waypoints(left_phase1 + left_phase2), "right": _waypoints(right_all)}
