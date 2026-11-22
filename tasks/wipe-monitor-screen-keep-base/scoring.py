"""Score for wiping the LG 27GX790A glass without the foot leaving the desk.

Pure geometry: no simulator imports. Poses are metres, rotations are 3x3 row-major
tuples (columns are the body's local axes in the world).
"""

from __future__ import annotations

import math

TOP = 0.7475
DESK_X = (-0.19, 0.91)   # station table, 1.10 m deep, centred at x = 0.36
DESK_Y = (-0.65, 0.65)   # 1.30 m wide, centred at y = 0
ORIGIN = (0.56, 0.0, TOP + 0.001)  # bottom centre of the foot, 20 cm forward of the desk centre
YAW = math.pi
HEIGHT = -0.110          # height slide at its low stop (authored travel 0.11288 m)
# Glass is the flat face at local x = 0.0531, normal +x, y ±0.2956, z 0.2364..0.5732.
MARKS = tuple((0.0545, y, z) for z in (0.32, 0.50) for y in (-0.24, -0.08, 0.08, 0.24))
FOOT = tuple((x, y, 0.001) for x in (-0.120, 0.120) for y in (-0.150, 0.150))
# Link-frame boxes the hands must leave. Base includes the column; panel is the head.
BASE_BOX = ((-0.125, -0.155, 0.0), (0.125, 0.155, 0.447))
PANEL_BOX = ((-0.017, -0.303, 0.219), (0.054, 0.303, 0.581))
CLOTH_SIZE = (0.20, 0.20, 0.010)  # soft 200 mm sheet, 10 mm thick, lying on the desk
CLOTH_POS = (0.14, 0.0, TOP + 0.006)  # mid-surface, so the underside rests on the desk
SLIDE_M = 0.025
PRESS_N = 1.5            # kept for the spring estimate; cleaning uses contact on the glass
STAIN_DIAMETER = 0.08    # metres. The arm and the cloth must meet inside this circle
HOLD_STEPS = 40
CLEAR_M = 0.06
HOLD_OPENING = 0.045     # finger-gap sum, metres; full open is 0.094
HOLD_REACH = 0.07
GAP = 0.025              # how far a point may sit off the glass and still count as contact
ALIGN = math.cos(math.radians(25))
MONITOR_TILT_MAX = math.cos(math.radians(45))  # monitor "fallen" if tilted past 45°
DRIFT_M = 0.003
DRIFT_DEG = 1.0


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _norm(a):
    return math.sqrt(_dot(a, a))


def _mul(R, v):
    return tuple(_dot(R[i], v) for i in range(3))


def _tmul(R, v):
    return (
        R[0][0] * v[0] + R[1][0] * v[1] + R[2][0] * v[2],
        R[0][1] * v[0] + R[1][1] * v[1] + R[2][1] * v[2],
        R[0][2] * v[0] + R[1][2] * v[1] + R[2][2] * v[2],
    )


def _col(R, i):
    return (R[0][i], R[1][i], R[2][i])


def yaw180(local, drop=0.0):
    """Asset point into the world at the starting yaw, with the height slide applied."""
    x, y, z = local
    ox, oy, oz = ORIGIN
    return (ox - x, oy - y, oz + z + drop)


def new_state():
    return {
        "failed": False,
        "slide": [0.0] * len(MARKS),
        "clean": [False] * len(MARKS),
        "prev": None,
        "window": [],
        "score": 0.0,
        "success": False,
    }


def _box_distance(point, origin, R, box):
    local = _tmul(R, _sub(point, origin))
    lo, hi = box
    clamped = tuple(min(max(local[i], lo[i]), hi[i]) for i in range(3))
    return _norm(_sub(local, clamped))


def foot_ok(base_pos, base_R) -> bool:
    """All four underside corners on the desktop, and the foot not tipped past 8 degrees."""
    up = _col(base_R, 2)
    if _dot(up, (0.0, 0.0, 1.0)) < TILT_COS:
        return False
    for corner in FOOT:
        c = _add(base_pos, _mul(base_R, corner))
        if not (DESK_X[0] <= c[0] <= DESK_X[1] and DESK_Y[0] <= c[1] <= DESK_Y[1]):
            return False
        if not (TOP - 0.008 <= c[2] <= TOP + 0.012):
            return False
    return True


def covering(mark, cloth_pos, cloth_R, glass_normal) -> bool:
    """The pad's broad face is on this mark and square to the glass."""
    if abs(_dot(_col(cloth_R, 2), glass_normal)) < ALIGN:
        return False
    local = _tmul(cloth_R, _sub(mark, cloth_pos))
    half = tuple(s / 2 for s in CLOTH_SIZE)
    return abs(local[0]) <= half[0] + 0.004 and abs(local[1]) <= half[1] + 0.004 and abs(local[2]) <= half[2] + GAP


def hands_clear(tips, base_pos, base_R, panel_pos, panel_R) -> bool:
    for tip in tips:
        if _box_distance(tip, base_pos, base_R, BASE_BOX) < CLEAR_M:
            return False
        if _box_distance(tip, panel_pos, panel_R, PANEL_BOX) < CLEAR_M:
            return False
    return True


def on_glass(point, mark, panel_R) -> bool:
    """True when ``point`` is on the glass, inside the circle centred on ``mark``.

    ``along`` is the distance off the glass, positive out toward the robot. Only a contact
    gap counts. A point in the air, or the cloth lying on the desk, does not.
    """
    normal = _col(panel_R, 0)
    delta = _sub(point, mark)
    along = _dot(delta, normal)
    lateral = _sub(delta, (along * normal[0], along * normal[1], along * normal[2]))
    return _norm(lateral) <= STAIN_DIAMETER / 2 and abs(along) <= GAP


def _reach(state, obs) -> None:
    """Mark a stain clean when the gripper and the cloth are both on the glass inside its circle."""
    fingers = obs.get("fingers") or ()
    cloth = obs.get("cloth_pts") or ()
    origin, rotation = obs["panel_pos"], obs["panel_R"]
    for index, local in enumerate(MARKS):
        if state["clean"][index]:
            continue
        mark = _add(origin, _mul(rotation, local))
        arm = any(on_glass(point, mark, rotation) for point in fingers)
        sheet = any(on_glass(point, mark, rotation) for point in cloth)
        if arm and sheet:
            state["clean"][index] = True


def monitor_upright(panel_R) -> bool:
    """The monitor head has not fallen over. Its local +Z should still point mostly world +Z."""
    up = _col(panel_R, 2)
    return _dot(up, (0.0, 0.0, 1.0)) >= MONITOR_TILT_MAX


def step(state, obs) -> None:
    """One control step.

    Scoring:
      1.0 — all 8 stains cleaned and the monitor is still upright.
      0.5 — 4 or more stains cleaned, monitor upright.
      0.0 — fewer than 4 cleaned, or the monitor has fallen.
    """
    upright = monitor_upright(obs["panel_R"])
    if not upright:
        state["failed"] = True
    _reach(state, obs)
    state["prev"] = obs["cloth_pos"]
    n_clean = sum(state["clean"])
    if state["failed"]:
        state["score"] = 0.0
        state["success"] = False
    elif n_clean == len(MARKS):
        state["score"] = 1.0
        state["success"] = True
    elif n_clean >= 4:
        state["score"] = 0.5
        state["success"] = False
    else:
        state["score"] = 0.0
        state["success"] = False
