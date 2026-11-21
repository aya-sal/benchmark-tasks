"""Score for wiping the LG 27GX790A glass without the foot leaving the desk.

Pure geometry: no simulator imports. Poses are metres, rotations are 3x3 row-major
tuples (columns are the body's local axes in the world).
"""

from __future__ import annotations

import math

TOP = 0.7475
DESK_X = (-0.19, 0.91)   # station table, 1.10 m deep, centred at x = 0.36
DESK_Y = (-0.65, 0.65)   # 1.30 m wide, centred at y = 0
ORIGIN = (0.433, 0.470, TOP + 0.001)  # asset origin: bottom centre of the foot
YAW = math.pi
HEIGHT = -0.110          # height slide at its low stop (authored travel 0.11288 m)
# Glass is the flat face at local x = 0.0531, normal +x, y ±0.2956, z 0.2364..0.5732.
MARKS = tuple((0.0545, y, z) for z in (0.32, 0.50) for y in (-0.24, -0.08, 0.08, 0.24))
FOOT = tuple((x, y, 0.001) for x in (-0.120, 0.120) for y in (-0.150, 0.150))
# Link-frame boxes the hands must leave. Base includes the column; panel is the head.
BASE_BOX = ((-0.125, -0.155, 0.0), (0.125, 0.155, 0.447))
PANEL_BOX = ((-0.017, -0.303, 0.219), (0.054, 0.303, 0.581))
CLOTH_SIZE = (0.11, 0.11, 0.018)  # folded microfiber pad, thin axis local z
CLOTH_POS = (0.26, 0.08, TOP + 0.009)
SLIDE_M = 0.025
HOLD_STEPS = 40
CLEAR_M = 0.06
HOLD_OPENING = 0.045     # finger-gap sum, metres; full open is 0.094
HOLD_REACH = 0.07
GAP = 0.008              # cloth face may sit this far off a mark along its thickness
ALIGN = math.cos(math.radians(25))
TILT_COS = math.cos(math.radians(8))
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


def step(state, obs) -> None:
    """One control step. ``obs`` keys: base_pos, base_R, panel_pos, panel_R, cloth_pos, cloth_R, tips, held."""
    if state["failed"] or not foot_ok(obs["base_pos"], obs["base_R"]):
        state["failed"] = True
        state["window"].clear()
        state["score"] = 0.0
        state["success"] = False
        state["prev"] = obs["cloth_pos"]
        return
    normal = _col(obs["panel_R"], 0)  # glass faces local +x
    moved = (0.0, 0.0, 0.0) if state["prev"] is None else _sub(obs["cloth_pos"], state["prev"])
    state["prev"] = obs["cloth_pos"]
    along = _norm(_sub(moved, tuple(_dot(moved, normal) * n for n in normal)))
    for i, local in enumerate(MARKS):
        if state["clean"][i]:
            continue
        mark = _add(obs["panel_pos"], _mul(obs["panel_R"], local))
        if obs["held"] and covering(mark, obs["cloth_pos"], obs["cloth_R"], normal):
            state["slide"][i] += along
            if state["slide"][i] >= SLIDE_M:
                state["clean"][i] = True
    n_clean = sum(state["clean"])
    clear = hands_clear(obs["tips"], obs["base_pos"], obs["base_R"], obs["panel_pos"], obs["panel_R"])
    up = _col(obs["base_R"], 2)
    window = state["window"]
    window.append((obs["base_pos"], up) if n_clean == len(MARKS) and clear else None)
    del window[:-HOLD_STEPS]
    held_still = False
    if len(window) == HOLD_STEPS and all(sample is not None for sample in window):
        p0, u0 = window[0]
        held_still = all(
            _norm(_sub(p, p0)) < DRIFT_M and math.degrees(math.acos(max(-1.0, min(1.0, _dot(u, u0))))) < DRIFT_DEG
            for p, u in window
        )
    if n_clean == len(MARKS) and clear and held_still:
        value, done = 1.0, True
    elif n_clean == len(MARKS):
        value, done = 0.5, False
    elif n_clean >= 4:
        value, done = 0.25, False
    else:
        value, done = 0.0, False
    state["score"] = value
    state["success"] = done
