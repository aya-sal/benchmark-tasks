"""Geometry and measurements for pricing two boxed products on a moving belt with a hand price-label gun.

Two light retail boxes, each with a white price panel printed on its top face, ride a slow belt
conveyor (conveyor.py) past the arms. Beside the belt a price gun lies on a low rest block with its
nose roller clear of the bench. The gun is held by its handle and squeeze lever (the trigger);
squeezing the lever feeds one label to the nose, and touching the nose roller on a surface sticks
that label there.
Frame: metres, x away from the arm bases, y left, z up; TOP is the bench.
"""
from __future__ import annotations

import math

import numpy as np

import conveyor as C
from simkit import Hinge, Part, Piece

TOP = C.TOP

WHITE = (0.97, 0.97, 0.95)
INK = (0.06, 0.06, 0.07)
LABEL_COL = (0.97, 0.36, 0.08)       # fluorescent orange price labels
RUBBER = (0.08, 0.08, 0.09)
GUN_COL = (0.90, 0.90, 0.87)         # off-white plastic body
TRIGGER_COL = (0.78, 0.10, 0.10)
REST_COL = (0.22, 0.23, 0.25)

# -- labels and price panels ---------------------------------------------------------------------
LABEL = (0.022, 0.012)               # long edge (across the gun, along the roller), short edge
PANEL = (0.040, 0.028)               # the printed price panel on each box's top face (x by y)
N_LABELS = 6                         # labels loaded in the gun
THIN = 0.0005

# -- the two boxes (free), riding the belt ----------------------------------------------------------
# size x, y, z; centre x, y on the belt when the belt starts; mass; colour; price panel centre on the
# top face (box frame). The shorter green box leads; the flat blue one follows it, 30 mm behind.
# Both are short enough that the belt's surface velocity does not tip them when it ramps up.
BOXES = {
    # Start far enough upstream that the right arm can take the gun without sweeping into either box.
    "box_a": dict(size=(0.110, 0.160, 0.045), centre=(C.X, -0.620), mass=0.18, color=(0.13, 0.36, 0.66),
                  panel=(-0.020, 0.030)),
    "box_b": dict(size=(0.090, 0.100, 0.050), centre=(C.X, -0.460), mass=0.22, color=(0.16, 0.50, 0.26),
                  panel=(0.0, 0.0)),
}
BOX_NAMES = tuple(BOXES)
BOX_FRICTION = (1.0, 0.85)           # printed carton board on the belt


def box_pos(name, travel=0.0):
    """The box's centre where the belt has carried it, ``travel`` metres on from where it started."""
    b = BOXES[name]
    return np.array([b["centre"][0], b["centre"][1] + travel, TOP + C.CARRIER_TOP + b["size"][2] / 2])


def panel_world(name, travel=0.0):
    """Centre of the box's price panel on its top face, ``travel`` metres on from where it started."""
    b = BOXES[name]
    return np.array([b["centre"][0] + b["panel"][0], b["centre"][1] + b["panel"][1] + travel,
                     TOP + C.CARRIER_TOP + b["size"][2]])


def outline_objects():
    return [C.outline_object(name, b["size"], b["centre"]) for name, b in BOXES.items()]


def carrier_objects():
    return [C.carrier_object(name, b["size"], b["centre"]) for name, b in BOXES.items()]


def _label_pieces():
    return [Piece("box", (LABEL[0], LABEL[1], THIN), (0.0, 0.0, 0.0), color=LABEL_COL, collide=False)
            for _ in range(N_LABELS)]


def _box_art(name):
    """Printed artwork on the top face: the white price panel, a barcode and a brand band (visual only)."""
    b = BOXES[name]
    lx, ly, h = b["size"]
    z = h / 2 + 0.0002
    px, py = b["panel"]
    art = [Piece("box", (PANEL[0], PANEL[1], 0.0004), (px, py, z), color=WHITE, collide=False)]
    if name == "box_a":
        art.append(Piece("box", (lx, 0.022, 0.0004), (0.0, -0.045, z), color=(0.06, 0.16, 0.36), collide=False))
        art.append(Piece("box", (0.034, 0.020, 0.0004), (0.030, 0.030, z), color=WHITE, collide=False))   # barcode
        art += [Piece("box", (0.003, 0.016, 0.0004), (0.030 + dx, 0.030, z + 0.0003), color=INK, collide=False)
                for dx in (-0.012, -0.006, 0.0, 0.006, 0.012)]
    else:
        # the tall box carries its barcode and band on the face toward the arms
        x = -lx / 2 - 0.0002
        art.append(Piece("box", (0.0004, ly, 0.018), (x, 0.0, 0.020), color=(0.05, 0.25, 0.10), collide=False))
        art.append(Piece("box", (0.0004, 0.034, 0.020), (x, 0.0, -0.012), color=WHITE, collide=False))
        art += [Piece("box", (0.0004, 0.003, 0.016), (x - 0.0003, dy, -0.012), color=INK, collide=False)
                for dy in (-0.012, -0.006, 0.0, 0.006, 0.012)]
    return art


BOX_LABEL0 = {name: len(_box_art(name)) for name in BOXES}     # index of a box's first label piece


def box_object(name) -> Part:
    b = BOXES[name]
    return Part(name, pos=tuple(float(v) for v in box_pos(name)), size=b["size"], exact_z=True, mass=b["mass"],
                color=b["color"], friction=BOX_FRICTION, restitution=0.0, contact_offset=0.002,
                pieces=tuple(_box_art(name) + _label_pieces()))


# -- the gun's rest (fixed) --------------------------------------------------------------------------
# A low rubber block beside the belt that the gun's body lies on, so its nose roller hangs clear of the bench.
GUN_XY = (0.130, -0.320)             # the gun's body centre at the start (clear of the belt)
REST_SIZE = (0.050, 0.070, 0.020)
REST_NAME = "gun_rest"
REST_LABEL0 = 0


def rest_pos():
    return np.array([GUN_XY[0], GUN_XY[1], TOP + REST_SIZE[2] / 2])


def rest_object() -> Part:
    """The rest block. Its hidden label pieces are the ones drawn where a label lands on the bench or the belt."""
    return Part(REST_NAME, pos=tuple(float(v) for v in rest_pos()), size=REST_SIZE, exact_z=True, color=REST_COL,
                friction=(1.0, 0.8), restitution=0.0, contact_offset=0.002, pieces=tuple(_label_pieces()))


# -- price gun (free) --------------------------------------------------------------------------------
# Gun frame: f forward (toward the nose), a across, u up. The gun starts with its nose toward +y, so
# its body frame is the world's at the start: across = x, forward = y, up = z.
def L(f, a, u):
    """A point given as (forward, across, up) in the gun's body frame."""
    return np.array([a, f, u], float)


def S(lf, la, lu):
    """Box edges given as (forward, across, up) in the gun's body frame."""
    return (la, lf, lu)


GUN_MASS = 0.22
BODY = (0.090, 0.036, 0.030)         # forward, across, up
HANDLE_F = (-0.038, -0.022)          # the fixed handle's rear and front faces
HANDLE_U = (0.015, 0.062)            # it stands on the body's top
HANDLE_W = 0.030
LEVER_T = 0.008                      # the squeeze lever (trigger), a bar in front of the handle
LEVER_W = 0.026
LEVER_F0 = HANDLE_F[1] + 0.0005      # its rear face when squeezed home
LEVER_U = (0.017, 0.060)
LEVER_OPEN_DEG = 16.0                # how far its spring swings it forward
LEVER_K = 0.0048                     # N m per degree, toward LEVER_TARGET_DEG: about 9 N at its tip, 10 N squeezed
LEVER_TARGET_DEG = 90.0
LEVER_MASS = 0.020
ROLLER = (0.008, 0.026)              # radius, width
ROLLER_AT = (0.078, -0.015)          # roller axis: forward, up. Its bottom is 8 mm below the body's
ROLLER_MASS = 0.010
FEED_AT = 0.85                       # lever squeezed this fraction of its travel feeds a label
REARM_AT = 0.30                      # and it must come back out to this before it can feed again
TOUCH = 0.002                        # the roller this near a surface sticks the label on

GRIP_U = 0.030                       # where the pad tips go on the handle, above the body centre
GUN_Z = TOP + REST_SIZE[2] + BODY[2] / 2


def gun_pos():
    return np.array([GUN_XY[0], GUN_XY[1], GUN_Z])


def _lever_tip_open():
    """The open lever's top front corner, (forward, up) in the gun frame: where a pad meets it."""
    a = math.radians(LEVER_OPEN_DEG)
    f, u = LEVER_T / 2, LEVER_U[1] - LEVER_U[0]
    return (LEVER_F0 + LEVER_T / 2 + f * math.cos(a) + u * math.sin(a), LEVER_U[0] - f * math.sin(a) + u * math.cos(a))


SPAN_SHUT = LEVER_F0 + LEVER_T - HANDLE_F[0]          # handle's back to the lever's front, squeezed
SPAN_OPEN = _lever_tip_open()[0] - HANDLE_F[0]        # and with the lever sprung out
GRIP_F_SHUT = HANDLE_F[0] + SPAN_SHUT / 2             # the middle of that span: where the hand's centre sits
GRIP_F_OPEN = HANDLE_F[0] + SPAN_OPEN / 2

PRESENT_PIECE = 6                    # index of the fed label among the gun's pieces


def gun_object() -> Part:
    hf = (HANDLE_F[0] + HANDLE_F[1]) / 2
    hu = (HANDLE_U[0] + HANDLE_U[1]) / 2
    rf, ru = ROLLER_AT
    pieces = (
        Piece("box", S(HANDLE_F[1] - HANDLE_F[0], HANDLE_W, HANDLE_U[1] - HANDLE_U[0]), tuple(L(hf, 0.0, hu)),
              color=RUBBER, friction=(1.4, 1.2)),                                               # handle
        Piece("box", S(0.040, 0.004, 0.012), tuple(L(0.060, 0.017, -0.011)), collide=False),    # nose side plates
        Piece("box", S(0.040, 0.004, 0.012), tuple(L(0.060, -0.017, -0.011)), collide=False),
        Piece("cylinder", (0.013, BODY[1] + 0.004, 0.013), tuple(L(0.012, 0.0, 0.0)), axis="X", color=LABEL_COL,
              collide=False),                                                                   # label roll, in the body
        Piece("cylinder", (0.005, BODY[1] + 0.006, 0.005), tuple(L(0.012, 0.0, 0.0)), axis="X", color=RUBBER,
              collide=False),                                                                   # its hub
        Piece("box", S(0.036, LABEL[0], 0.001), tuple(L(0.060, 0.0, -0.0045)), color=LABEL_COL, collide=False),  # strip to the nose
        Piece("box", S(0.0006, LABEL[0], LABEL[1]), tuple(L(rf + ROLLER[0] + 0.0012, 0.0, ru - 0.002)), color=LABEL_COL,
              collide=False),                                                                   # the fed label at the nose
    )
    assert pieces[PRESENT_PIECE].size == S(0.0006, LABEL[0], LABEL[1])
    return Part("gun", pos=tuple(float(v) for v in gun_pos()), size=S(*BODY), exact_z=True, mass=GUN_MASS,
                color=GUN_COL, friction=(0.6, 0.5), restitution=0.0, contact_offset=0.0015,
                com=tuple(L(-0.012, 0.0, 0.0)), pieces=pieces)


def trigger_object() -> Part:
    """The squeeze lever: hinged at its foot to the gun, sprung forward, squeezed back against the handle."""
    g = gun_pos()
    c = g + L(LEVER_F0 + LEVER_T / 2, 0.0, (LEVER_U[0] + LEVER_U[1]) / 2)
    pivot = g + L(LEVER_F0 + LEVER_T / 2, 0.0, LEVER_U[0])
    # about +x (across), the lever's top swings forward (+y) for negative angles. Smooth plastic: its tip
    # rises 3 mm as it swings home, and has to slide on the jaw rather than drag the gun down in the grip.
    return Part("trigger", pos=tuple(float(v) for v in c), size=S(LEVER_T, LEVER_W, LEVER_U[1] - LEVER_U[0]),
                exact_z=True, mass=LEVER_MASS, color=TRIGGER_COL, friction=(0.35, 0.30), restitution=0.0,
                contact_offset=0.0015,
                hinge=Hinge(pivot=tuple(float(v) for v in pivot), axis="X", damping=0.0002,
                            lower_deg=-LEVER_OPEN_DEG, upper_deg=0.0, stiffness=LEVER_K,
                            target_deg=-LEVER_TARGET_DEG, parent="gun", parent_pos=tuple(float(v) for v in g)))


def roller_object() -> Part:
    """The rubber applicator roller at the nose, on its own free-turning axle."""
    g = gun_pos()
    c = g + L(ROLLER_AT[0], 0.0, ROLLER_AT[1])
    r, w = ROLLER
    marks = []
    for sx in (-1, 1):                                              # yellow end caps with a black bar: its turning shows
        x = sx * (w / 2 + 0.0008)
        marks.append(Piece("cylinder", (r + 0.001, 0.0016, r + 0.001), (x, 0.0, 0.0), axis="X",
                           color=(0.98, 0.78, 0.05), collide=False))
        marks.append(Piece("box", (0.0020, 0.003, 2 * r + 0.0015), (x + sx * 0.0003, 0.0, 0.0), color=INK, collide=False))
    return Part("roller", pos=tuple(float(v) for v in c), size=(r, w, r), shape="cylinder", axis="X", exact_z=True,
                mass=ROLLER_MASS, color=RUBBER, friction=(1.2, 1.0), restitution=0.0, contact_offset=0.0015,
                pieces=tuple(marks),
                hinge=Hinge(pivot=tuple(float(v) for v in c), axis="X", damping=0.00002, lower_deg=-1e6, upper_deg=1e6,
                            parent="gun", parent_pos=tuple(float(v) for v in g)))


# -- measurements, from measured poses only ----------------------------------------------------------
def quat_matrix(q):
    x, y, z, w = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def matrix_quat(r):
    """(x, y, z, w) of a rotation matrix."""
    t = np.trace(r)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = ((r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s, 0.25 * s)
    else:
        i = int(np.argmax(np.diag(r)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = math.sqrt(max(1e-12, 1.0 + r[i, i] - r[j, j] - r[k, k])) * 2
        q = [0.0, 0.0, 0.0, (r[k, j] - r[j, k]) / s]
        q[i], q[j], q[k] = 0.25 * s, (r[j, i] + r[i, j]) / s, (r[k, i] + r[i, k]) / s
    return tuple(float(v) for v in q)


def trigger_fraction(gun_pose, trigger_pose):
    """How far the lever is squeezed: 0 sprung out, 1 home against the handle."""
    rel = quat_matrix(np.asarray(gun_pose)[3:7]).T @ quat_matrix(np.asarray(trigger_pose)[3:7])
    angle = math.degrees(math.atan2(rel[2, 1], rel[1, 1]))          # about the gun's across axis; out is negative
    return float(np.clip(1.0 + angle / LEVER_OPEN_DEG, 0.0, 1.0))


def handle_centre(gun_pose):
    p = np.asarray(gun_pose, float)
    return p[:3] + quat_matrix(p[3:7]) @ L((HANDLE_F[0] + HANDLE_F[1]) / 2, 0.0, (HANDLE_U[0] + HANDLE_U[1]) / 2)


def gun_lowest(gun_pose, roller_pose):
    """Height of the gun's lowest point above the bench."""
    p = np.asarray(gun_pose, float)
    r = quat_matrix(p[3:7])
    zs = [float(np.asarray(roller_pose)[2]) - ROLLER[0]]
    for f in (-BODY[0] / 2, BODY[0] / 2):
        for a in (-BODY[1] / 2, BODY[1] / 2):
            zs.append(float((p[:3] + r @ L(f, a, -BODY[2] / 2))[2]))
            zs.append(float((p[:3] + r @ L(f, a, BODY[2] / 2))[2]))
    for f in HANDLE_F:
        for a in (-HANDLE_W / 2, HANDLE_W / 2):
            zs.append(float((p[:3] + r @ L(f, a, HANDLE_U[1]))[2]))
    return min(zs) - TOP


def gun_over_conveyor(gun_pose, roller_pose):
    """Whether any of the gun (its body's corners, its handle or its roller) is above the conveyor."""
    p = np.asarray(gun_pose, float)
    r = quat_matrix(p[3:7])
    points = [np.asarray(roller_pose, float)[:3]]
    points += [p[:3] + r @ L(f, a, 0.0) for f in (-BODY[0] / 2, BODY[0] / 2) for a in (-BODY[1] / 2, BODY[1] / 2)]
    points += [p[:3] + r @ L(f, a, HANDLE_U[1]) for f in HANDLE_F for a in (-HANDLE_W / 2, HANDLE_W / 2)]
    return any(C.over_conveyor(q) for q in points)


def _closest_on_box(local, half):
    """Closest point of a box (centred, half-extents ``half``) to ``local``: (signed distance, point, face normal)."""
    local, half = np.asarray(local, float), np.asarray(half, float)
    inside = half - np.abs(local)
    if np.all(inside > 0):                                 # inside the box: out through the nearest face
        i = int(np.argmin(inside))
        q = local.copy()
        s = 1.0 if local[i] >= 0 else -1.0
        q[i] = s * half[i]
        n = np.zeros(3)
        n[i] = s
        return -float(inside[i]), q, n
    q = np.clip(local, -half, half)
    d = local - q
    i = int(np.argmax(np.abs(d)))
    n = np.zeros(3)
    n[i] = 1.0 if d[i] >= 0 else -1.0
    return float(np.linalg.norm(d)), q, n


def box_distance(name, pose, point):
    """(distance from the box's surface, closest point and face normal in the box frame)."""
    p = np.asarray(pose, float)
    local = quat_matrix(p[3:7]).T @ (np.asarray(point, float) - p[:3])
    return _closest_on_box(local, np.asarray(BOXES[name]["size"]) / 2)


def box_tilt(pose):
    """How far the box leans from upright (degrees)."""
    z = quat_matrix(np.asarray(pose)[3:7])[2, 2]
    return math.degrees(math.acos(float(np.clip(z, -1.0, 1.0))))


def roller_touch(root_poses):
    """What the nose roller is touching: (surface name, point, face normal, roller axis), the last three in
    that surface's frame (the world for the bench, the belt and the rest block), or None.

    The roller touches a surface when the middle of its axle is within its radius plus TOUCH of it.
    """
    c = np.asarray(root_poses["roller"], float)[:3]
    axis_w = quat_matrix(np.asarray(root_poses["gun"])[3:7])[:, 0]         # the roller's axle: the gun's across axis
    reach = ROLLER[0] + TOUCH
    best = None
    for name in BOX_NAMES:
        pose = np.asarray(root_poses[name], float)
        d, q, n = box_distance(name, pose, c)
        if d <= reach and (best is None or d < best[0]):
            best = (d, name, q, n, quat_matrix(pose[3:7]).T @ axis_w)
    d, q, n = _closest_on_box(c - rest_pos(), np.asarray(REST_SIZE) / 2)
    if d <= reach and (best is None or d < best[0]):
        best = (d, REST_NAME, q + rest_pos(), n, axis_w)
    floor, z = ("belt", TOP + C.CARRIER_TOP) if C.over_conveyor(c) else ("bench", TOP)
    d = float(c[2] - z)
    if d <= reach and (best is None or d < best[0]):
        best = (d, floor, np.array([c[0], c[1], z]), np.array([0.0, 0.0, 1.0]), axis_w)
    return None if best is None else best[1:]


def label_frame(normal, axis):
    """The stuck label's rotation in its surface's frame: long edge along the roller, flat on the face."""
    n = np.asarray(normal, float)
    u = np.asarray(axis, float) - n * float(np.dot(axis, n))
    if np.linalg.norm(u) < 1e-6:                           # the axle square to the face: any direction in it
        u = np.cross(n, [1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.cross(n, [0.0, 1.0, 0.0])
    u = u / np.linalg.norm(u)
    return np.column_stack([u, np.cross(n, u), n])


def panel_margin(name, point, normal, rot):
    """How far inside the box's price panel the label's worst corner lies (m; negative: outside it).

    None when the label is not on the box's top face.
    """
    if normal[2] < 0.5:
        return None
    px, py = BOXES[name]["panel"]
    worst = math.inf
    for su in (-1, 1):
        for sv in (-1, 1):
            corner = np.asarray(point) + su * LABEL[0] / 2 * rot[:, 0] + sv * LABEL[1] / 2 * rot[:, 1]
            worst = min(worst, PANEL[0] / 2 - abs(corner[0] - px), PANEL[1] / 2 - abs(corner[1] - py))
    return float(worst)
