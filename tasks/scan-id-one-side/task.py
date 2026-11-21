"""Copy one side of an ID card onto a sheet using a flatbed scanner/copier.

Pick up the ID card, place it face-down on the platen glass, close the lid, press the Scan button, and wait
for the copier to scan and print. The sheet starts standing in the rear slit. After the scan the rollers take
it from the slit, a patch is laid down where the card sat on the glass, and the sheet comes out of the front
mouth face up on the tray.

Run it:  inspect-robots run --embodiment yam_arena --policy yam_scripted \\
           -E task=task.py -E object_jitter=0 -E control_hz=20 \\
           --instruction "<see task.json>" --scorer task_score --max-steps 1200 --no-prompt --grader none
"""

import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import labkit  # noqa: E402
import sidegrip  # noqa: E402  (before the demo is built: it lets a Waypoint carry a full gripper rotation)
from labkit import Joint, Mesh, on_reset, on_step, part  # noqa: E402
from yam_arena import ik  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN, LEFT_BASE_POS, READY, RIGHT_BASE_POS, TCP_OFFSET  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]

DARK = (0.16, 0.17, 0.19)
GREY = (0.55, 0.56, 0.58)
WHITE = (0.94, 0.94, 0.92)
BLUE = (0.16, 0.36, 0.78)
RED = (0.78, 0.16, 0.14)
CARD_BODY = (0.78, 0.74, 0.64)
BLACK = (0.08, 0.08, 0.09)
LAMP_C = (0.95, 0.85, 0.45)
GREEN = (0.20, 0.55, 0.28)

# Dimensions in metres. Paper feeds portrait, short edge first, along the chassis X axis. The platen is landscape:
# its long side runs along Y, so a place on the glass maps onto the sheet turned a quarter turn.
CARD = (0.0856, 0.05398, 0.00076)  # x, y, z
PAPER = (0.297, 0.210, 0.00010)
GLASS = (0.220, 0.300)  # visible platen, x (depth) by y (width)
PRINT = (CARD[1], CARD[0])  # the card's patch on the sheet, in sheet x and y
HALF_C = (CARD[0] / 2, CARD[1] / 2)
HALF_P = (PAPER[0] / 2, PAPER[1] / 2)
HALF_I = (PRINT[0] / 2, PRINT[1] / 2)

SX, SY = 0.30, 0.035
PX, PY = SX, SY
GLASS_Z = 0.125  # top of the recessed platen, in the scanner frame
CARRIER_RISE = .003
SEAT_Z = GLASS_Z + CARRIER_RISE + CARD[2] / 2
MOUTH_X = -0.160  # printer-frame x of the front exit mouth
TRAY_Z = 0.200  # height of the raised exit tray's top, where a printed sheet comes to rest, within reach of a side pinch
SLOT_Z = 0.035  # printer-frame height of the sheet's centre in the paper path
INSIDE_X = -0.010  # printer-frame x of the sheet's centre while the roller runs; the whole sheet is under the platen
EJECT_X = -0.260  # sheet centre this far outside the mouth after the pass

# Rear feed: a top-facing slit behind the platen. The sheet stands in it on a short edge, printable face to the front.
# The feeder leans back 25 degrees from vertical (like a real printer's rear paper tray).
SLIT_LEAN = math.radians(25)  # lean angle away from the printer body
SLIT_X = 0.164  # printer-frame x of the middle of the slit (at the base)
SLIT_GAP = 0.008  # front wall to paper support
SLIT_FLOOR = 0.030  # printer-frame height the sheet's lower edge rests at
SLIT_GUIDE_Y = 0.113  # side guides, 8 mm clear of each long edge
SUPPORT_TOP = 0.200  # top of the paper support behind the slit
STAND_Q = (0.0, -math.sin(math.radians(65) / 2), 0.0, math.cos(math.radians(65) / 2))  # sheet leans back 25 deg
FLAT_Q = (0.0, 0.0, 0.0, 1.0)

# Where the two copies sit on the glass: centres 100 mm apart across the platen, the card 54 mm across, so a
# 46 mm gap. On the sheet they land one above the other along its length.
LEFT_XY = (0.0, 0.050)
RIGHT_XY = (0.0, -0.050)

LID_OPEN = 1.55  # nearly upright, so the platen in front of the hinge is clear
LID_SHUT = 0.07  # closer than this and the detent takes the lid
HINGE = (0.125, 0.0, GLASS_Z + 0.004)  # along the back edge of the platen
KNOB = (-0.235, 0.0, 0.010)  # from the hinge to the handle on the lid's front edge, at the closed angle
LAMP_HOME, LAMP_END = -0.140, 0.140
LAMP_SPEED = 0.10  # m/s
ROLLER_END = 0.120
ROLLER_SPEED = 0.08
BTN_ON = -0.005  # button slide, metres; more negative is further down
LATCH_HOLD = -0.014
LATCH_RELEASE = -0.004

CARD_XY0 = (0.15, 0.320)
CARD_STAND_Z = 0.023
PAPER_XY0 = (SX + MOUTH_X + EJECT_X, PY)  # where a printed sheet comes to rest on the exit tray
PAPER_START = (SX + SLIT_X + PAPER[0] / 2 * math.sin(SLIT_LEAN), PY,
               TOP + SLIT_FLOOR + PAPER[0] / 2 * math.cos(SLIT_LEAN) + 0.0005)  # standing in the slit, leaning back


def _slab(size, color, z=0.0, xy=(0.0, 0.0)) -> Mesh:
    """A visual box centred at z, with no collision."""
    hx, hy, hz = (float(v) / 2 for v in size)
    c = np.array([[x, y, z + w] for x in (-hx, hx) for y in (-hy, hy) for w in (-hz, hz)])
    order = [0, 1, 3, 2, 4, 5, 7, 6]
    p = c[order] + (xy[0], xy[1], 0.)
    faces = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    pts, tris = [], []
    for a, b, d, e in faces:
        base = len(pts)
        pts.extend([p[a], p[b], p[d], p[e]])
        tris.extend([(base, base + 1, base + 2), (base, base + 2, base + 3)])
    return Mesh(points=np.asarray(pts, float), tris=np.asarray(tris, int), color=color, collide="none")


# -- scanner ----------------------------------------------------------------------------------------------------------
_LID_BOXES = [
    ((.0025, 0, GLASS_Z + .011), (.245, .330, .014), WHITE),
    ((-.112, 0, GLASS_Z + .014), (.015, .060, .015), DARK),
]
# No spring: the lid is passive. Hinge friction holds it open near upright, and below LID_HANDOFF gravity (about
# 0.096*cos(angle) N.m) wins and swings it shut. Damping stops it bouncing off the glass.
LID_FRICTION = 0.02  # holds the open lid above roughly LID_HANDOFF; calibrated in a run
LID_DAMPING = 0.06
LID_HANDOFF = math.radians(78.0)  # held above this, falls below it
LID = Joint(
    "lid", kind="hinge", axis="Y", anchor=HINGE, lower=0., upper=1.62,
    initial=LID_OPEN, target=LID_OPEN, stiffness=0.0, damping=LID_DAMPING, friction=LID_FRICTION, mass=0.08,
    # upper keeps the open lid in front of a sheet standing in the rear slit
    boxes=_LID_BOXES,
)
# The control strip in front of the lid: the latch catches the lid's handle, the Scan key sits to the right.
BUTTON_ANCHOR = (-.140, -.150, GLASS_Z + .009)
LATCH_ANCHOR = (-.141, 0.0, GLASS_Z + .015)
BUTTON = Joint("button", kind="slide", axis="Z", anchor=BUTTON_ANCHOR, lower=-.008, upper=0.,
               stiffness=60., damping=2., mass=.01, boxes=[(BUTTON_ANCHOR, (.018,.016,.010), GREEN)])
LATCH = Joint("latch", kind="slide", axis="Z", anchor=LATCH_ANCHOR, lower=-.016, upper=.002,
              stiffness=40., damping=1.5, mass=.01, boxes=[(LATCH_ANCHOR, (.016,.040,.014), BLACK)])
LAMP = Joint("lamp", kind="slide", axis="Y", anchor=(0,0,GLASS_Z-.006),
             lower=LAMP_HOME-.002, upper=LAMP_END+.002, initial=LAMP_HOME, target=LAMP_HOME,
             stiffness=400, damping=40, mass=.02,
             boxes=[((0,0,GLASS_Z-.006), (.210,.012,.004), LAMP_C)])
ROLLER = Joint("roller", kind="slide", axis="X", anchor=(0,0,SLOT_Z+.008), lower=-.002,
               upper=ROLLER_END+.002, stiffness=400, damping=40, mass=.02,
               cylinders=[((0,0,SLOT_Z+.008), .007,.215,BLACK,"Y")])

# The shell bridges the paper tunnel. Its upper deck supports the platen.
# A clear strip in the glass exposes the moving lamp. No solid body fills the paper path.
# The rear feeder is bolted to the back wall: a floor, a tall paper support and two side guides around the slit.
_SLIT_BACK = SLIT_X + SLIT_GAP / 2
_SLIT_FRONT = SLIT_X - SLIT_GAP / 2
# The feeder leans back by SLIT_LEAN. Axis-aligned boxes approximate the tilt with stacked slices.
_SL_SIN, _SL_COS = math.sin(SLIT_LEAN), math.cos(SLIT_LEAN)
_SUPPORT_SLICES = 4
_FEEDER_BOXES = [
    (((_SLIT_FRONT + _SLIT_BACK + .016) / 2, 0, SLIT_FLOOR / 2), (SLIT_GAP + .016, .250, SLIT_FLOOR), DARK),
]
for _i in range(_SUPPORT_SLICES):
    _lo = SLIT_FLOOR + (SUPPORT_TOP - SLIT_FLOOR) * _i / _SUPPORT_SLICES
    _hi = SLIT_FLOOR + (SUPPORT_TOP - SLIT_FLOOR) * (_i + 1) / _SUPPORT_SLICES
    _mid_h = (_lo + _hi) / 2
    _dx = (_mid_h - SLIT_FLOOR) * _SL_SIN  # lean-back offset
    _FEEDER_BOXES.append(
        ((_SLIT_BACK + .008 + _dx, 0, _mid_h), (.016, .250, _hi - _lo), GREY))
# Side guides, tilted to match the lean
_GUIDE_MID_H = 0.095
_GUIDE_DX = (_GUIDE_MID_H - SLIT_FLOOR) * _SL_SIN
_FEEDER_BOXES += [
    ((SLIT_X + _GUIDE_DX, SLIT_GUIDE_Y + .004, _GUIDE_MID_H), (SLIT_GAP, .008, .130), DARK),
    ((SLIT_X + _GUIDE_DX, -SLIT_GUIDE_Y - .004, _GUIDE_MID_H), (SLIT_GAP, .008, .130), DARK),
    # the bed the sheet lies on while it is under the platen
    (((MOUTH_X + .146) / 2, 0, (.026 + SLOT_Z) / 2), (.146 - MOUTH_X, .235, SLOT_Z - .026), GREY),
]
ALL_IN_ONE = part("all_in_one", pos=(SX,SY,TOP), fixed=True, friction=.6,
    boxes=_FEEDER_BOXES + [
        ((0,0,.013), (.32,.42,.026), WHITE),
        ((0,-.189,.066), (.32,.042,.108), WHITE),
        ((0,.189,.066), (.32,.042,.108), WHITE),
        ((.153,0,.066), (.014,.34,.108), GREY),
        ((0,0,.090), (.32,.34,.050), WHITE),
        ((0,0,GLASS_Z-.002), (GLASS[0]+.010,GLASS[1]+.010,.004), (.75,.82,.86)),
        ((0,0,GLASS_Z+CARRIER_RISE/2), (.035,.260,CARRIER_RISE), (.75,.82,.86)),
        ((-(GLASS[0]/2+.025),0,GLASS_Z-.004), (.050,.420,.008), DARK),
        (((GLASS[0]/2+.025),0,GLASS_Z-.004), (.050,.420,.008), DARK),
        ((0,(GLASS[1]/2+.030),GLASS_Z-.004), (GLASS[0],.060,.008), DARK),
        ((0,-(GLASS[1]/2+.030),GLASS_Z-.004), (GLASS[0],.060,.008), DARK),
       
    ], joints=[LID,BUTTON,LATCH,LAMP,ROLLER])

CARD_STAND = part("card_stand", pos=(*CARD_XY0,TOP), fixed=True,
                  boxes=[((0,0,CARD_STAND_Z/2), (.055,.027,CARD_STAND_Z), DARK)])

def artwork(front, z, normal=1):
    # Abstract markings only: a portrait block and stripes, with no personal data.
    base = _slab((CARD[0]-.001,CARD[1]-.001,.000004), BLUE if front else RED, z)
    details = [(-.026,0,.018,.025),(.008,.014,.035,.002),(.008,.006,.035,.002)] if front else [
        (0,.016,.076,.008),(0,-.008,.070,.002)]
    return [base] + [_slab((w,h,.000004), BLACK, z+normal*.000006, (x,y)) for x,y,w,h in details]


CARD_PART = part("card", pos=(*CARD_XY0, TOP+CARD_STAND_Z+CARD[2]/2),
    mass=.005, friction=1.0, boxes=[((0,0,0), CARD, CARD_BODY)],
    meshes=artwork(True,CARD[2]/2+.000002) + artwork(False,-CARD[2]/2-.000002,-1))
PAPER_PART = part("paper", pos=PAPER_START, rot_xyzw=STAND_Q,
                  mass=.005,friction=.8,boxes=[((0,0,0), PAPER, WHITE)])
INK_FRONT = part("ink_front", pos=(SX,SY,TOP+.10), mass=.001, disable_gravity=True, meshes=artwork(True,0))
INK_BACK = part("ink_back", pos=(SX,SY,TOP+.11), mass=.001, disable_gravity=True, meshes=artwork(False,0))

OBJECTS = [ALL_IN_ONE, CARD_STAND, CARD_PART, PAPER_PART, INK_FRONT, INK_BACK]


# -- mechanism --------------------------------------------------------------------------------------------------------
def _fresh():
    return {
        "lamp": LAMP_HOME,
        "lamp_dir": 0,  # 0 idle, +1 across the glass, -1 back home
        "pass_ok": False,
        "pass_face": None,
        "pass_xy": None,
        "armed": True,  # the button has been seen up, so the next press counts
        "detent": False,
        "opened": False,  # set once the lid is seen open, so a shut reading at spawn cannot latch it
        "seen_open": False,  # stays true once the lid has been seen open
        "pending": {},  # face -> (x, y) on the glass, not yet printed
        "order": [],
        "printed": {},  # face -> (x, y) where it was printed on the sheet
        "roller": 0.0,
        "rolling": None,  # face currently being printed
        "feed": None,  # "in" or "out"
        "feed_q": None,  # start quat xyzw
        "feed_a": None,
        "feed_b": None,
        "feed_u": 0.0,
        "left_slot": True,  # the sheet has been fully out of the mouth
        "hold": 0.0,
        "t": 0.0,
        "shut_for": 0.0,
    }


_S = _fresh()


def _face(rot) -> str | None:
    up = float(rot[2, 2])
    if up <= -0.96:
        return "front"  # the blue face, local +Z, is against the glass
    if up >= 0.96:
        return "back"
    return None


def _card_on_glass(card, scanner) -> tuple[str, np.ndarray] | None:
    local = scanner.to_local(card.pos)
    face = _face(card.rotation())
    on = (
        face is not None
        and abs(card.rotation()[2, 2]) >= math.cos(math.radians(12.0))
        and abs(local[2] - SEAT_Z) <= 0.006
        and abs(local[0]) <= 0.016
        and abs(local[1]) <= 0.095
        and abs(card.lin_vel[0]) < 0.05
        and abs(card.lin_vel[1]) < 0.05
    )
    if not on:
        return None
    return face, np.array([local[0], local[1]])


LID_PASS_MAX = 0.12  # rad: above this the lid is not shut on the glass, so the lamp must not scan


def _lid_down(lid: float) -> bool:
    return float(lid) <= LID_PASS_MAX


def _glass_to_sheet(xy) -> tuple[float, float]:
    """Where a place on the landscape platen lands on the portrait sheet: across the glass is along the page."""
    return float(xy[1]), float(-xy[0])


PATCH_TURN = (0.0, 0.0, -math.sqrt(0.5), math.sqrt(0.5))  # the patch's card outline on the sheet, xyzw


def _qmul(a, b) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw,
                     aw * bw - ax * bx - ay * by - az * bz])


def _sheet_up(paper) -> bool:
    return paper.tilt_deg() <= 12.0 and float(paper.rotation()[2, 2]) >= 0.96 and abs(paper.rotation()[0, 0]) >= 0.96


def _mouth_world():
    return PX + MOUTH_X


def _leading_x(paper) -> float:
    """World x of the sheet's +X edge, the last one to leave the front mouth."""
    return float(paper.pos[0] + paper.rotation()[0, 0] * HALF_P[0])


def _fully_out(paper) -> bool:
    return _leading_x(paper) < _mouth_world() - 0.02


def _in_slit(paper) -> bool:
    """Let go in the rear slit: leaning on its -X edge within 12 degrees of the feeder's lean angle,
    printable face toward the front, lower edge between the slit walls and seated on the slit floor."""
    R = paper.rotation()
    foot = paper.pos - R[:, 0] * HALF_P[0]
    # The expected lean direction for the sheet's +X axis and the expected face normal (+Z body)
    lean_up = np.array([math.sin(SLIT_LEAN), 0.0, math.cos(SLIT_LEAN)])
    lean_face = np.array([-math.cos(SLIT_LEAN), 0.0, math.sin(SLIT_LEAN)])
    return bool(
        float(R[:, 0] @ lean_up) >= math.cos(math.radians(12.0))
        and float(R[:, 2] @ lean_face) >= 0.96
        and abs(foot[0] - (PX + SLIT_X)) <= SLIT_GAP
        and abs(paper.pos[1] - PY) <= 0.012
        and foot[2] - (TOP + SLIT_FLOOR) <= 0.008
        and float(np.linalg.norm(paper.lin_vel)) < 0.05
    )


def _write_pose(asset, pos_w, quat):
    import torch

    pose = torch.tensor([[*map(float, pos_w), *map(float, quat)]], dtype=torch.float32, device=asset.device)
    vel = torch.zeros((1, 6), dtype=torch.float32, device=asset.device)
    asset.write_root_pose_to_sim_index(root_pose=pose)
    asset.write_root_velocity_to_sim_index(root_velocity=vel)


def _drive(asset, targets: dict):
    import torch

    names = list(asset.joint_names)
    ids, vals = [], []
    for name, value in targets.items():
        ids.append(names.index(name))
        vals.append(float(value))
    asset.set_joint_position_target_index(target=torch.tensor([vals], dtype=torch.float32, device=asset.device), joint_ids=ids)


def _tcps(base, origin):
    out = []
    for arm in ("left_robot", "right_robot"):
        robot = base.scene[arm]
        i = next(j for j, n in enumerate(robot.body_names) if n == "gripper" or str(n).endswith("/gripper"))
        g = labkit._np(robot.data.body_link_pose_w)[0, i]
        rot = labkit.PartState(pos=g[:3], quat_xyzw=g[3:7], lin_vel=np.zeros(3), ang_vel=np.zeros(3),
                               low=np.zeros(3), high=np.zeros(3)).rotation()
        out.append(g[:3] - origin + rot @ np.array(TCP_OFFSET))
    return out


def _hands_clear(tcps, points, gap) -> bool:
    return all(np.linalg.norm(t - p) >= gap for t in tcps for p in points)


def _stick(base, origin, name, paper, xy):
    R = paper.rotation()
    local = np.array([xy[0], xy[1], PAPER[2] / 2 + 0.000012])
    pos = origin + paper.pos + R @ local
    _write_pose(base.scene[name], pos, _qmul(paper.quat_xyzw, PATCH_TURN))


def _park_ink(base, origin, name, side):
    pos = origin + np.array([PX, PY + side * 0.004, TOP + 0.020])
    _write_pose(base.scene[name], pos, np.array([0.0, 0.0, 0.0, 1.0]))


@on_reset
def _reset(env):
    _S.clear()
    _S.update(_fresh())


@on_step
def _machine(env):
    base = env.unwrapped
    origin = labkit._np(base.scene.env_origins[0])
    st = {n: labkit._part_state(env, labkit._SPECS[n]) for n in ("all_in_one", "card", "paper")}
    scanner, card, paper = st["all_in_one"], st["card"], st["paper"]
    dt = float(getattr(base, "step_dt", 0.05))
    lid = scanner.joints["lid"]
    button = scanner.joints["button"]
    latch = scanner.joints["latch"]

    t_prev = _S.get("t", 0.0)
    _S["t"] = t_prev + dt
    # The latch rests at 0. A release counts only after the detent has first pulled it down.
    # One touch against the stop does not count: the lid has to stay shut, so a bump cannot latch it.
    if (not _S["detent"]) and _S["opened"] and lid <= LID_SHUT:
        _S["shut_for"] = _S.get("shut_for", 0.0) + dt
        if _S["shut_for"] >= 0.30:
            _S["detent"] = True
            _S["opened"] = False
            _S["latch_caught"] = False
    else:
        _S["shut_for"] = 0.0
    if _S["detent"]:
        if latch <= -0.010:
            _S["latch_caught"] = True
        elif _S.get("latch_caught") and latch >= LATCH_RELEASE:
            _S["detent"] = False
            _S["latch_caught"] = False
    if lid > 0.50:
        _S["opened"] = True
        _S["seen_open"] = True
    # The lid is passive: friction holds it open, gravity shuts it. Pulling the latch only clears the detent;
    # an arm has to lift the lid afterwards.
    latch_target = LATCH_HOLD if _S["detent"] else 0.0

    # The lid is passive, so a pass needs the lid actually shut, not only the detent flag. Lifting the lid mid-pass
    # drops `seated`, and that voids the pass below.
    seated = _card_on_glass(card, scanner) if _S["detent"] and _lid_down(lid) else None
    if button > -0.0015:
        _S["armed"] = True
    if _S["lamp_dir"] == 0 and _S["armed"] and button <= BTN_ON and seated is not None and _S["feed"] is None:
        _S["lamp_dir"] = 1
        _S["pass_ok"] = True
        _S["pass_face"], _S["pass_xy"] = seated
        _S["armed"] = False
    if _S["lamp_dir"] == 1:
        if seated is None or not _S["detent"]:
            _S["pass_ok"] = False
            _S["lamp_dir"] = -1
        else:
            _S["pass_face"], _S["pass_xy"] = seated
            _S["lamp"] = min(LAMP_END, _S["lamp"] + LAMP_SPEED * dt)
            if _S["lamp"] >= LAMP_END - 1e-4:
                face, xy = _S["pass_face"], _S["pass_xy"]
                if _S["pass_ok"] and face not in _S["printed"]:
                    _S["pending"][face] = (float(xy[0]), float(xy[1]))
                    if face not in _S["order"]:
                        _S["order"].append(face)
                _S["lamp_dir"] = -1
    elif _S["lamp_dir"] == -1:
        _S["lamp"] = max(LAMP_HOME, _S["lamp"] - LAMP_SPEED * dt)
        if _S["lamp"] <= LAMP_HOME + 1e-4:
            _S["lamp_dir"] = 0
    _drive(base.scene["all_in_one"], {"lamp_joint": _S["lamp"], "latch_joint": latch_target})

    tcps = _tcps(base, origin)
    if _fully_out(paper) and _S["feed"] is None and _S["rolling"] is None:
        _S["left_slot"] = True

    waiting = next((f for f in _S["order"] if f in _S["pending"]), None)
    if _S["feed"] is None and _S["rolling"] is None and waiting is not None and _S["left_slot"] and _in_slit(paper):
        if all(np.linalg.norm(np.maximum(np.abs(paper.to_local(tcp)) - np.array(PAPER)/2, 0)) >= .04 for tcp in tcps):
            _S["feed"] = "in"
            _S["feed_u"] = 0.0
            _S["feed_a"] = np.array(paper.pos)
            _S["feed_b"] = np.array([PX + SLIT_X + HALF_P[0] * math.sin(SLIT_LEAN), PY,
                                      TOP + SLIT_FLOOR + HALF_P[0] * math.cos(SLIT_LEAN) + 0.0005])
            _S["feed_q"] = np.array(STAND_Q)
            _S["rolling"] = waiting
            _S["left_slot"] = False

    if _S["feed"] == "in":
        # The pick roller squares the sheet up in the slit. The turn under the platen is not modelled:
        # a rigid sheet cannot follow it, so the next step has the sheet lying on the bed inside.
        _S["feed_u"] = min(1.0, _S["feed_u"] + dt / 0.6)
        pos = _S["feed_a"] + _S["feed_u"] * (_S["feed_b"] - _S["feed_a"])
        _write_pose(base.scene["paper"], origin + pos, _S["feed_q"])
        paper_pos, paper_quat = pos, _S["feed_q"]
        if _S["feed_u"] >= 1.0:
            _S["feed"] = "roll"
            _S["roller"] = 0.0
            _S["feed_b"] = np.array([PX + INSIDE_X, PY, TOP + SLOT_Z + 0.0005])
            _S["feed_q"] = np.array(FLAT_Q)
    elif _S["feed"] == "roll":
        paper_pos, paper_quat = _S["feed_b"], _S["feed_q"]
        _write_pose(base.scene["paper"], origin + paper_pos, paper_quat)
        _S["roller"] = min(ROLLER_END, _S["roller"] + ROLLER_SPEED * dt)
        if _S["roller"] >= ROLLER_END - 1e-4:
            face = _S["rolling"]
            _S["printed"][face] = _glass_to_sheet(_S["pending"].pop(face))
            _S["feed"] = "out"
            _S["feed_u"] = 0.0
            _S["feed_a"] = np.array(paper_pos)
            _S["feed_b"] = np.array([_mouth_world() + EJECT_X, PY, TOP + TRAY_Z + 0.001])
    elif _S["feed"] == "out":
        _S["feed_u"] = min(1.0, _S["feed_u"] + dt / 2.5)
        a, b = _S["feed_a"], _S["feed_b"]
        pos = a + _S["feed_u"] * (b - a)
        # level through the mouth, and once the rear edge is clear it climbs onto the raised tray
        x_clear = _mouth_world() - HALF_P[0] - 0.01
        climb = min(1.0, max(0.0, (x_clear - pos[0]) / (x_clear - b[0])))
        pos[2] = a[2] + (b[2] - a[2]) * climb * climb * (3.0 - 2.0 * climb)
        _write_pose(base.scene["paper"], origin + pos, _S["feed_q"])
        paper_pos, paper_quat = pos, _S["feed_q"]
        if _S["feed_u"] >= 1.0:
            _S["feed"] = None
            _S["rolling"] = None
            _S["roller"] = 0.0
    else:
        paper_pos = None
    _drive(base.scene["all_in_one"], {"roller_joint": _S["roller"]})

    # Patches follow the face they were printed on. Unprinted patches stay inside the housing.
    if paper_pos is None:
        shown = paper
    else:
        shown = labkit.PartState(pos=paper_pos, quat_xyzw=paper_quat, lin_vel=np.zeros(3), ang_vel=np.zeros(3),
                                 low=np.zeros(3), high=np.zeros(3))
    for face, name, side in (("front", "ink_front", -1.0), ("back", "ink_back", 1.0)):
        if face in _S["printed"]:
            _stick(base, origin, name, shown, _S["printed"][face])
        else:
            _park_ink(base, origin, name, side)

    if _S["lamp_dir"] or _S["feed"] or _S["rolling"]:
        _S["hold"] = 0.0
    if int(_S["t"] // 1) != int(t_prev // 1):
        print(
            f"t={_S['t']:5.1f} lid={lid:+.2f} detent={int(_S['detent'])} lamp={_S['lamp']:+.3f}/{_S['lamp_dir']}"
            f" pending={list(_S['pending'])} printed={list(_S['printed'])} feed={_S['feed']}"
            f" card=({card.pos[0]:.3f},{card.pos[1]:.3f},z={card.pos[2] - TOP:.3f},{card.tilt_deg():.0f}°,up={card.rotation()[2, 2]:+.2f})"
            f" sheet=({paper.pos[0]:.3f},{paper.pos[1]:.3f},z={paper.pos[2] - TOP:.3f})",
            flush=True,
        )


# -- scoring ----------------------------------------------------------------------------------------------------------
SUCCESS_HOLD_S = 1.0


def _patch_on_sheet(ink, paper, xy) -> bool:
    local = paper.to_local(ink.pos)
    return abs(local[0] - xy[0]) <= 0.008 and abs(local[1] - xy[1]) <= 0.008 and local[2] > PAPER[2] / 2


def _inside(xy) -> bool:
    return abs(xy[0]) + HALF_I[0] <= HALF_P[0] - 0.002 and abs(xy[1]) + HALF_I[1] <= HALF_P[1] - 0.002


def _overlap(a, b) -> bool:
    return abs(a[0] - b[0]) < PRINT[0] - 0.001 and abs(a[1] - b[1]) < PRINT[1] - 0.001


def score(objects):
    """0.25 after one finished scan. 0.5 once that copy is on the up face of the sheet, within 8 mm.
    1.0 when the sheet is at least 20 mm out of the mouth, face up, tilted at most 12 degrees, slower
    than 2 cm/s, and both grippers are 5 cm clear. The runner holds 1.0 for SUCCESS_HOLD_S."""
    paper, card = objects["paper"], objects["card"]
    front = _S["printed"].get("front")
    back = _S["printed"].get("back")
    scanned = bool(_S["pending"]) or bool(_S["printed"])
    if not scanned:
        return 0.0
    on = {}
    if front is not None and _patch_on_sheet(objects["ink_front"], paper, front):
        on["front"] = front
    if back is not None and _patch_on_sheet(objects["ink_back"], paper, back):
        on["back"] = back
    if not on:
        return 0.25
    clear = _fully_out(paper) and _sheet_up(paper) and float(np.linalg.norm(paper.lin_vel)) < 0.02
    hands = _hands_clear(_CACHE["tcps"], [paper.pos, card.pos], 0.05)
    if clear and hands:
        return 1.0
    return 0.5


# Hands are read from the robots each step; score() only receives the object states.
_CACHE = {"tcps": [np.array([9.0, 9.0, 9.0]), np.array([9.0, 9.0, 9.0])]}


@on_step
def _cache_hands(env):
    base = env.unwrapped
    origin = labkit._np(base.scene.env_origins[0])
    _CACHE["tcps"] = _tcps(base, origin)
    _CACHE["origin"] = origin


def success(objects):
    return score(objects) == 1.0


def distance(objects):
    return 0.0 if success(objects) else 1.0 - score(objects)


# -- scripted demo ----------------------------------------------------------------------------------------------------
# The left hand picks up the card and places it on the glass. The right hand closes the lid with a side grip,
# and gravity finishes shutting it. The left hand presses Scan and both arms wait at home while the copier
# scans, prints and ejects the sheet.
TIP = 0.013  # TCP sits this far above the pad tips
CARD_WIDE, CARD_HOLD = -0.040, -0.020
SLAB_WIDE, SLAB_HOLD = -0.030, -0.003
LID_THICK = 0.014
PINCH_S = 0.20
PINCH_S_OUT = 0.30
PINCH_Y = -0.115
LID_RELEASE = math.radians(66.0)
TRAVEL = TOP + GLASS_Z + 0.040
_BASES = {"left": LEFT_BASE_POS, "right": RIGHT_BASE_POS}


def _dur(wps):
    return sum(w.duration for w in wps)


def home(side, duration, label="home", finger=FINGER_OPEN):
    pos, rot = ik.tcp_pose(np.array(READY, dtype=float), _BASES[side])
    return Waypoint(duration, tuple(float(v) for v in pos), finger, label, sidegrip.SideYaw(0.0, rot))


def _wait(arm, t, side):
    gap = t - _dur(arm)
    if gap <= 0.05:
        return
    if not (arm and arm[-1].label.startswith("home")):
        move = min(2.0, gap)
        arm.append(home(side, move))
        gap -= move
    if gap > 0.05:
        arm.append(home(side, gap, "home (wait)"))


def demo():
    # 1. left hand: card onto the glass, blue side up (the red side is copied).
    cx, cy = CARD_XY0
    gz = TOP + SEAT_Z
    lx, ly = SX + LEFT_XY[0], SY + LEFT_XY[1]
    left = [
        Waypoint(2.0, (cx, cy, TRAVEL), CARD_WIDE, "above the card"),
        Waypoint(1.2, (cx, cy, TOP + CARD_STAND_Z + CARD[2] / 2 + TIP - .005), CARD_WIDE, "down to the card"),
        Waypoint(0.6, (cx, cy, TOP + CARD_STAND_Z + CARD[2] / 2 + TIP - .005), CARD_HOLD, "pinch the card"),
        Waypoint(1.0, (cx, cy, TRAVEL), CARD_HOLD, "lift the card"),
        Waypoint(2.0, (lx, ly, TRAVEL), CARD_HOLD, "over the left of the glass"),
        Waypoint(1.4, (lx, ly, gz + TIP), CARD_HOLD, "set it on the left of the glass"),
        Waypoint(0.5, (lx, ly, gz + TIP), CARD_WIDE, "let go"),
        Waypoint(0.8, (lx, ly, TRAVEL), CARD_WIDE, "up"),
    ]
    # 2. Push the upper panel shut with the right hand.
    def swing_lid(arm, label):
        hinge = np.array([SX, SY, TOP]) + HINGE

        def pinch(q, s):
            n = np.array([math.sin(q), 0.0, math.cos(q)])
            u = np.array([-math.cos(q), 0.0, math.sin(q)])
            p = hinge + s * u + 0.5 * LID_THICK * n
            return (float(p[0]), PINCH_Y, float(p[2])), sidegrip.side(n, (0.0, 1.0, 0.0))

        pos, grip = pinch(LID_OPEN, PINCH_S_OUT)
        arm.append(Waypoint(2.8, pos, SLAB_WIDE, "beyond the free edge of the lid", grip))
        pos, grip = pinch(LID_OPEN, PINCH_S)
        arm.append(Waypoint(1.4, pos, SLAB_WIDE, "slide down the panel", grip))
        arm.append(Waypoint(0.6, pos, SLAB_HOLD, "pinch the panel", grip))
        for angle in np.linspace(LID_OPEN, LID_RELEASE, 9)[1:]:
            pos, grip = pinch(float(angle), PINCH_S)
            arm.append(Waypoint(0.5, pos, SLAB_HOLD, label, grip))
        arm.append(Waypoint(0.5, pos, SLAB_WIDE, "let go past the hand-off angle", grip))
        pos, grip = pinch(LID_RELEASE, PINCH_S_OUT)
        arm.append(Waypoint(0.8, pos, SLAB_WIDE, "back out past the free edge", grip))
        arm.append(home("right", 1.5))
        arm.append(home("right", 2.0, "home (gravity shuts the lid, the detent catches)"))

    right = [home("right", 1.0)]
    _wait(right, _dur(left), "right")
    swing_lid(right, "swing the lid shut")
    left.append(home("left", 1.5, finger=CARD_WIDE))
    _wait(left, _dur(right), "left")

    # 3. press Scan and wait for the lamp pass + print + eject.
    bx, by, bz = np.array([SX, SY, TOP]) + BUTTON_ANCHOR
    left += [
        Waypoint(1.2, (bx, by, bz + 0.04), FINGER_CLOSED, "above the scan button"),
        Waypoint(0.7, (bx, by, bz + TIP - 0.004), FINGER_CLOSED, "press Scan"),
        Waypoint(0.4, (bx, by, bz + TIP - 0.004), FINGER_CLOSED, "hold Scan"),
        Waypoint(0.6, (bx, by, bz + 0.04), FINGER_CLOSED, "off the button"),
        home("left", 12.0, "home (scan + print + eject)", FINGER_CLOSED),
    ]
    t_end = max(_dur(left), _dur(right)) + 1.5
    _wait(left, t_end, "left")
    _wait(right, t_end, "right")
    print(f"demo duration {t_end:.1f} s ({int(math.ceil(t_end * 20))} steps)")
    return {"left": left, "right": right}
