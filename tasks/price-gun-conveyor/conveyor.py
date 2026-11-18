"""The slow belt conveyor the boxes ride on: its geometry, its parts, and what makes the belt move.

A low flat-belt conveyor lies across the bench in front of the arms and runs toward +y (to the arms'
left). Steel carrier plates ride the belt and are driven exactly along it (pose and velocity written
before every physics step, like a servo-driven line); the boxes rest on the carriers and are carried
by friction, so they can be held back, pushed or knocked off. Thin marks drawn on the belt (cross
stripes and an outline round each box's place) are slid along with the belt's travel.
Frame: metres, x away from the arm bases, y left, z up; TOP is the bench.
"""
from __future__ import annotations

import math

from simkit import Part, Piece

TOP = 0.7475

SPEED = 0.020                        # belt speed, m/s, toward +y
RAMP_S = 0.50                        # seconds to reach SPEED from a standing start
X = 0.290                            # the belt's centre line
WIDTH = 0.160
Y0, Y1 = -0.720, 0.600               # the belt's flat top: upstream end, and the face of the end stop
DECK = 0.020                         # the belt's surface above the bench
CARRIER_T = 0.008                    # carrier plate thickness
CARRIER_TOP = DECK + CARRIER_T       # top of a carrier above the bench
RAIL = 0.008                         # side rails, just below the belt's surface
STOP = (0.006, 0.040)                # end stop plate: thickness, height above the belt
LENGTH = Y1 - Y0
X_EDGE = (X - WIDTH / 2 - RAIL, X + WIDTH / 2 + RAIL)      # the conveyor's footprint across the bench
CARRIER_MASS = 50.0                  # kg; heavy, so contacts barely disturb a plate before the next write

BELT_COL = (0.12, 0.13, 0.14)        # charcoal PVC belt
LINE_COL = (0.30, 0.32, 0.34)
FRAME_COL = (0.62, 0.64, 0.66)       # anodised aluminium
CARRIER_COL = (0.55, 0.57, 0.60)     # steel carrier plates
MARK_COL = (0.93, 0.93, 0.90)
DRIVE_COL = (0.85, 0.47, 0.08)

BELT_NAME = "belt"
FRAME_NAME = "conveyor"
STRIPES_NAME = "belt_stripes"
PITCH = 0.100                        # cross stripes on the belt, this far apart
N_STRIPES = int((LENGTH - 0.020) / PITCH)
MARK_GAP = 0.004                     # a box's outline on the belt stands this far off its footprint
MARK_W = 0.002
THIN = 0.0004
CARRIER_MARGIN = 0.010               # carrier overhang past each box's footprint


def belt_object() -> Part:
    """The belt's flat top run (fixed, visual under the carriers)."""
    z = DECK / 2 + 0.0002
    lines = tuple(Piece("box", (0.003, LENGTH, THIN), (sx * (WIDTH / 2 - 0.006), 0.0, z), color=LINE_COL, collide=False)
                  for sx in (-1, 1))                         # tracking lines along its edges
    return Part(BELT_NAME, pos=(X, (Y0 + Y1) / 2, TOP + DECK / 2), size=(WIDTH, LENGTH, DECK), exact_z=True,
                color=BELT_COL, friction=(0.4, 0.3), restitution=0.0, contact_offset=0.002, pieces=lines,
                collision="none")


def frame_object() -> Part:
    """The end stop, the upstream roller's cover and the drive (all fixed).

    The side rails are separate Parts (``rail_objects``): putting them on this body as pieces would
    make one convex hull that filled the space between them and blocked the boxes on the belt.
    """
    t, h = STOP
    root = (X, Y1 + t / 2, TOP + (DECK + h) / 2)             # the end stop plate, standing on the bench

    def at(x, y, z):
        return (x - root[0], y - root[1], TOP + z - root[2])

    pieces = [
        Piece("cylinder", (DECK / 2, WIDTH, DECK / 2), at(X, Y0, DECK / 2), axis="X", color=BELT_COL, collide=False),
        Piece("box", (0.050, 0.090, 0.050), at(X_EDGE[1] + 0.025, Y0 + 0.060, 0.025), color=DRIVE_COL),
    ]
    return Part(FRAME_NAME, pos=root, size=(WIDTH + 2 * RAIL, t, DECK + h), exact_z=True, color=FRAME_COL,
                friction=(0.4, 0.3), restitution=0.0, contact_offset=0.002, pieces=tuple(pieces))


def rail_objects() -> list[Part]:
    """One fixed rail along each side of the belt, well below its surface."""
    rail_h = DECK - 0.010
    return [Part(f"rail_{side}", pos=(X + sx * (WIDTH + RAIL) / 2, (Y0 + Y1) / 2, TOP + rail_h / 2),
                 size=(RAIL, LENGTH, rail_h), exact_z=True, color=FRAME_COL,
                 friction=(0.4, 0.3), restitution=0.0, contact_offset=0.002)
            for side, sx in (("near", -1), ("far", 1))]


def stripes_object() -> Part:
    """Cross stripes printed on the belt (visual only). The whole set is slid along with the belt and
    wrapped every PITCH, so a stripe comes up at the upstream end as one goes under at the other."""
    return Part(STRIPES_NAME, pos=stripes_pos(0.0), size=(WIDTH - 0.030, 0.003, THIN), exact_z=True, color=LINE_COL,
                collision="none",
                pieces=tuple(Piece("box", (WIDTH - 0.030, 0.003, THIN), (0.0, k * PITCH, 0.0), color=LINE_COL,
                                   collide=False) for k in range(1, N_STRIPES)))


def stripes_pos(travel: float):
    return (X, Y0 + 0.010 + math.fmod(travel, PITCH), TOP + DECK + 0.0003)


def outline_name(box: str) -> str:
    return f"outline_{box}"


def outline_object(box: str, size, centre) -> Part:
    """A thin outline printed on the belt round a box's place (visual only), moved with the belt."""
    lx, ly = size[0] + 2 * MARK_GAP, size[1] + 2 * MARK_GAP
    bars = [Piece("box", (lx + MARK_W, MARK_W, THIN), (0.0, sy * ly / 2, 0.0), color=MARK_COL, collide=False)
            for sy in (-1, 1)]
    bars += [Piece("box", (MARK_W, ly + MARK_W, THIN), (sx * lx / 2, 0.0, 0.0), color=MARK_COL, collide=False)
             for sx in (-1, 1)]
    # the root is a small dot in the middle, under the box
    return Part(outline_name(box), pos=outline_pos(centre, 0.0), size=(0.004, 0.004, THIN), exact_z=True,
                color=MARK_COL, collision="none", pieces=tuple(bars))


def outline_pos(centre, travel: float):
    return (float(centre[0]), float(centre[1]) + travel, TOP + CARRIER_TOP + 0.0003)


def carrier_name(box: str) -> str:
    return f"carrier_{box}"


def carrier_size(box_size) -> tuple[float, float, float]:
    return (box_size[0] + 2 * CARRIER_MARGIN, box_size[1] + 2 * CARRIER_MARGIN, CARRIER_T)


def carrier_pos(centre, travel: float = 0.0):
    """World centre of a carrier whose box started at ``centre``, ``travel`` metres on."""
    return (float(centre[0]), float(centre[1]) + travel, TOP + DECK + CARRIER_T / 2)


def carrier_object(box: str, box_size, centre) -> Part:
    """A heavy steel plate under one box; ``install_drive`` writes its pose every physics step."""
    return Part(carrier_name(box), pos=carrier_pos(centre, 0.0), size=carrier_size(box_size), exact_z=True,
                mass=CARRIER_MASS, color=CARRIER_COL, friction=(0.9, 0.8), restitution=0.0,
                contact_offset=0.002)


def disable_gravity(obj):
    """Turn gravity off on a spawned rigid body (carriers are driven; they must not sag)."""
    seen = set()
    for cfg in (obj.spawner_cfg, getattr(obj.object_cfg, "spawn", None)):
        if cfg is None or id(cfg) in seen:
            continue
        seen.add(id(cfg))
        props = getattr(cfg, "rigid_props", None)
        if props is not None and hasattr(props, "disable_gravity"):
            props.disable_gravity = True
    return obj


def over_conveyor(point) -> bool:
    """Whether a point is above the conveyor's footprint on the bench."""
    return X_EDGE[0] <= point[0] <= X_EDGE[1] and Y0 - DECK / 2 <= point[1] <= Y1 + STOP[0]


def speed_at(t: float) -> float:
    """Belt speed at time ``t`` (s from the first control step): ramps up, then holds SPEED."""
    if t <= 0.0:
        return 0.0
    if t < RAMP_S:
        return SPEED * (t / RAMP_S)
    return SPEED


def travel_at(t: float) -> float:
    """How far a mark on the belt has ridden by time ``t`` (s from the first control step)."""
    if t <= 0.0:
        return 0.0
    if t < RAMP_S:
        return 0.5 * SPEED * t * t / RAMP_S
    return SPEED * (t - 0.5 * RAMP_S)


def install_drive(env, centres: dict[str, tuple[float, float]], hz: float):
    """Drive every carrier along the belt before each physics step; clock advances with control steps.

    ``centres`` maps box name -> (x, y) start on the belt. Returns ``(clock, drive_once)`` where
    ``clock`` is a mutable ``{"t": seconds}`` and ``drive_once`` writes the carriers at the current ``t``.
    """
    import torch

    base = env.unwrapped
    sim = base.sim
    dt = float(sim.get_physics_dt()) if hasattr(sim, "get_physics_dt") else float(base.physics_dt)
    clock = {"t": 0.0, "running": False}
    origin = base.scene.env_origins
    assets = {name: base.scene[carrier_name(name)] for name in centres}

    def drive_once():
        t = clock["t"]
        travel = travel_at(t)
        speed = speed_at(t)
        for name, asset in assets.items():
            pos = carrier_pos(centres[name], travel)
            pose = asset.data.default_root_pose.torch.clone()
            pose[:, :3] = torch.tensor(pos, dtype=pose.dtype, device=pose.device) + origin
            twist = torch.zeros((pose.shape[0], 6), dtype=pose.dtype, device=pose.device)
            twist[:, 1] = float(speed)
            asset.write_root_pose_to_sim_index(root_pose=pose)
            asset.write_root_velocity_to_sim_index(root_velocity=twist)

    if not getattr(sim, "_price_gun_conveyor_drive", False):
        step = sim.step

        def step_hook(*a, **k):
            drive_once()
            out = step(*a, **k)
            if clock["running"]:
                clock["t"] += dt
            return out

        sim.step = step_hook
        sim._price_gun_conveyor_drive = True

    # Hold carriers still through the settling steps before control starts.
    drive_once()
    return clock, drive_once
