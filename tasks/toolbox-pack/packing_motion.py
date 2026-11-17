"""Contact-only scripted reference policy for the open toolbox task."""

import math

import numpy as np
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN
from yam_arena.urdf import quat_xyzw_to_matrix


def yaw_of(state):
    rotation = quat_xyzw_to_matrix(state.quat_xyzw)
    return math.atan2(rotation[1,0], rotation[0,0])


def wrap(yaw):
    return (yaw + math.pi / 2) % math.pi - math.pi / 2


def target(t, name):
    x, y, angle = t.SLOTS[name]
    return np.array([x, y, t.BOX_FLOOR]), math.radians(angle)


def park(arm, t):
    yield from arm.move((.16, arm.base[1], t.TOP + .20), finger=FINGER_OPEN, duration=1.1)


CARRY_MARGIN = .032
BOTTLES = ("lubricant", "spray")
# Asset-local targets measured on solid grip surfaces, plus total approach gap in metres.
GRASPS = {
    "hammer": ((.002, .059, .012), .035),
    "pliers": ((.039, 0, .013), .029),
    "wrench": ((.058, .001, .008), .034),
    "screwdriver": ((-.065, 0, .018), .039),
    "lubricant": ((0, 0, .225), .067),
    "spray": ((0, 0, .166), .070),
}


def bottle_sample(t, state):
    up = quat_xyzw_to_matrix(state.quat_xyzw)[:, 2]
    return {
        "tilt_deg": round(math.degrees(math.acos(float(np.clip(up[2], -1, 1)))), 2),
        "base_minus_floor_mm": round(float((state.low[2] - t.BOX_FLOOR) * 1000), 2),
        "linear_speed_m_s": round(float(np.linalg.norm(state.lin_vel)), 4),
        "angular_speed_rad_s": round(float(np.linalg.norm(state.ang_vel)), 4),
    }


def seat_bottle(ctx, t, name, arm, destination, yaw):
    """Use the live base pose to seat the bottle before opening the jaws."""
    goal = np.array(destination)
    pitch = 0.0
    for _ in range(5):
        state = ctx.states()[name]
        up = quat_xyzw_to_matrix(state.quat_xyzw)[:, 2]
        if up[2] < math.cos(math.radians(15)):
            return fail(ctx, name, "bottle tilted before release")
        height_error = state.low[2] - t.BOX_FLOOR
        lean = math.atan2(up[0], up[2])
        if abs(height_error) <= .0015 and abs(lean) <= math.radians(2):
            break
        goal[2] -= np.clip(height_error, -.002, .005)
        pitch -= np.clip(lean, -math.radians(4), math.radians(4))
        yield from arm.move(goal, yaw=yaw, pitch=pitch, duration=.55)
    yield from arm.wait(.8)
    state = ctx.states()[name]
    up = quat_xyzw_to_matrix(state.quat_xyzw)[:, 2]
    if (abs(state.low[2] - t.BOX_FLOOR) > .003
            or up[2] < math.cos(math.radians(5)) or not t.settled(state)):
        return fail(ctx, name, "bottle base did not settle upright on the floor")
    return goal, pitch


def carry(arm, start, destination, pick_yaw, place_yaw, lift_z, release=FINGER_OPEN,
          *, still_held=None, seat=None, observe_release=None, carry_pitch=0.0):
    yield from arm.move((*start[:2], lift_z), yaw=pick_yaw, pitch=carry_pitch, duration=1.4)
    yield from arm.move((*destination[:2], lift_z), yaw=place_yaw, pitch=carry_pitch, duration=2.7)
    if still_held is not None and not still_held():
        return False
    if seat is not None:
        yield from arm.move(destination + [0, 0, .022], yaw=place_yaw, duration=1.5)
    yield from arm.move(destination, yaw=place_yaw, duration=2.1 if seat else 1.5)
    pitch = 0.0
    if seat is not None:
        placement = yield from seat()
        if placement is False:
            return False
        destination, pitch = placement
        if observe_release is not None:
            observe_release("before_open")
    yield from arm.grip(release, duration=1.5 if seat else .55)
    if seat is not None:
        if observe_release is not None:
            observe_release("after_open")
        yield from arm.wait(.6)
        if observe_release is not None:
            observe_release("before_retreat")
    retreat_z = destination[2] + .025 if seat else lift_z
    yield from arm.move((*destination[:2], retreat_z), yaw=place_yaw, pitch=pitch, duration=1.2)
    return True


def fail(ctx, name, reason):
    ctx.failure = {"item": name, "reason": reason}
    print(f"Stopping at {name}: {reason}", flush=True)
    return False


def pick_place(ctx, t, name):
    arm = ctx.left if name in t.SUPPLIES else ctx.right
    bottle = name in BOTTLES
    local, gap = GRASPS.get(name, (None, -2 * FINGER_OPEN))
    state = ctx.states()[name]
    destination, place_yaw = target(t, name)
    rotation = quat_xyzw_to_matrix(state.quat_xyzw)
    start = np.array(state.pos) if local is None else state.origin + rotation @ (np.asarray(local) * t.SPEC[name].scale)
    if name == "screwdriver":
        start[0] += .008
    grip_offset = math.pi / 2 if name == "hammer" else 0.0
    pick_yaw = 0.0 if bottle else wrap(yaw_of(state) + grip_offset)
    end_yaw = pick_yaw if name in (*BOTTLES, "tape") else wrap(place_yaw + grip_offset)
    if bottle and rotation[2, 2] < math.cos(math.radians(25)):
        return fail(ctx, name, "bottle tipped before pickup")
    start[2] = max(start[2] + .006, t.TOP + .0135)
    yield from arm.move(start + [0, 0, .032 if bottle else .065], yaw=pick_yaw, finger=-gap / 2, duration=1.4)
    yield from arm.move(start, yaw=pick_yaw, duration=1.15)
    yield from arm.grip(FINGER_CLOSED, duration=1.0)
    yield from arm.wait(.2)

    clearance = start[2] - state.low[2]
    destination[:2] += (t.rz(end_yaw - pick_yaw) @ (start - state.pos))[:2]
    destination[2] += clearance + (.004 if bottle else .005)
    lift_z = max(start[2] + .035, t.BOX_RIM + CARRY_MARGIN + clearance)

    def still_held():
        held = ctx.states()[name]
        return held.low[2] >= t.BOX_RIM - .012 and np.linalg.norm(held.pos[:2] - destination[:2]) < .13

    def observe_release(phase):
        ctx.bottle_release.setdefault(name, {})[phase] = bottle_sample(t, ctx.states()[name])

    placed = yield from carry(arm, start, destination, pick_yaw, end_yaw, lift_z, release=-gap / 2,
                              still_held=still_held,
                              seat=(lambda: seat_bottle(ctx, t, name, arm, destination, end_yaw)) if bottle else None,
                              observe_release=observe_release if bottle else None,
                              carry_pitch=math.radians(-38) if bottle else 0.0)
    if not placed:
        if ctx.failure is None:
            fail(ctx, name, "item did not stay in the gripper through transfer")
        return False
    yield from arm.wait(.7 if bottle else .45)
    if bottle:
        observe_release("after_retreat")
    if not t.item_ok(name, ctx.states()[name]):
        return fail(ctx, name, "item did not settle in its assigned placement")
    yield from park(arm, t)
    return True


BRIDGE = np.array([.0785, 0, .052])
TEMPLE_GRIP_X = -.0473
TEMPLE_GRIP_Z = .054


def glasses_point(t, state, point):
    return state.origin + quat_xyzw_to_matrix(state.quat_xyzw) @ (t.SPEC["glasses"].scale * point)


def grip_glasses(arm, point, yaw, hover=.055):
    p = point + [0, 0, .006]
    yield from arm.move(p + [0, 0, hover], yaw=yaw, finger=-.014, duration=1.0)
    yield from arm.move(p, yaw=yaw, duration=.8)
    yield from arm.grip(FINGER_CLOSED, .45)


def bridge_yaw(arm, yaw):
    angle = wrap(yaw + math.pi / 2)
    if (arm.name == "right" and angle < 0) or (arm.name == "left" and angle > 0):
        angle += math.pi if arm.name == "right" else -math.pi
    return angle


def fold_temple(ctx, t, joint, arm, direction):
    # The other arm stays closed on the bridge throughout both folds.
    state = ctx.states()["glasses"]
    pivot = np.asarray(t.HINGES[joint])
    grip = np.array([TEMPLE_GRIP_X, direction * .0803, TEMPLE_GRIP_Z]) - pivot
    start_angle = state.joints[joint]
    yield from grip_glasses(arm, glasses_point(t, state, pivot + t.rz(start_angle) @ grip),
                           wrap(yaw_of(state) + start_angle))
    for angle in np.linspace(start_angle, direction * math.radians(89), 20)[1:]:
        state = ctx.states()["glasses"]
        point = glasses_point(t, state, pivot + t.rz(angle) @ grip) + [0, 0, .006]
        wrist_angle = np.clip(angle, -math.radians(62), math.radians(85))
        yield from arm.move(point, yaw=wrap(yaw_of(state) + wrist_angle), duration=.23)
    yield from arm.grip(-.014, .4)
    yield from arm.move(arm.tcp() + [0, 0, .035], yaw=arm._yaw, duration=.55)
    yield from park(arm, t)


def fold_and_place_glasses(ctx, t):
    holder, folder = ctx.left, ctx.right
    state = ctx.states()["glasses"]
    yaw = bridge_yaw(holder, yaw_of(state))
    bridge = glasses_point(t, state, BRIDGE)
    yield from grip_glasses(holder, bridge, yaw)
    grasp_height = bridge[2] + .006
    yield from holder.move((*bridge[:2], grasp_height + .065), yaw=yaw, duration=1.25)
    yield from holder.move((.260, -.025, grasp_height + .065), yaw=yaw, duration=1.4)
    state = ctx.states()["glasses"]
    if glasses_point(t, state, BRIDGE)[2] < bridge[2] + .040:
        return fail(ctx, "glasses", "bridge did not stay in the holding gripper")
    for joint, direction in ((t.TEMPLE_LEFT, 1), (t.TEMPLE_RIGHT, -1)):
        yield from fold_temple(ctx, t, joint, folder, direction)
        angle = ctx.states()["glasses"].joints[joint]
        if direction * angle < t.FOLD_MIN:
            return fail(ctx, "glasses", f"{joint} stopped at {math.degrees(angle):.1f} degrees")

    state = ctx.states()["glasses"]
    if not t.folded(state):
        return fail(ctx, "glasses", "both temples did not stay folded")
    destination, place_yaw = target(t, "glasses")
    tcp = holder.tcp()
    points = t.world_points("glasses", state)
    centre = (points.max(axis=0) + points.min(axis=0)) / 2
    end_yaw = wrap(place_yaw + math.pi / 2)
    destination[:2] += (t.rz(end_yaw - yaw) @ (tcp - centre))[:2]
    clearance = tcp[2] - points[:, 2].min()
    destination[2] += clearance + .005
    yield from carry(holder, tcp, destination, yaw, end_yaw, max(tcp[2], t.BOX_RIM + clearance + CARRY_MARGIN),
                     release=-.020)
    yield from holder.wait(.45)
    if not t.item_ok("glasses", ctx.states()["glasses"]):
        return fail(ctx, "glasses", "glasses did not settle folded inside the toolbox")
    yield from park(holder, t)


def pack(ctx, t):
    for name in t.ORDER[:-1]:
        print(f"Packing {name}", flush=True)
        if not (yield from pick_place(ctx, t, name)):
            return
        yield from ctx.right.wait(.45)
    states = ctx.states()
    missing = [name for name in t.ORDER[:-1] if not t.item_ok(name, states[name])]
    if missing:
        return fail(ctx, missing[0], "item moved out of its placement before the goggles")

    print("Lifting goggles and folding both temples in the air", flush=True)
    yield from fold_and_place_glasses(ctx, t)
