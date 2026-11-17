"""propkit: USD props (fixed or free) for yam_arena task files. Hooks yam_arena.scene like cablekit does."""

from __future__ import annotations

import importlib.abc
import importlib.util
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from yam_arena import task_file
from yam_arena.task_file import Box, ObjectState

__all__ = ["Prop", "prop", "rot_matrix", "install"]


@dataclass(frozen=True)
class Prop(Box):
    """``pos``/``size`` are the world bbox centre and the scaled bbox size (in the prop's own frame, turned by yaw)."""

    usd: str = ""
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    fixed: bool = False
    bmin: tuple = (0.0, 0.0, 0.0)   # asset bbox in the asset frame at scale 1
    bmax: tuple = (0.0, 0.0, 0.0)

    joints: tuple[str, ...] = ()

    def _check(self):
        super()._check()
        if not Path(self.usd).is_file():
            raise FileNotFoundError(self.usd)

    @property
    def spawn_origin(self) -> tuple[float, float, float]:
        """Undo the scaled, yawed offset from the asset origin to its bounding-box centre."""
        cx, cy, cz = (np.asarray(self.bmin) + self.bmax) * np.asarray(self.scale) / 2
        ct, st = math.cos(self.yaw), math.sin(self.yaw)
        return (float(self.pos[0] - ct * cx + st * cy),
                float(self.pos[1] - st * cx - ct * cy),
                float(self.pos[2] - cz))

    @property
    def free(self) -> bool:
        return not self.fixed


def prop(name, usd, bounds, xy, *, table_top, yaw=0.0, scale=1.0, mass=None, fixed=False, **kw) -> Prop:
    """``xy`` is where the asset's ORIGIN goes. ``bounds`` is (min, max) of the asset bbox from inspect_asset."""
    bmin, bmax = np.array(bounds[0], float), np.array(bounds[1], float)
    scale = tuple(map(float, np.broadcast_to(scale, (3,))))
    c = (bmin + bmax) / 2 * scale
    size = tuple(float(v) for v in (bmax - bmin) * scale)
    ct, st = math.cos(yaw), math.sin(yaw)
    cx, cy = xy[0] + ct * c[0] - st * c[1], xy[1] + st * c[0] + ct * c[1]
    if mass is not None:
        kw["mass"] = mass
    return Prop(name=name, pos=(cx, cy, table_top + 0.001 + size[2] / 2), size=size, yaw=yaw,
                color=(0.6, 0.6, 0.6), usd=str(usd), scale=scale, fixed=fixed,
                bmin=tuple(bmin), bmax=tuple(bmax), **kw)


def rot_matrix(q) -> np.ndarray:
    """Rotation matrix from a quaternion (x, y, z, w)."""
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


# -- Isaac Sim side ------------------------------------------------------------------------------------------

_SPECS: dict[str, Prop] = {}


def _arena_object(spec: Prop):
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.managers import EventTermCfg, SceneEntityCfg
    from isaaclab_physx.sim.schemas import PhysxArticulationCfg
    from isaaclab_arena.terms.events import reset_articulation_pose_and_joints
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.utils.pose import Pose

    class PropObject(Object):
        def _generate_articulation_cfg(self):
            cfg = super()._generate_articulation_cfg()
            cfg.init_state.joint_pos = dict.fromkeys(spec.joints, 0.0)
            cfg.init_state.joint_vel = dict.fromkeys(spec.joints, 0.0)
            cfg.actuators = {"temples": ImplicitActuatorCfg(
                joint_names_expr=list(spec.joints), stiffness=0.0, damping=0.001,
                friction=0.0015, dynamic_friction=0.0015)}
            return cfg

        def _build_reset_event(self):
            if not spec.joints:
                return super()._build_reset_event()
            return EventTermCfg(func=reset_articulation_pose_and_joints, mode="reset",
                                params={"asset_cfg": SceneEntityCfg(self.name), "pose": self.initial_pose})

    s = spec.scale
    addons = {"collision_props": sim_utils.CollisionPropertiesCfg(contact_offset=0.001, rest_offset=0.0)}
    if spec.joints:
        addons["articulation_props"] = {
            "/Geometry/frame_01_obj_00": [PhysxArticulationCfg(
                enabled_self_collisions=False, solver_position_iteration_count=16,
                solver_velocity_iteration_count=4)]}
    else:
        addons["rigid_props"] = sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=spec.fixed, max_depenetration_velocity=1.0,
            solver_position_iteration_count=16, solver_velocity_iteration_count=4)
        if spec.mass is not None and not spec.fixed:
            addons["mass_props"] = sim_utils.MassPropertiesCfg(mass=spec.mass)

    return PropObject(
        name=spec.name,
        object_type=ObjectType.ARTICULATION if spec.joints else ObjectType.RIGID,
        usd_path=spec.usd, scale=s, spawn_cfg_addon=addons,
        initial_pose=Pose(position_xyz=spec.spawn_origin,
                          rotation_xyzw=(0.0, 0.0, math.sin(spec.yaw / 2), math.cos(spec.yaw / 2))),
    )


@dataclass(frozen=True)
class PropState(ObjectState):
    origin: np.ndarray
    joints: dict[str, float]


def _prop_state(env, spec: Prop) -> PropState:
    """Live state from the sim: pos = bbox centre, low/high = world AABB."""
    base = env.unwrapped
    art = base.scene[spec.name]
    origin = base.scene.env_origins[0]

    def t(x):
        return getattr(x, "torch", x)

    to_np = lambda a: a.detach().cpu().numpy().astype(np.float64)  # noqa: E731
    d = art.data
    pose = t(d.root_link_pose_w)[0]
    p = to_np(pose[:3] - origin)
    q = to_np(pose[3:])
    R = rot_matrix(q)
    bmin, bmax = np.array(spec.bmin), np.array(spec.bmax)
    centre = R @ ((bmin + bmax) / 2 * spec.scale) + p
    half = np.abs(R) @ ((bmax - bmin) / 2 * spec.scale)
    joints = dict(zip(art.joint_names, to_np(t(d.joint_pos)[0]), strict=True)) if spec.joints else {}
    velocity = to_np(t(d.root_vel_w)[0])
    return PropState(pos=centre, quat_xyzw=q, lin_vel=velocity[:3], ang_vel=velocity[3:],
                     low=centre - half, high=centre + half, origin=p, joints=joints)


def _patch_scene(scene) -> None:
    if getattr(scene, "_propkit_patched", False):
        return
    orig_task_object, orig_states = scene.task_object, scene.object_states

    def task_object(spec):
        if isinstance(spec, Prop):
            _SPECS[spec.name] = spec
            return _arena_object(spec)
        return orig_task_object(spec)

    def object_states(env, names):
        plain = [n for n in names if n not in _SPECS]
        out = orig_states(env, plain) if plain else {}
        for n in names:
            if n in _SPECS:
                out[n] = _prop_state(env, _SPECS[n])
        return {n: out[n] for n in names}

    scene.task_object = task_object
    scene.object_states = object_states
    scene._propkit_patched = True


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname != "yam_arena.scene":
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None
        orig_exec = spec.loader.exec_module

        def exec_module(module):
            orig_exec(module)
            _patch_scene(module)

        spec.loader.exec_module = exec_module
        return spec


def install_trial_reset():
    from yam_arena.inspect_embodiment import YamArenaEmbodiment

    base_reset = YamArenaEmbodiment.reset
    if getattr(base_reset, "_toolbox_pack_reset", False):
        return

    def reset(self, *args, **kwargs):
        task = getattr(self, "task", None)
        callback = getattr(getattr(task, "module", None), "reset_trial", None)
        if callback is not None:
            callback()
        return base_reset(self, *args, **kwargs)

    reset._toolbox_pack_reset = True
    YamArenaEmbodiment.reset = reset


def install() -> None:
    if "yam_arena.scene" in sys.modules:
        _patch_scene(sys.modules["yam_arena.scene"])
    elif not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())


install()
assert task_file