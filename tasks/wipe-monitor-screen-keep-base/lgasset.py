"""LG 27GX790A from the LG SimReady dataset, kept as a three-link articulation.

The package is CC-BY-4.0. Files download once into ``~/.cache/simready`` (``SIMREADY_CACHE``
overrides) at the pinned revision; nothing large is committed here.

The authored file turns gravity off on the head and the hinge, and its tilt drive has no
stiffness, so the head would sag once gravity is on. Spawn enables gravity, leaves the
height slide on its lower stop, and adds friction so the head stays at the shipped angle
until something pushes it.
"""

from __future__ import annotations

import hashlib
import importlib.abc
import importlib.util
import json
import math
import os
import sys
import urllib.request
from dataclasses import dataclass

from yam_arena.task_file import Box

__all__ = ["Cloth", "Monitor", "fetch", "install", "on_reset", "on_step"]

REPO = "LGElectronics/simready-assets"
REVISION = "e965976d90f025237ad9ea8881f5f82755f26d9d"
HF = "https://huggingface.co"
CACHE_ROOT = os.environ.get("SIMREADY_CACHE", "~/.cache/simready")
HEIGHT_TARGET = -0.110
TILT_HOLD_NM = 8.0
SLIDE_HOLD_N = 40.0

_SPECS: dict[str, "Monitor"] = {}
_SMUDGES: list[tuple[int, str]] = []
_SPAWN: dict = {}
_STEP: list = []
_RESET: list = []
# Local-frame bounds of each rigid-body link, measured from the USD. ArenaWorld will not
# build a box for an articulation, because the hinge and the panel move relative to the foot.
_LINK_BOXES: dict[str, dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]]] = {}


@dataclass(frozen=True)
class Monitor(Box):
    """The monitor USD. ``pos`` is the asset origin (bottom of the foot). ``quat_xyzw`` is its orientation."""

    package: str = "monitor_27gx790a"
    usd: str = "monitor_27gx790a.usd"
    quat_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    marks: tuple = ()

    def _check(self) -> None:
        super()._check()
        if abs(math.fsum(v * v for v in self.quat_xyzw) - 1.0) > 1e-6:
            raise ValueError(f"{self.name}.quat_xyzw must be a unit quaternion")


@dataclass(frozen=True)
class Cloth(Box):
    """A PhysX surface cloth. ``pos`` is the centre of the sheet; ``size`` is (width, depth, thickness)."""


def _get(url: str) -> bytes:
    headers = {"User-Agent": "robocurve-benchmark-task"}
    if os.environ.get("HF_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['HF_TOKEN']}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=180) as response:
        return response.read()


def _wanted(path: str, package: str) -> bool:
    if path == f"{package}/.metadata/LICENSE.txt":
        return True
    if not path.startswith(f"{package}/simready_usd/") or "/.thumbs/" in path:
        return False
    return not path.endswith(".png") or "/textures/" in path


def fetch(package: str, usd: str):
    from pathlib import Path

    root = Path(CACHE_ROOT).expanduser() / REPO / REVISION
    target, done = root / package / "simready_usd" / usd, root / package / ".complete"
    if done.is_file() and target.is_file():
        return target
    listing = json.loads(_get(f"{HF}/api/datasets/{REPO}/tree/{REVISION}/{package}?recursive=true"))
    files = [e for e in listing if e.get("type") == "file" and _wanted(e["path"], package)]
    if not files:
        raise FileNotFoundError(f"{REPO}@{REVISION[:8]} has no package {package!r}")
    for entry in files:
        dest = root / entry["path"]
        if dest.is_file() and dest.stat().st_size == entry.get("size"):
            continue
        data = _get(f"{HF}/datasets/{REPO}/resolve/{REVISION}/{entry['path']}")
        sha = (entry.get("lfs") or {}).get("oid")
        if sha and hashlib.sha256(data).hexdigest() != sha:
            raise OSError(f"checksum mismatch for {entry['path']}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        tmp.write_bytes(data)
        tmp.replace(dest)
    if not target.is_file():
        raise FileNotFoundError(f"{package}: no {usd} in simready_usd/")
    done.write_text(REVISION)
    return target


def _axis_friction(prim, axis: str, static: float, dynamic: float) -> None:
    from pxr import Sdf

    if "PhysxJointAxisAPI" not in prim.GetAppliedSchemas():
        prim.ApplyAPI("PhysxJointAxisAPI", axis)
    for name, value in (("staticFrictionEffort", static), ("dynamicFrictionEffort", dynamic)):
        attr = prim.GetAttribute(f"physxJointAxis:{axis}:{name}")
        if not attr:
            attr = prim.CreateAttribute(f"physxJointAxis:{axis}:{name}", Sdf.ValueTypeNames.Float)
        attr.Set(float(value))


def _prepare(prim, spec: Monitor) -> None:
    """Gravity on, head held up by friction, height slide resting on its lower stop, marks on the glass."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    for body in Usd.PrimRange(prim):
        attr = body.GetAttribute("physxRigidBody:disableGravity")
        if attr:
            attr.Set(False)
    stage = prim.GetStage()
    monitor = None
    for body in Usd.PrimRange(prim):
        if body.GetName() == "monitor" and body.HasAPI(UsdPhysics.RigidBodyAPI):
            monitor = body
        if body.GetName() == "physics_mat_plastic":
            rest = body.GetAttribute("physics:restitution")
            if rest:
                rest.Set(0.05)
        if not body.IsA(UsdPhysics.Joint):
            continue
        if body.GetName() == "height_slide":
            drive = UsdPhysics.DriveAPI.Apply(body, "linear")
            drive.CreateTargetPositionAttr().Set(HEIGHT_TARGET)
            drive.CreateStiffnessAttr().Set(200.0)
            drive.CreateDampingAttr().Set(400.0)
            _axis_friction(body, "linear", SLIDE_HOLD_N, SLIDE_HOLD_N * 0.7)
        elif body.GetName() == "tilt_hinge":
            drive = UsdPhysics.DriveAPI.Apply(body, "angular")
            drive.CreateTargetPositionAttr().Set(0.0)
            drive.CreateStiffnessAttr().Set(0.0)
            drive.CreateDampingAttr().Set(2.0)
            _axis_friction(body, "angular", TILT_HOLD_NM, TILT_HOLD_NM * 0.75)
    if monitor is None:
        raise RuntimeError(f"{spec.name}: the USD has no monitor rigid body")
    _LINK_BOXES[spec.name] = _measure_links(prim)
    color = Gf.Vec3f(0.42, 0.30, 0.16)
    for i, local in enumerate(spec.marks):
        path = f"{monitor.GetPath()}/fingerprint_{i}"
        cube = UsdGeom.Cube.Define(stage, path)
        cube.CreateSizeAttr(1.0)
        cube.AddTranslateOp().Set(Gf.Vec3d(*local))
        cube.AddScaleOp().Set(Gf.Vec3f(0.0012, 0.024, 0.024))
        cube.CreateDisplayColorAttr([color])
        _SMUDGES.append((i, str(path)))


def _spawn_raw(prim_path: str, cfg, *args, **kwargs):
    from isaaclab.sim.spawners.from_files import spawn_from_usd

    prim = spawn_from_usd.__wrapped__(prim_path, cfg, *args, **kwargs)
    leaf = prim_path.rstrip("/").rsplit("/", 1)[-1]
    _prepare(prim, _SPECS.get(leaf) or next(iter(_SPECS.values())))
    return prim


def _spawner():
    if "fn" not in _SPAWN:
        from isaaclab.sim.utils import clone

        _SPAWN["fn"] = clone(_spawn_raw)
    return _SPAWN["fn"]


def _arena_object(spec: Monitor):
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.utils.pose import Pose

    usd = fetch(spec.package, spec.usd)
    _SPECS[spec.name] = spec
    obj = Object(
        name=spec.name,
        usd_path=str(usd),
        object_type=ObjectType.ARTICULATION,
        initial_pose=Pose(position_xyz=tuple(float(v) for v in spec.pos), rotation_xyzw=tuple(spec.quat_xyzw)),
        spawn_cfg_addon={"func": _spawner()},
    )
    obj.object_cfg.init_state.joint_pos = {"height_slide": HEIGHT_TARGET, "tilt_hinge": 0.0}
    return obj


def hide_marks(clean) -> None:
    """Hide a fingerprint once it has been wiped. Paths were recorded at spawn."""
    if not _SMUDGES or not any(clean):
        return
    from pxr import Gf, UsdGeom
    import isaaclab.sim as sim_utils

    stage = sim_utils.get_current_stage()
    for index, path in _SMUDGES:
        prim = stage.GetPrimAtPath(path)
        if not prim:
            continue
        imageable = UsdGeom.Imageable(prim)
        xformable = UsdGeom.Xformable(prim)
        scale = next((op for op in xformable.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeScale), None)
        if clean[index]:
            imageable.MakeInvisible()
            if scale is not None:
                scale.Set(Gf.Vec3f(0.0, 0.0, 0.0))
        else:
            imageable.MakeVisible()
            if scale is not None:
                scale.Set(Gf.Vec3f(0.0012, 0.024, 0.024))


def seat_joints(environment) -> None:
    """Put the height slide on its lower stop and the tilt at zero after every reset."""
    import torch

    base = environment.unwrapped
    for name in _SPECS:
        asset = base.scene.articulations[name] if name in getattr(base.scene, "articulations", {}) else base.scene[name]
        joint = asset.data.joint_pos
        joint = joint.torch if hasattr(joint, "torch") else joint
        q = joint.clone()
        for i, joint_name in enumerate(asset.joint_names):
            if "height" in joint_name:
                q[0, i] = HEIGHT_TARGET
            elif "tilt" in joint_name:
                q[0, i] = 0.0
        zeros = torch.zeros_like(q)
        asset.write_joint_state_to_sim_index(position=q, velocity=zeros)
        asset.set_joint_position_target_index(target=q)


def on_step(fn):
    _STEP.append(fn)
    return fn


def on_reset(fn):
    _RESET.append(fn)
    return fn


def _measure_links(prim):
    """Axis-aligned bounds of each rigid-body link, in that link's own frame."""
    from pxr import Usd, UsdGeom, UsdPhysics

    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    boxes = {}
    for body in Usd.PrimRange(prim):
        if not body.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        span = cache.ComputeLocalBound(body).ComputeAlignedRange()
        if span.IsEmpty():
            continue
        boxes[body.GetName()] = (
            tuple(float(v) for v in span.GetMin()),
            tuple(float(v) for v in span.GetMax()),
        )
    if not boxes:
        raise RuntimeError(f"{prim.GetPath()}: no rigid-body link has geometry")
    return boxes


def _rotate(xyzw, vector):
    x, y, z, w = (float(v) for v in xyzw)
    vx, vy, vz = (float(v) for v in vector)
    return (
        (1 - 2 * (y * y + z * z)) * vx + 2 * (x * y - w * z) * vy + 2 * (x * z + w * y) * vz,
        2 * (x * y + w * z) * vx + (1 - 2 * (x * x + z * z)) * vy + 2 * (y * z - w * x) * vz,
        2 * (x * z - w * y) * vx + 2 * (y * z + w * x) * vy + (1 - 2 * (x * x + y * y)) * vz,
    )


def _link_bounds(asset, boxes):
    """World-frame bounds from each link's own box at its current pose."""
    import numpy as np

    poses = asset.data.body_link_pose_w.torch[0].detach().cpu().numpy().astype(np.float64)
    low, high = np.full(3, np.inf), np.full(3, -np.inf)
    found = False
    for index, name in enumerate(asset.body_names):
        box = boxes.get(name.rsplit("/", 1)[-1])
        if box is None:
            continue
        (x0, y0, z0), (x1, y1, z1) = box
        position, quat = poses[index, :3], poses[index, 3:7]
        for corner in (
            (x0, y0, z0), (x0, y0, z1), (x0, y1, z0), (x0, y1, z1),
            (x1, y0, z0), (x1, y0, z1), (x1, y1, z0), (x1, y1, z1),
        ):
            world = position + np.asarray(_rotate(quat, corner))
            low, high = np.minimum(low, world), np.maximum(high, world)
        found = True
    if not found:
        raise RuntimeError(f"none of {list(asset.body_names)} match the measured links {sorted(boxes)}")
    return low, high


def _articulation_states(env, names):
    """Object states for articulations, which ArenaWorld will not bound."""
    import numpy as np

    from yam_arena.task_file import ObjectState

    base = env.unwrapped
    world = base.arena_world
    origin = base.scene.env_origins[0].detach().cpu().numpy().astype(np.float64)

    def arr(value):
        return value.detach().cpu().numpy().astype(np.float64)

    out = {}
    for name in names:
        asset = base.scene.articulations[name]
        pose = arr(world.get_pose_w(name)[0])
        low, high = _link_bounds(asset, _LINK_BOXES[name])
        low, high = low - origin, high - origin
        out[name] = ObjectState(
            pos=(low + high) / 2,
            quat_xyzw=pose[3:],
            lin_vel=arr(world.get_root_linear_velocity_w(name)[0]),
            ang_vel=arr(world.get_root_angular_velocity_w(name)[0]),
            low=low,
            high=high,
        )
    return out


def _cloth_object(spec: Cloth):
    """A soft flat sheet. The left hand bunches it; a stiff spawn ridge only fell flat."""
    import isaaclab.sim as sim_utils
    from isaaclab_arena.assets.deformable_object import DeformableObject
    from isaaclab_arena.utils.pose import Pose
    from isaaclab_physx.sim.schemas import PhysxCollisionCfg, PhysxDeformableBodyPropertiesCfg
    from isaaclab_physx.sim.spawners.materials import PhysxSurfaceDeformableBodyMaterialCfg
    from yam_arena.station import linear_rgb

    width, depth, thick = (float(v) for v in spec.size)
    return DeformableObject(
        name=spec.name,
        spawner_cfg=sim_utils.MeshRectangleCfg(
            size=(width, depth),
            edge_refinement=8,
            deformable_props=PhysxDeformableBodyPropertiesCfg(linear_damping=0.2),
            collision_props=[PhysxCollisionCfg(rest_offset=0.002, contact_offset=0.008)],
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=linear_rgb(spec.color), roughness=0.85),
            physics_material=PhysxSurfaceDeformableBodyMaterialCfg(
                density=150.0,
                surface_thickness=thick,
                youngs_modulus=3.0e4,
                poissons_ratio=0.25,
                surface_bend_stiffness=80.0,
                elasticity_damping=0.2,
                bend_damping=0.1,
                static_friction=1.2,
                dynamic_friction=1.0,
            ),
        ),
        initial_pose=Pose(position_xyz=tuple(float(v) for v in spec.pos), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)),
    )


def _deformable_states(env, names):
    """Object states for a surface cloth. ArenaWorld will not return a root pose for one."""
    import numpy as np

    from yam_arena.task_file import ObjectState

    base = env.unwrapped
    world = base.arena_world
    origin = base.scene.env_origins[0].detach().cpu().numpy().astype(np.float64)

    def arr(value):
        return value.detach().cpu().numpy().astype(np.float64)

    out = {}
    for name in names:
        nodes = arr(world.get_nodal_positions_w(name)[0])
        centre = nodes.mean(axis=0)
        box = world.get_aabb_w(name)
        out[name] = ObjectState(
            pos=centre - origin,
            quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
            lin_vel=arr(world.get_mean_linear_velocity_w(name)[0]),
            ang_vel=np.zeros(3),
            low=arr(box.min_point[0]) - origin,
            high=arr(box.max_point[0]) - origin,
        )
    return out


def _patch_scene(scene) -> None:
    if getattr(scene, "_lg_monitor", False):
        return
    task_object = scene.task_object
    original_states = scene.object_states

    def object_states(env, names):
        live = env.unwrapped.scene
        articulations = getattr(live, "articulations", {})
        soft = getattr(live, "deformable_objects", {})
        plain = [name for name in names if name not in articulations and name not in soft]
        out = original_states(env, plain) if plain else {}
        hinged = [name for name in names if name in articulations]
        floppy = [name for name in names if name in soft]
        if hinged:
            out.update(_articulation_states(env, hinged))
        if floppy:
            out.update(_deformable_states(env, floppy))
        return out

    scene.object_states = object_states

    def task_object_hook(spec):
        if isinstance(spec, Monitor):
            return _arena_object(spec)
        if isinstance(spec, Cloth):
            return _cloth_object(spec)
        return task_object(spec)

    scene.task_object = task_object_hook
    scene._lg_monitor = True


def _patch_embodiment(module) -> None:
    cls = module.YamArenaEmbodiment
    if getattr(cls, "_lg_monitor", False):
        return
    base_step, base_reset = cls._task_step, cls.reset

    def _task_step(self, *args, **kwargs):
        for fn in _STEP:
            fn(self._env)
        return base_step(self, *args, **kwargs)

    def reset(self, *args, **kwargs):
        obs = base_reset(self, *args, **kwargs)
        seat_joints(self._env)
        for fn in _RESET:
            fn(self._env)
        return obs

    cls._task_step, cls.reset = _task_step, reset
    cls._lg_monitor = True


class _Finder(importlib.abc.MetaPathFinder):
    targets = {"yam_arena.scene": _patch_scene, "yam_arena.inspect_embodiment": _patch_embodiment}

    def find_spec(self, fullname, path, target=None):
        if fullname not in self.targets:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None
        loader, patch = spec.loader, self.targets[fullname]
        exec_module = loader.exec_module

        def exec_and_patch(module):
            exec_module(module)
            patch(module)

        loader.exec_module = exec_and_patch
        return spec


def install() -> None:
    for name, patch in _Finder.targets.items():
        if name in sys.modules:
            patch(sys.modules[name])
    if not any(isinstance(finder, _Finder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _Finder())


install()
