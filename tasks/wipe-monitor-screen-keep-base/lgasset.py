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

__all__ = ["Monitor", "fetch", "install", "on_reset", "on_step"]

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
    from pxr import UsdGeom
    import isaaclab.sim as sim_utils

    stage = sim_utils.get_current_stage()
    for index, path in _SMUDGES:
        prim = stage.GetPrimAtPath(path)
        if prim and clean[index]:
            UsdGeom.Imageable(prim).MakeInvisible()
        elif prim:
            UsdGeom.Imageable(prim).MakeVisible()


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


def _patch_scene(scene) -> None:
    if getattr(scene, "_lg_monitor", False):
        return
    task_object = scene.task_object
    original_build = scene.build_task_environment

    def task_object_hook(spec):
        return _arena_object(spec) if isinstance(spec, Monitor) else task_object(spec)

    def build(*args, **kwargs):
        environment = original_build(*args, **kwargs)
        task = kwargs.get("task")
        if task is not None and any(isinstance(obj, Monitor) for obj in task.objects):
            previous = environment.env_cfg_callback

            def configure(cfg):
                if previous is not None:
                    cfg = previous(cfg)
                for robot in (cfg.scene.left_robot, cfg.scene.right_robot):
                    x, y, z = robot.init_state.pos
                    robot.init_state.pos = (x, y, z + 0.35)
                return cfg

            environment.env_cfg_callback = configure
        return environment

    scene.task_object = task_object_hook
    scene.build_task_environment = build
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
