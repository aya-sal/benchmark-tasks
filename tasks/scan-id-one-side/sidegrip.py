"""Side grips for the scripted demo: a waypoint whose gripper is any full rotation, not only the straight-down grip.

``yam_arena.demo.Waypoint`` carries one number, ``yaw``, and ``ArmPlan`` turns it into the TCP rotation by calling the
module-level ``grip_rotation(w.yaw)`` every step. A side grip needs a whole rotation, so a waypoint here stores it on
its ``yaw``: a float subclass that still equals a yaw but also carries the rotation. Importing this module wraps
``grip_rotation`` so it returns the stored rotation when there is one and calls the original otherwise. Plain floats
behave exactly as before, so the left hand's straight-down waypoints are unaffected. The import checks that the plan
honours such a waypoint and raises if it does not.

Convention (TCP frame): x is the finger-closing axis, z the approach axis (from the TCP toward the pad tips), y = z x x.
"""

from __future__ import annotations

import numpy as np
import yam_arena.demo as _demo

MARK = "_sidegrip_original"


class SideYaw(float):
    """A yaw (rad) that also carries the full TCP rotation of a side-grip waypoint."""

    rotation: np.ndarray

    def __new__(cls, yaw: float, rotation: np.ndarray):
        obj = super().__new__(cls, yaw)
        obj.rotation = np.array(rotation, dtype=float)
        return obj

    def __reduce__(self):
        return (SideYaw, (float(self), self.rotation))


def _wrap(module, name: str) -> None:
    fn = getattr(module, name)
    original = getattr(fn, MARK, fn)

    def grip_rotation(yaw: float = 0.0) -> np.ndarray:
        rot = getattr(yaw, "rotation", None)
        return rot.copy() if rot is not None else original(yaw)

    setattr(grip_rotation, MARK, original)
    setattr(module, name, grip_rotation)


_wrap(_demo, "grip_rotation")
_KIN = getattr(_demo, "_k", None)  # tools/offline's stand-in only
if _KIN is not None and hasattr(_KIN, "grip_rotation"):
    _wrap(_KIN, "grip_rotation")


def rotation(fingers, approach) -> np.ndarray:
    """The TCP rotation whose finger-closing axis is ``fingers`` and approach axis ``approach`` (world frame).
    ``fingers`` is made square to ``approach``."""
    z = np.asarray(approach, dtype=float)
    z = z / np.linalg.norm(z)
    x = np.asarray(fingers, dtype=float)
    x = x - z * float(x @ z)
    x = x / np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])


def side(fingers, approach) -> SideYaw:
    """A waypoint ``yaw`` value for that grip."""
    return SideYaw(0.0, rotation(fingers, approach))


def _self_check() -> None:
    w = _demo.Waypoint(1.0, (0.30, -0.10, 0.80), 0.0, "check", side((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))
    plan = _demo.ArmPlan([w], (0.0, -0.305, 0.75))
    _, rot, _, _ = plan.target(10.0)
    if not np.allclose(rot, w.yaw.rotation, atol=1e-9):
        raise RuntimeError("sidegrip.py: yam_arena.demo.ArmPlan ignores full-rotation waypoints on this stack")


_self_check()
