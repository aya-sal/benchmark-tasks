"""Scripted wipe. Vertical strokes along the glass.

The monitor stands 20 cm forward of the centre of the desk. The right arm picks up the cloth.
Run from this directory inside the Isaac Lab-Arena environment:

    python demo.py task.py demo_out --max-steps 2400
    DEMO_VARIANT=idle python demo.py task.py idle_out --no-record
    DEMO_VARIANT=shove python demo.py task.py shove_out --no-record
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

CONTROL_HZ = 20


class Arm:
    def __init__(self, name, base_pos):
        from yam_arena import ik
        from yam_arena.constants import READY

        self.name, self.base = name, np.asarray(base_pos, float)
        self.q = np.array(READY, float)
        self.grip = 1.0
        self.pos, self.rot = ik.tcp_pose(self.q, self.base)
        self.cmd = np.r_[self.q, self.grip]
        self.worst_mm = 0.0

    def _solve(self, pos, rot):
        from yam_arena import ik
        from scipy.optimize import least_squares

        def residual(q):
            p, R = ik.tcp_pose(q, self.base)
            return np.r_[p - np.asarray(pos), 0.3 * ik._rot_error(rot, R)]

        fit = least_squares(
            residual, np.clip(self.q, ik.LOWER + 1e-7, ik.UPPER - 1e-7),
            bounds=(ik.LOWER, ik.UPPER), max_nfev=80, ftol=1e-8, xtol=1e-8, gtol=1e-8,
        )
        reached, _ = ik.tcp_pose(fit.x, self.base)
        self.q, self.pos, self.rot = fit.x, np.asarray(pos, float), rot
        self.worst_mm = max(self.worst_mm, float(np.linalg.norm(reached - pos)) * 1000)
        self.cmd = np.r_[self.q, self.grip]

    def move(self, pos, rot, duration, grip=None):
        from yam_arena import ik

        p0, r0, g0 = self.pos.copy(), self.rot.copy(), self.grip
        g1 = g0 if grip is None else grip
        n = max(1, round(duration * CONTROL_HZ))
        for k in range(1, n + 1):
            s = ik.smoothstep(k / n)
            self.grip = g0 + s * (g1 - g0)
            self._solve(p0 + s * (np.asarray(pos) - p0), ik.slerp_matrix(r0, rot, s))
            yield

    def set_grip(self, grip, duration=0.6):
        yield from self.move(self.pos, self.rot, duration, grip)


def wait(seconds):
    for _ in range(max(1, round(seconds * CONTROL_HZ))):
        yield


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def wipe_rotation(grasp_R, glass_out):
    """TCP pose that keeps the cloth's thin axis (world z at the grasp) on the glass normal.

    The pad is flat on the desk when it is grasped, so its local z is world z. After the
    grasp, cloth_R = R_tcp @ grasp_R.T, and cloth local z must end up along ``glass_out``.
    """
    z0 = grasp_R.T @ np.array([0.0, 0.0, 1.0])
    z0 = _unit(z0)
    helper = np.array([1.0, 0.0, 0.0]) if abs(z0[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x0 = _unit(np.cross(helper, z0))
    y0 = np.cross(z0, x0)
    z = _unit(glass_out)
    up = np.array([0.0, 0.0, 1.0])
    x = up - z * float(np.dot(up, z))
    x = _unit(x)
    y = np.cross(z, x)
    return np.column_stack([x, y, z]) @ np.column_stack([x0, y0, z0]).T


def script(task, ctx):
    import scoring
    from yam_arena.demo import grip_rotation

    variant = os.environ.get("DEMO_VARIANT", "")
    arm = ctx.right
    if variant == "idle":
        yield from wait(3)
        return
    from yam_arena.constants import PAD_TIP_PAST_TCP

    down = grip_rotation(0.0)
    x, y, _ = scoring.CLOTH_POS
    z = scoring.TOP + PAD_TIP_PAST_TCP
    edge_y = y + scoring.CLOTH_SIZE[1] / 2 + 0.02
    left = ctx.left
    yield from left.move((x, edge_y, z + 0.12), down, 1.6, grip=0.0)
    yield from left.move((x, edge_y, z), down, 1.0)
    yield from left.move((x, y, z), down, 2.2)
    yield from left.move((x, y, z + 0.14), down, 0.8)
    yield from left.move((x, edge_y + 0.04, z + 0.14), down, 0.8)
    pinch = down
    cloth = np.array([x, y - 0.03, z])
    yield from arm.move(cloth + (0, 0, 0.12), pinch, 1.6, grip=1.0)
    yield from arm.move(cloth, pinch, 1.2, grip=1.0)
    yield from wait(0.8)
    yield from arm.set_grip(0.0, 0.8)
    yield from wait(0.6)
    yield from arm.move(cloth + (0, 0, 0.18), pinch, 1.0)
    face = task._wipe_rotation()
    if variant == "shove":
        # Drive the cloth into the outer bezel and sideways.
        outer = np.array(scoring.yaw180((-0.01, -0.28, 0.42), scoring.HEIGHT))
        yield from arm.move(outer + (-0.08, 0, 0), face, 2.0)
        yield from arm.move(outer + (0.0, 0.10, 0), face, 2.5)
        yield from wait(1.5)
        return
    yield from arm.move((0.36, -0.10, 1.00), face, 1.6)
    for duration, pos, rot, _label in task._stain_rubs():
        yield from arm.move(pos, rot, duration)
    yield from wait(1.5)


class Ctx:
    def __init__(self):
        from yam_arena import station

        self.left = Arm("left", np.asarray(station.BASES["left"], float))
        self.right = Arm("right", np.asarray(station.BASES["right"], float))

    def action(self):
        from inspect_robots.types import Action
        from inspect_robots_yam.packing import pack

        return Action(data=pack(self.left.cmd, self.right.cmd))


def _write_mp4(frames, path: Path) -> None:
    import imageio

    imageio.mimsave(
        path, [np.asarray(f) for f in frames], fps=CONTROL_HZ, codec="libx264",
        output_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )


def run(task_path, out_dir, max_steps=2400, record=True, cam_width=320, cam_height=176, window=False):
    from types import SimpleNamespace

    from inspect_robots import Scene
    from yam_arena.inspect_embodiment import yam_arena_embodiment
    from yam_arena.scoring import task_score

    task_path = str(Path(task_path).resolve())
    sys.path.insert(0, str(Path(task_path).parent))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    emb = yam_arena_embodiment(
        task=task_path, object_jitter=0, control_hz=CONTROL_HZ, cam_width=cam_width,
        cam_height=cam_height, device=os.environ.get("DEMO_DEVICE", "cpu"),
        headless=not window,
    )
    obs = emb.reset(Scene(id="demo", instruction="demo"))
    ctx = Ctx()
    gen = script(emb.task.module, ctx)
    frames: dict[str, list] = {}

    def snap(observation):
        if record:
            for cam, img in observation.images.items():
                frames.setdefault(cam, []).append(np.asarray(img))

    snap(obs)
    steps, ended, res = 0, None, None
    while steps < max_steps:
        try:
            next(gen)
        except StopIteration:
            ended = steps if ended is None else ended
        res = emb.step(ctx.action())
        steps += 1
        snap(res.observation)
        if steps % 50 == 0:
            print(f"step {steps} score {res.info.get('score')}", flush=True)
        if res.terminated or (ended is not None and steps - ended >= 2.5 * CONTROL_HZ):
            break
    scored = task_score()(SimpleNamespace(steps=[SimpleNamespace(result=res)]), None)
    result = {"task_score": scored.value, "explanation": scored.explanation, "steps": steps,
              "variant": os.environ.get("DEMO_VARIANT", ""),
              "worst_ik_mm": {"left": round(ctx.left.worst_mm, 2), "right": round(ctx.right.worst_mm, 2)}}
    if record and frames:
        import imageio

        cams = sorted(frames, key=lambda c: ("left" not in c, "top" not in c))
        _write_mp4([np.hstack([frames[c][i] for c in cams]) for i in range(len(frames[cams[0]]))], out / "demo.mp4")
        imageio.imwrite(out / "setup.png", np.hstack([frames[c][0] for c in cams]))
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("out")
    ap.add_argument("--max-steps", type=int, default=2400)
    ap.add_argument("--no-record", action="store_true")
    ap.add_argument("--window", action="store_true", help="open the Isaac Sim window on the browser desktop")
    args = ap.parse_args()
    run(args.task, args.out, args.max_steps, record=not args.no_record, window=args.window)
