# Release both toolbox clips and open the lid

The two YAM arms face a closed hard-plastic toolbox on the station table. Release the two yellow clips, then lift and push the lid to its 90° stop. The base is anchored at `(0.37, 0, 0.7485)` m, with the hinge away from the arms. The case is 31.4 × 53.1 cm in plan and the clips are 31 cm apart.

## Asset

The original USD loads with its geometry, materials, bodies and PhysX colliders. It uses metres and Z-up. Its two clips are disjoint mesh islands on one rigid link and one hinge, and it has no articulation root.

`assets/toolbox_open.usdc` is the task-ready layer. It references the original asset, splits the clip geometry and UVs, gives each clip an independent passive joint, and adds small hook and catch colliders. It also targets the actual rigid bodies in the lid and handle joints. At spawn, the task adapter anchors the base at its station pose. The render asset and its relative dependencies should stay together under `assets/`.

The converted layer is included. With a Python environment that has NumPy and the OpenUSD `pxr` bindings, inspect or regenerate it from this directory.

```bash
python inspect_asset.py assets/storage_toolbox*/simready_usd/sm*.usd
python prepare_asset.py
python inspect_asset.py assets/toolbox_open.usdc
```

## Run a trial

Activate the YAM Isaac Lab-Arena environment on a machine with a compatible NVIDIA GPU, then run from this directory.

```bash
source /data/IsaacLab-Arena/yam.env
python demo.py task.py demo_out --max-steps 1200
```

The script turns each gripper 90° so its jaws cross the clasp edge, approaches 20° below horizontal, and swings both clips outward. It raises the lid using the clips, and uses one gripper to hold the lid while the other gets underneath and pushes. It exits a phase early if the measured joint angles are too low to proceed. It writes a setup image, camera videos and `result.json` with the final angles, score, phase and worst IK error per arm. Add `--no-record` to skip saving camera media.

To evaluate Dream-1 through the standard benchmark pipeline, run:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Release both yellow toolbox clips, then lift and push the lid all the way open." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir demo_out/dream
```

Each trial allows 1,200 steps at 20 Hz, or 60 simulated seconds. The layout is fixed across seeds. Reset restores all four joint positions and velocities.

## Check clasp reach

Run `python measure_grasp.py` with the same OpenUSD Python environment used for asset inspection. It prints the clasp bounds, the two mesh intersections along the jaw-closing axis, the commanded TCP, and the TCP computed from the YAM URDF. Coordinates are in station metres.

At the closed pose, the selected clasp faces intersect the jaw axis at `(0.227660, ±0.155000, 0.875644)` and `(0.233075, ±0.155000, 0.890522)`, 15.83 mm apart. Their midpoint sets the grasp target. The commanded TCP is `(0.227549, ±0.155000, 0.884109)`. Positive Y is the left arm.

The straight insertion crosses the clasp and case on the upper jaw centreline, so free-space IK alone cannot establish reach under contact. The demo logs the actual TCP and finger opening after closing and before swinging in `result.json` under `grasp_tracking`. A short TCP in the simulation indicates that the approach needs clearance. A TCP at target with a finger opening near the 7.5 mm command suggests the clasp was missed or slipped out. These measurements still need a GPU rollout.

## Scoring

| Score | Condition |
|---|---|
| 0 | Neither clip is at least 25° open |
| 0.25 | One clip is at least 25° open |
| 0.5 | Both clips are at least 25° open |
| 0.75 | Both clips are released and the lid is at least 45° open |
| 1.0 | Both clips are released and the lid stays at or above 85° with angular speed at most 2°/s for one second |

Partial credit uses the final state. Full credit uses Arena's one-second success hold.

USD composition, joint targets, scorer boundaries and nominal robot IK were checked offline. GPU validation is pending for resets under PhysX, hook contact with zero or one clip released, and the full scripted rollout. Real-time runtime per trial is pending that rollout.

Original asset by NVIDIA, `storage_toolbox_standard_hardCase_d01`, CC BY 4.0. The license is included in the original asset's `.metadata` directory.
