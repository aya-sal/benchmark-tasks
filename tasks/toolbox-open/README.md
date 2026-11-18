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

The script approaches 30° below horizontal with the jaws fully open, pinches each lower tab across its width and swings both clips outward. Once both clips reach the release threshold, the arms lift from the measured clip angles. One gripper holds the lid while the other gets underneath and pushes. The script writes a setup image, camera videos and `result.json` with the final angles, score, phase and worst IK error per arm. Add `--no-record` to skip saving camera media.

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

At `x = 0.222 m` and `z = 0.877 m`, the lower tab spans 71.92 mm along station Y. The target TCP is `(0.214206, ±0.155000, 0.881500)`. Positive Y is the left arm. The jaws enter at 93.9 mm and command a 63.85 mm gap to apply pressure. The demo logs the actual TCP and opening after closing in `result.json` under `grasp_tracking`. A GPU rollout measured openings of 71.69 mm on the left and 71.63 mm on the right.

## Scoring

| Score | Condition |
|---|---|
| 0 | Neither clip is at least 25° open |
| 0.25 | One clip is at least 25° open |
| 0.5 | Both clips are at least 25° open |
| 0.75 | Both clips are released and the lid is at least 45° open |
| 1.0 | Both clips are released and the lid stays at or above 85° with angular speed at most 2°/s for one second |

Partial credit uses the final state. Full credit uses Arena's one-second success hold.

The fixed-layout GPU demo scored 1.0 after 540 control steps (27 simulated seconds), with the lid at 90.0° and the clips at 82.0° and 68.8°. Peak nominal IK position error later in the motion was approximately 131 mm. The recorded grasp positions were within 0.5 mm of their targets. Separate trials for hook contact with zero or one clip released remain unverified.

Original asset by NVIDIA, `storage_toolbox_standard_hardCase_d01`, CC BY 4.0. The license is included in the original asset's `.metadata` directory.
