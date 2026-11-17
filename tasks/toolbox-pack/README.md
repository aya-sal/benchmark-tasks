# Sort tools and supplies into an open toolbox

Put the grease bottle and spray can upright on the left half of the open toolbox with the sealant tube. Put the hammer, pliers, wrench, screwdriver, and duct tape on the right half. Fold both safety-glasses temples and put the glasses anywhere inside last.

## Geometry and assets

The scene uses the copied SimReady hard-case toolbox. Its lid starts open at 90° and the case is fixed. The wrapper retains the source visuals and approximates the cavity with a floor, four wall colliders and a lid collider. It scales the asset by 1.49 along its front-to-back axis, 1.21 along its hinge axis and 0.45 vertically. The resulting usable footprint is about 33 × 55 cm, with the rim about 7.9 cm above the case origin. The hinge faces away from the YAM arms. The case is set 25 mm farther back to clear the approach to the goggles.

There is no physical divider. The scorer uses the cavity centreline along station Y = 0. The glasses can cross that line and still score when their full footprint fits inside.

The goggles have independent passive temple hinges. Reset starts both at 0° (open). The left gripper lifts and holds the frame while the right gripper folds both temples in the air. The left gripper then packs the folded glasses. The microfiber towel is outside this task.

To regenerate the local USD wrappers with OpenUSD Python bindings:

```bash
python prepare_toolbox.py
python prep_assets.py
```

## Run

From this folder on an NVIDIA GPU machine:

```bash
source /data/IsaacLab-Arena/yam.env
python demo.py task.py demo_out --max-steps 3600
```

All packing assets are 5% larger than in the previous layout. The demo grasps both bottles from above around their caps. Both tilt in transit for reach and return upright before placement. It uses measured grasp points on the hammer handle, pliers jaws, wrench and screwdriver handle. Each item gets one grasp and one continuous transfer. The demo keeps the gripper closed until placement and stops at the first missed transfer or failed placement. Bottle placement uses a slow final descent, live base-height and tilt checks, a closed-jaw hold on the floor, and a slow release before vertical retreat. The transfer target leaves 32 mm of clearance between the carried item’s bottom and the highest case rim. The goggles are attempted once after all other items are packed. The result JSON records the item and reason if the demo stops early. Use `--no-record` to skip saving camera media.

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Put the grease bottle and spray can upright on the left half of the open toolbox with the sealant tube. Put the hand tools and duct tape on the right half. Fold both temples of the safety glasses, then put them anywhere inside the toolbox last." \
  --scorer task_score --max-steps 3600 --store-frames --log-dir demo_out/dream
```

The layout is fixed. The limit is 3,600 steps at 20 Hz (180 simulated seconds). Wall-clock runtime and contact behaviour for the new case are pending a GPU rollout.

## Inspect bottle stability with `yam-sim edit`

Use the older YAM simulator environment that provides the editor. From this folder, open each prepared scene in a separate editor session.

```bash
/data/env_isaaclab/bin/yam-sim edit bottle-seating-original.usda
/data/env_isaaclab/bin/yam-sim edit bottle-seating-flat.usda
```

Both scenes start the upright bottles 1 mm above the same toolbox floor with the same masses and placements. The original scene uses the supplied SDF colliders. The flat scene uses the demo's measured cylindrical body and cap colliders, with flat bases and zero restitution. The visual assets are identical. In the **YAM scene** panel, use **Play** to let the bottles settle, **Pause** to inspect the poses, and **Stop** before changing physics settings. A bottle tipping in this static test points to floor contact or collider geometry. If both bottles stand, compare the demo's `bottle_release` measurements before opening, after opening, and after wrist retreat. The editor holds its arms in their ready pose, so it cannot reproduce the scripted release.

Regenerate the two scene files and the flat collider wrappers with `python bottle_editor.py` in a Python environment with OpenUSD and `yam_arena`. Scene edits made in the editor are for diagnosis. The benchmark's poses are authored in `task.py`, and the collider wrappers are referenced separately.

## Score

Each correctly packed item earns 1/9. Supplies and tools must fit fully on their assigned sides and sit close to the floor. The grease bottle and spray can must be within 15° of vertical. On the glasses’ first settled placement inside the cavity, both temples must already be at least 75° folded and all other items must already be packed and settled. All nine placements must remain valid and still for one second to terminate successfully.
