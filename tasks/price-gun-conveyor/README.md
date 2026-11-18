# price-gun-conveyor

**Instruction:** "Stick one price label from the price gun in the white price panel on top of each box as the boxes ride past on the belt."

Two light retail boxes, each with a white price panel on its top face, ride a slow belt past the
arms. Beside the belt a hand price-label gun lies on a rest block. The gun is taken by its handle
and squeeze lever. Squeezing the lever home feeds one label to the nose; touching the nose roller
on a surface sticks that label there. The lever has to be let out and squeezed again before the gun
feeds the next label. The belt does not stop: each label goes on while its box is moving, and the
gun is put back down off the conveyor.

## Scene

Positions are in the station frame (m): x away from the arm bases, y to the left, bench top z = 0.

| object | type | size / mass | pose |
|---|---|---|---|
| conveyor belt | fixed PVC belt under driven steel carriers | 160 mm wide, surface 20 mm above the bench, from y −0.560 to +0.600 at x 0.290 | ramps over 0.5 s to 20 mm/s toward +y from the first control step |
| side rails and end stop | fixed aluminium | rails 8 mm; end-stop plate 6 x 40 mm at y +0.600 | beside and at the left end of the belt |
| box B (leads) | free, printed carton board, green | 90 x 100 x 50 mm, 220 g | upright on its carrier, centre (0.290, −0.460) at the start |
| box B's price panel | printed on its top face, white | 40 x 28 mm | in the middle of the top |
| box A (follows) | free, printed carton board, blue | 110 x 160 x 45 mm, 180 g | lying flat on the belt, centre (0.290, −0.620), 30 mm behind B |
| box A's price panel | printed on its top face, white | 40 x 28 mm | 20 mm toward −x and 30 mm toward +y of the top's centre, beside a barcode |
| price gun | free | 250 g with lever and roller; body 90 x 36 x 30 mm; handle 16 x 30 x 47 mm; 16 mm roller; six orange 22 x 12 mm labels | on its rest, body centre (0.130, −0.320), nose toward +y |
| squeeze lever | part of the gun, red | 8 x 26 x 43 mm, sprung 16° forward | sprung out |
| gun rest | fixed, dark rubber block | 50 x 70 x 20 mm | under the gun's body, centre (0.130, −0.320) |
| carriers | driven steel plates | 8 mm thick, 10 mm larger than each box | under each box; driven exactly along the belt |

Untouched, box B's panel passes x 0.290, y −0.08 at about 19.5 s and box A's at about 26 s.

## Scoring

Measured with a rule on the boxes and a count of the labels:
- **0.25** a label is stuck anywhere on either box.
- **0.5** one box has a label lying wholly inside its 40 x 28 mm price panel.
- **0.75** both boxes do.
- **1.0** also exactly one label on each box and none anywhere else (two labels gone from the gun's
  six: none on the belt, the bench or the rest, none wasted), both boxes upright (within 10 degrees)
  and within 20 mm of their place on the belt (the outline that rides with them), the gun put down
  off the conveyor (its lowest point within 25 mm of the bench top and none of it over the belt, no
  gripper within 10 cm of its handle), both grippers more than 5 cm from both boxes, held 1 s.

Run limit: 960 control steps (48 s). The scripted demo takes 703 steps (35 s).

## Scripted demo

The right arm takes the gun (the first squeeze feeds a label), lifts it clear of the tall box, meets
box B's panel from above while the belt carries it past, and touches the roller on. It lifts off,
lets the lever out, squeezes again for the second label, meets box A's panel the same way, then lays
the gun back on its rest and stands clear. The left arm stays at its rest pose. The plan is open
loop: measured arm-sag and landing offsets (`SAG`, `AIM` in `task.py`) keep each touch on the moving
panel.

## Simulation approximations

- **Belt / carriers:** steel carrier plates under the boxes are driven exactly along the belt (pose
  and velocity written before every physics step, like a servo-driven line). Boxes ride by friction
  and can be held back, pushed or knocked off. Cross stripes and an outline round each box's place
  are drawn marks slid along with the belt's travel.
- **Squeeze lever / labels:** the same model as `price-gun-two-boxes` (TK-HQ9RXC): the lever angle
  drives the feed; a label sticks when the roller comes within 2 mm of a surface; a second squeeze
  with a label still at the nose wastes it. No adhesion force, peel, wipe or printing.
- **Gun shape:** adapted to a top-down parallel gripper; it lies on a rest so its nose is off the
  bench when it is picked up.

## Reproduction

`runtime/` holds the pinned runner. From this task directory, with the `yam-inspect-isaac:6.1.0`
image (or `python3 runtime/build_image.py --image yam-inspect-isaac:6.1.0`):

```bash
mkdir -p /tmp/rc-task-home/tmp ./outputs
docker run --rm --gpus device=0 --network host --shm-size 8g \
  --user "$(id -u):$(id -g)" -v "$PWD":/task:ro -v "$PWD/outputs":/outputs:rw \
  -v /tmp/rc-task-home:/tmp/task-home:rw -e HOME=/tmp/task-home -e TMPDIR=/tmp/task-home/tmp \
  -e NVIDIA_DRIVER_CAPABILITIES=all -e OMNI_KIT_ACCEPT_EULA=YES -e ACCEPT_EULA=Y \
  --workdir /outputs --entrypoint /isaac-sim/python.sh yam-inspect-isaac:6.1.0 \
  /task/run.py --policy scripted --episodes 1 --livestream 0 --viz none
python runtime/rescore.py --task task.py outputs/TASK/RUN/episode-01/trajectory.jsonl \
  --expect outputs/TASK/RUN/episode-01/result.json
```

Or, in an environment that already has Isaac Lab-Arena 0.3.1 and `yam_arena` installed at the
commits in `runtime/prepare.py`:

```bash
python run.py --policy scripted --episodes 1 --livestream 0 --viz none --output-dir ./outputs
```

Seed 0; the scene settles for 30 ticks after reset before control starts. The belt starts with the
first control step. Control is 20 Hz and physics 240 Hz.

## Files

`task.py` (scene, belt drive, label model, scripted demo), `scoring.py` (stateful scorer),
`gun.py` (gun, boxes and measurements), `conveyor.py` (belt geometry and surface velocity),
`simkit.py` (compound rigid parts, joints and the demo controller patches). No third-party assets.
No hardware validation has been performed.

## Credit

This task is based on TK-HQ9RXC (`tasks/price-gun-two-boxes`), itself based on TK-HYQRVJ
(`tasks/carton-tape-seal`). `runtime/`, `run.py` and `simkit.py` are unchanged copies from those
tasks. The belt, the moving-box layout, the scorer's "place on the belt" check and the timed demo
are new here. The belt's surface-velocity drive follows the same PhysX pattern used by accepted
conveyor tasks such as TK-CJ4EAQ (`tasks/push-rejects-off-moving-line`).
