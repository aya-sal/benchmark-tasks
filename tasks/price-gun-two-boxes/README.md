# price-gun-two-boxes

**Instruction:** "Stick one price label from the price gun in the white price panel on top of each box."

Two boxed products stand on the bench, each with a white price panel printed on its top face. Beside
them a hand price-label gun lies on a rest block. The gun is taken by its handle and squeeze lever.
Squeezing the lever home feeds one label to the nose; touching the nose roller on a surface sticks that
label there. The lever has to be let out and squeezed again before the gun feeds the next label. Each
box gets one label, wholly inside its panel, and the gun is put back down.

## Scene

Positions are in the station frame (m): x away from the arm bases, y to the left, bench top z = 0.

| object | type | size / mass | pose |
|---|---|---|---|
| box A | free, printed carton board, blue | 160 x 110 x 45 mm, 150 g | lying flat, centre (0.315, -0.130) |
| box A's price panel | printed on its top face, white | 40 x 28 mm, long edge along x | centre (0.280, -0.110): 35 mm nearer the arms and 20 mm left of the top's centre, beside a barcode |
| box B | free, printed carton board, green | 80 x 60 x 100 mm, 200 g | standing upright, centre (0.220, 0.005) |
| box B's price panel | printed on its top face, white | 40 x 28 mm, long edge along x | in the middle of the top, centre (0.220, 0.005) |
| price gun | free | 250 g with lever and roller; off-white body 90 x 36 x 30 mm; black handle 16 x 30 x 47 mm standing on the back of the body; 16 mm rubber roller, 26 mm wide, on a free axle at the nose, its bottom 8 mm below the body's; six orange 22 x 12 mm labels loaded | lying on its rest, body centre (0.150, -0.250), nose toward +y, roller 12 mm clear of the bench |
| squeeze lever | part of the gun, red | 8 x 26 x 43 mm bar hinged at its foot in front of the handle, sprung 16 degrees forward (12 mm at its tip), about 9 N at the tip to start it and 10 N to hold it home | sprung out |
| gun rest | fixed, dark rubber block | 50 x 70 x 20 mm | under the gun's body, centre (0.150, -0.250) |

## Scoring

Measured with a rule on the boxes and a count of the labels:
- **0.25** a label is stuck anywhere on either box.
- **0.5** one box has a label lying wholly inside its 40 x 28 mm price panel.
- **0.75** both boxes do.
- **1.0** also exactly one label on each box and none anywhere else (two labels gone from the gun's
  six: none on the bench or the rest, none wasted), both boxes upright (within 10 degrees) and within
  20 mm of where they started, the gun put down (its lowest point within 25 mm of the bench top, and no
  gripper within 10 cm of its handle), both grippers more than 5 cm from both boxes, held 1 s.

Gripper distances are measured from the point between the fingertips (the TCP).

Run limit: 1,600 control steps (80 s). The scripted demo takes 567 steps (28 s).

## Scripted demo

The right arm comes down round the gun's handle and lever and squeezes them: that takes the gun and
feeds the first label. It lifts the gun off its rest, carries it over box A's panel, comes down to 6 mm
above it, steadies, and touches the roller on the panel. It lifts the gun high enough for the taller
box, lets the lever right out (the gun is still held, by the lever's spring against the jaws),
squeezes again for the second label, carries the gun over box B's panel and touches it on the same
way. It lays the gun back on its rest, lets go and stands clear. The left arm is not needed and stays
at its rest pose. The plan is about 30 s; the trial ends on success when the score has held 1 s.

The plan is open loop. The arms hold position by joint stiffness, so the gun's weight lowers its nose
by 4 to 9 mm depending on the reach and the roller lands about 6 mm short of where the hand is sent;
both offsets were measured on the settled arm and are constants in `task.py` (`SAG`, `AIM`). With them
both labels land within 1 mm of their panel's centre, 7.7 mm inside its nearest edge.

## Simulation approximations

- **Squeeze lever:** a real sprung hinge on the gun, squeezed by the gripper; its measured angle
  drives the label feed. At 85 % of its travel one label is fed to the nose (most real guns feed on
  the release stroke), and it must come back out to 30 % before it can feed again. The gun holds six
  labels; a seventh squeeze feeds nothing.
- **Labels** are not a film. While a label is at the nose (drawn hanging in front of the roller), the
  middle of the roller's axle coming within the roller's radius plus 2 mm of a surface (a box face,
  the bench or the rest) sticks the whole 22 x 12 mm label there, centred under the roller with its
  long edge along the roller, and it stays on that body. There is no adhesion force, peel, wipe or
  printing.
- **Wasted labels:** squeezing again while a label is still at the nose pushes that label off. It is
  counted (so the full score is lost) but not drawn.
- **The gun's shape** is adapted to a top-down parallel gripper: its handle and lever stand upright on
  the body, where a hand-held gun's are raked back, and it lies on a rest block so that its nose is
  off the bench when it is picked up (squeezing it with the roller on the bench sticks the label to
  the bench).
- **Roller:** a separate rubber body on a free-turning axle.
- **Boxes** are rigid; their artwork is flat colour.

## Reproduction

`runtime/` holds the pinned runner. From this task directory, with the `yam-inspect-isaac:6.1.0` image
(or `python3 runtime/build_image.py --image yam-inspect-isaac:6.1.0`):

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

Or, in an environment that already has Isaac Lab-Arena 0.3.1 and `yam_arena` installed at the commits
in `runtime/prepare.py`:

```bash
python run.py --policy scripted --episodes 1 --livestream 0 --viz none --output-dir ./outputs
```

Seed 0; the scene settles for 30 ticks after reset before control starts. Control is 20 Hz and
physics 240 Hz. One scripted episode takes about 90 s of wall time on an L40S after about 2 minutes of
simulator start-up. Each episode writes four 20 fps camera videos, setup/final PNGs, the measured
state trajectory and result.json. The scorer reads only measured object poses, robot TCPs and the
task's own measured quantities (the lever's angle and the labels).

## Files

`task.py` (scene, label model, scripted demo), `scoring.py` (stateful scorer), `gun.py` (geometry and
measurements), `simkit.py` (compound rigid parts, joints and the demo controller patches). No
third-party assets. No hardware validation has been performed.

## Credit

This task is based on the accepted task TK-HYQRVJ, "Seal a filled carton with a tape gun: tape along
the seam, tear it off, smooth both tabs down" (`tasks/carton-tape-seal`). `runtime/`, `run.py` and
`simkit.py` are unchanged copies from that task, and `task.py` and `scoring.py` follow its structure.
The scene, the price gun and its label model, the scorer and the scripted demo are new here.
