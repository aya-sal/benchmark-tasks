# Copy one side of an ID card on a flatbed copier

The card starts blue-side up on a 55 × 27 × 23 mm stand. One A4 sheet starts leaning back 25 degrees in the rear slit. The left arm sets the card on the glass, the right arm pulls the lid down until gravity and the detent close it, and the left arm presses Scan. The copier then scans, prints and leaves the sheet on the raised front tray. Both arms wait at the rest pose.

The scripted policy lasts about 40 s (about 800 control steps at 20 Hz). The run limit is 1,200 steps, which includes the 1 s the runner holds a score of 1.0. `object_jitter=0` is the only seed: the start layout does not move between trials.

## Run a trial

From this folder, on a machine with the Isaac Lab-Arena environment:

```bash
# Scripted demo
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task="$PWD/task.py" -E object_jitter=0 -E control_hz=20 \
  --instruction "$(python -c 'import json; print(json.load(open("task.json"))["instruction"])')" \
  --scorer task_score --max-steps 1200 --store-frames --no-prompt --grader none \
  --log-dir "$HOME/yam-eval/scan-id-one-side-demo"

# Model run (Dream-1)
inspect-robots run --embodiment yam_arena --policy dream \
  -E task="$PWD/task.py" -E object_jitter=0 -E control_hz=20 \
  --instruction "$(python -c 'import json; print(json.load(open("task.json"))["instruction"])')" \
  --scorer task_score --max-steps 1200 --store-frames --no-prompt --grader none \
  --log-dir "$HOME/yam-eval/scan-id-one-side-dream"
```

Add `-E headless=false` to watch the simulator. The log prints `task_score` at the end. Turn a log into a video with `inspect-robots video`, then H.264 with:

```bash
ffmpeg -i in.mp4 -c:v libx264 -pix_fmt yuv420p -movflags +faststart out.mp4
```

## Checks without the simulator

```bash
python -m unittest test_scan
```

## Still needed before the pull request

Record the setup screenshot and both videos, then from the repository root:

```bash
export ROBOCURVE_TEAM=... ROBOCURVE_TOKEN=...
python tools/upload_media.py tasks/scan-id-one-side \
  --video model_run.mp4 --demo demo.mp4 --screenshot setup.png
```

Put the scores the scorer printed into `task.json` (`"scores"`). Do not commit the token.
