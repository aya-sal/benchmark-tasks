# Wipe fingerprints off a desktop monitor without sliding it off the desk

Eight oily marks sit on the glass of an LG UltraGear 27GX790A. The glass is wider than the foot and hangs past the desk's left edge; the foot's left corners are 30 mm from that edge. A stroke toward the overhang walks the foot off the desk, and the trial then scores 0 even if the glass was already clean. The scripted policy wipes each mark vertically and lets go.

The demo and the Dream-1 run have not been executed here. `test_score.py` checks the layout and the score table without Isaac Sim.

## The monitor

[LG 27GX790A-B](https://www.lg.com/uk/monitors/gaming/27gx790a-b/) is a flat 26.5 inch OLED, 2560×1440. With the stand it is 605.2 × 579.3 × 249.1 mm and 9.3 kg at full height, and 605.2 × 469.3 × 249.1 mm at the low stop (110 mm of travel). Without the stand the head is 605.2 × 351.0 × 45.3 mm and 5.1 kg. The manual gives tilt as 5° toward the viewer and 15° back. The owner's manual is [27GX790A](https://gscs-b2c.lge.com/downloadFile?fileId=XLzMDS6LF1IKP3KS1mW4A).

The mesh is LG's SimReady articulation `monitor_27gx790a`, revision `e965976d90f025237ad9ea8881f5f82755f26d9d` of `LGElectronics/simready-assets` (CC-BY-4.0). It is downloaded on first use into `~/.cache/simready` and is not committed. Measured on that file:

- Three bodies: foot and column 1.974 kg, hinge 0.055 kg, head 7.271 kg.
- Glass: the flat `lcd` face at local x = 0.0531 m, normal +x, local y −0.2956 to 0.2956 m, local z 0.2364 to 0.5732 m.
- Height slide along z, limits −0.11288 to 0 m. Tilt about local y, limits −15° to +5° (the same 20° range as the manual; the file's sign runs the other way, with +5° toward the glass).
- Authored static friction 0.4. The file ships with gravity off on the head and the hinge, and with no tilt stiffness.

Spawn turns gravity on, sets restitution to 0.05, holds the height slide on its lower stop (40 N of slide friction, a 200 N/m drive at −0.110 m) and holds the tilt at 0 with 8 N·m of friction. That friction is an estimate so the head does not sag under its own weight. The asset is yawed 180° so the glass faces the robot. The asset origin, the bottom of the foot, is (0.433, 0.470, 0.7485) m.

Two 200 × 200 × 350 mm pedestals raise the arm bases by 350 mm so the grippers can reach the glass. They are part of the physical setup.

## Run

```bash
python test_score.py
python demo.py task.py demo_out --max-steps 2400
DEMO_VARIANT=idle python demo.py task.py idle_out --no-record
DEMO_VARIANT=shove python demo.py task.py shove_out --no-record
inspect-robots run --embodiment yam_arena --policy dream \
  -P model=dreamlabs/dream-1-beta -P rows_per_request=4 \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E device=cpu \
  -E cam_width=320 -E cam_height=176 \
  --instruction "Wipe every fingerprint off the monitor glass with the cloth. Leave the monitor standing on the desk, and let go." \
  --scorer task_score --max-steps 2400 --seed 1 --store-frames --log-dir model_out
```

`DEMO_VARIANT=shove` drives the cloth into the outer bezel and 10 cm toward the open edge.

## Score

- **1.0** — all 8 fingerprints wiped, both gripper centres at least 60 mm clear of the monitor, foot still on the desk, held for 2 s with under 3 mm of travel and under 1° of tip.
- **0.5** — all 8 wiped, but a hand is still on the monitor or the 2 s hold is not met.
- **0.25** — 4 to 7 wiped, foot still on the desk.
- **0** — fewer than 4 wiped, or the foot left the desk at any time.

A wipe counts only while a gripper holds the cloth (opening under 45 mm, gripper centre within 70 mm of the cloth) with the broad face on the mark, within 25° of the glass normal, inside the 110 mm face with 4 mm of slack, and within 8 mm of the mark. The cloth then slides at least 25 mm across the glass. The foot is off the desk when any underside corner leaves x −0.19 to 0.91 or y −0.65 to 0.65, drops more than 8 mm below the top, rises more than 12 mm, or the foot tips more than 8°. That failure does not clear.

## Third-party assets

| Asset | Source | License | Bundled |
|---|---|---|---|
| LG 27GX790A SimReady USD | [LGElectronics/simready-assets](https://huggingface.co/datasets/LGElectronics/simready-assets/tree/e965976d90f025237ad9ea8881f5f82755f26d9d/monitor_27gx790a) revision `e965976d90f025237ad9ea8881f5f82755f26d9d` | CC-BY-4.0 | No |
| YAM arm and gripper | i2rt | MIT | No |
| Station | yam-isaac-lab-arena | station SDK has no license yet; YAM models are MIT | No |

Task code is MIT, in `LICENSE`.
