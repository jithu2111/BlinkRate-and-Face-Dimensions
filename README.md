# Face Metrics — Blink Rate & Facial Dimension Estimator

A computer vision pipeline that processes video footage of a study session and reports:

- **(A) Blink rate** — blinks per second (and per minute) over the full session.
- **(B) Facial dimensions** in centimeters — face, eyes, nose, mouth, plus derived head height and measured ear-to-ear.

Built on **MediaPipe Face Landmarker** (478 landmarks including iris) and **MediaPipe Pose Landmarker** (ears). Calibration to real-world units uses the iris horizontal diameter as an anatomical reference (~11.7 mm), so no physical ruler is needed in the frame.

---

## 1. Prerequisites

- **Python 3.10 or 3.11** (MediaPipe wheels do not yet cover 3.12+ on all platforms)
- macOS, Linux, or Windows
- ~10 GB free disk space if you're processing the full 4-hour GoPro dataset (~55 GB of video input; no intermediate files are written)

Check your Python version:

```bash
python3 --version
```

---

## 2. Set up a virtual environment

From the project root (`MiniProject/`):

```bash
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
# OR
venv\Scripts\activate           # Windows
```

Your prompt should now start with `(venv)`.

---

## 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Dependencies (also listed in `requirements.txt`):

- `opencv-python>=4.8.0` — video decoding
- `mediapipe>=0.10.9` — face + pose landmarkers (Tasks API)
- `numpy>=1.24.0` — vector math

---

## 4. Prepare the videos

Put your session videos in a folder named `Videos/` at the project root:

```
MiniProject/
├── face_metrics.py
├── requirements.txt
├── README.md
└── Videos/
    ├── GX010680.MP4
    ├── GX010681.MP4
    └── ...
```

Supported extensions: `.mp4`, `.mov`, `.avi`, `.mkv`, `.m4v` (case-insensitive).

The first time you run the script, it will download two MediaPipe model files (~7 MB total) next to `face_metrics.py`:

- `face_landmarker.task`
- `pose_landmarker.task`

These are cached and reused on subsequent runs.

---

## 5. Run the program

### Full session (recommended)

Process every video in `Videos/` and report a session-wide aggregate:

```bash
python face_metrics.py --videos-dir Videos/ --sample-fps 5 --aggregate --out results.csv
```

### Single video (quick sanity check)

```bash
python face_metrics.py --video Videos/GX010680.MP4 --out smoke.csv
```

### Flags

| Flag | Default | Purpose |
|---|---|---|
| `--video PATH` | — | Process one video file. |
| `--videos-dir DIR` | — | Process every video in this directory. |
| `--sample-fps N` | `5` | Frames per second to analyze. Lower = faster, but too low may miss short blinks (<200 ms). |
| `--aggregate` | off | Add a final session-wide row (total blinks / total duration, median of per-file dimensions). |
| `--out PATH` | `results.csv` | Output CSV path. |

Exactly one of `--video` or `--videos-dir` is required.

---

## 6. What you get

### Console output (per video + aggregate)

```
=== Videos/GX010681.MP4 ===
Duration: 00:02:30  |  frames analyzed: 751
Blinks: 4   Rate: 0.0266 blinks/sec (1.6 blinks/min)
Calibration: iris = 41.4 px -> 1.17 cm
Dimensions (cm):
  Face     forehead-to-chin: 14.93   cheek-to-cheek: 12.80
  Head     head-to-chin (derived, x1.3): 19.41
  Head     ear-to-ear (Pose): 14.30
  L eye    W: 2.49   H: 0.75
  ...
```

### CSV output

One row per video, plus one aggregate row if `--aggregate` is set. Columns:

| Column | Meaning |
|---|---|
| `video` | Source file path |
| `duration_sec` | Video duration in seconds |
| `processed_frames` | Number of frames actually analyzed |
| `blinks` | Count of detected blinks |
| `blink_rate_per_sec` | blinks ÷ duration |
| `forehead_to_chin_cm` | Measured (Face Mesh landmark 10 → 152) |
| `cheek_to_cheek_cm` | Measured (landmark 234 → 454, bizygomatic width) |
| `head_height_cm_derived` | `forehead_to_chin × 1.30` (Farkas 1994 anthropometric ratio) |
| `ear_to_ear_cm_pose` | Measured via Pose ear landmarks (bitragion) |
| `left_eye_width_cm` / `left_eye_height_cm` | Outer-to-inner corner / top-to-bottom |
| `right_eye_width_cm` / `right_eye_height_cm` | Same |
| `nose_length_cm` / `nose_width_cm` | Bridge-to-tip / ala-to-ala |
| `mouth_width_cm` / `mouth_height_cm` | Corner-to-corner / outer-top to outer-bottom |
| `iris_px` | Median iris diameter in pixels (calibration anchor) |

---

## 7. How it works (brief)

- **Frame sampling.** The 4-hour dataset contains hundreds of thousands of frames; processing every one is wasteful. `--sample-fps 5` picks ~5 frames/second, which is still well above the Nyquist rate for blinks (100–400 ms).
- **Blink detection.** Computes the **Eye Aspect Ratio (EAR)** from 6 landmarks per eye (Soukupová & Čech 2016). A blink is counted when EAR drops below `0.21` for ≥2 consecutive sampled frames, then reopens — the debouncing stops a single long closure from being double-counted.
- **Pixel-to-cm calibration.** Uses the horizontal iris diameter, which is anatomically near-constant at **1.17 cm** in adults. Taking the **median** across the whole session removes outliers from head-turn frames.
- **Dimensions.** Per-frame pixel distances between landmark pairs, aggregated as the **median** across all frames (robust to head movement and blinks).
- **Ear-to-ear.** MediaPipe Face Mesh has no ear landmarks, so a second model (Pose) runs on each frame to measure bitragion width directly.

---

## 8. Performance expectations

On an Apple M3 with default `--sample-fps 5`:

- **~400–600 sampled frames/minute** of wall time.
- A **25-file, ~83-minute** dataset takes roughly **45–75 minutes**.

To speed up: lower `--sample-fps` to 3. Blinks are still reliably caught at 3 fps.

---

## 9. Known limitations

- **"Head height" is derived**, not directly measured. MediaPipe Face Mesh has no crown/vertex landmark, so head-to-chin is estimated as forehead-to-chin × 1.30 using Farkas's published adult ratio.
- **Ear-to-ear foreshortening.** The Pose-based bitragion width underestimates when the head is turned off-axis. Using the session median mitigates this, but for heavily off-axis footage the value may still come out smaller than cheek-to-cheek. In that case, fall back to `cheek_to_cheek_cm × 1.10` (Farkas).
- **Eye height is slightly low** because the median includes mid-blink and partial-closure frames. This is usually a 10–20% underestimate of the true open-eye height.
- **Iris calibration assumes a roughly frontal face.** Extreme yaw/pitch shrinks the projected iris ellipse; the session-wide median still works because most study footage is near-frontal.

---

## 10. Troubleshooting

**`AttributeError: module 'mediapipe' has no attribute 'solutions'`** — Your MediaPipe version is ≥0.10.15 where the legacy `mp.solutions` API was removed. The script already uses the new Tasks API; make sure you're running the latest `face_metrics.py`, not an older copy.

**`Cannot open video: ...`** — OpenCV can't decode the file. GoPro `.MP4` files usually work out of the box; if not, try `pip install opencv-python-headless` or convert with `ffmpeg -i input.MP4 -c copy fixed.mp4`.

**Model download fails** — Delete the partial `.task` file next to `face_metrics.py` and re-run. The script will retry the download.

**All dimensions are `0.000`** — Means no iris was ever detected (so no calibration scale). Usually the face is off-frame, too dark, or too small. Check a sample frame manually.