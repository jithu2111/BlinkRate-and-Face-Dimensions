"""
Blink-rate estimation and facial-dimension measurement from study video footage.

Uses MediaPipe Face Mesh (468 landmarks + 10 iris landmarks) to:
  (A) Detect blinks via Eye Aspect Ratio (EAR) and report blinks/second.
  (B) Measure face / eye / nose / mouth dimensions in centimeters, calibrated
      against the iris horizontal diameter (anatomical constant ~11.7 mm).

Usage:
    python face_metrics.py --video Videos/GX010680.MP4
    python face_metrics.py --videos-dir Videos/ --sample-fps 5
    python face_metrics.py --videos-dir Videos/ --aggregate   # one session total
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import urllib.request

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


FACE_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
POSE_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
FACE_MODEL_PATH = Path(__file__).parent / "face_landmarker.task"
POSE_MODEL_PATH = Path(__file__).parent / "pose_landmarker.task"


def _download(url: str, dest: Path) -> Path:
    if not dest.exists():
        print(f"Downloading {dest.name} ...")
        urllib.request.urlretrieve(url, dest)
    return dest


def ensure_face_model() -> Path:
    return _download(FACE_LANDMARKER_MODEL_URL, FACE_MODEL_PATH)


def ensure_pose_model() -> Path:
    return _download(POSE_LANDMARKER_MODEL_URL, POSE_MODEL_PATH)


# Anthropometric ratio from Farkas (1994) craniofacial norms:
# vertex-to-menton (true head height) ≈ 1.30 × trichion-to-menton (forehead-to-chin).
# Used to derive head_height_cm from the measured forehead-to-chin distance, since
# MediaPipe Face Mesh has no landmark at the crown/vertex.
HEAD_HEIGHT_RATIO = 1.30


# MediaPipe Face Mesh landmark indices.
# Eye contour points used for EAR (6-point formulation, per Soukupova & Cech 2016).
LEFT_EYE_EAR = [33, 160, 158, 133, 153, 144]   # outer, upper1, upper2, inner, lower2, lower1
RIGHT_EYE_EAR = [362, 385, 387, 263, 373, 380]

# Full eye outline for width/height measurement.
LEFT_EYE_CORNERS = (33, 133)   # outer, inner
LEFT_EYE_VERT = (159, 145)     # top, bottom
RIGHT_EYE_CORNERS = (362, 263)
RIGHT_EYE_VERT = (386, 374)

# Iris landmarks (require refine_landmarks=True). 4 points around each iris.
LEFT_IRIS = [468, 469, 470, 471, 472]   # 468 is center
RIGHT_IRIS = [473, 474, 475, 476, 477]

# Face dimensions (Face Mesh).
FACE_TOP = 10           # mid-forehead / glabella (NOT hairline)
FACE_BOTTOM = 152       # chin (menton)
FACE_LEFT = 234         # left cheek, zygomatic arch (bizygomatic width point)
FACE_RIGHT = 454        # right cheek

# Pose landmark indices for ear-to-ear (bitragion) width.
POSE_LEFT_EAR = 7
POSE_RIGHT_EAR = 8

# Nose.
NOSE_TOP = 168          # bridge between eyes
NOSE_BOTTOM = 2         # tip
NOSE_LEFT = 98          # left ala
NOSE_RIGHT = 327        # right ala

# Mouth. Outer-lip top/bottom so the "height" is upper-lip-top to lower-lip-bottom
# (i.e. the vertical extent of the mouth feature), not the inter-lip gap which
# collapses to ~0 when the mouth is closed.
MOUTH_LEFT = 61
MOUTH_RIGHT = 291
MOUTH_TOP = 0           # upper lip outer (cupid's bow)
MOUTH_BOTTOM = 17       # lower lip outer

# Anatomical reference: adult horizontal iris diameter is remarkably constant.
IRIS_DIAMETER_CM = 1.17

# EAR thresholds. Below threshold for >= MIN_FRAMES = blink.
EAR_THRESHOLD = 0.21
EAR_CONSECUTIVE_FRAMES = 2   # at the sub-sampled rate


@dataclass
class DimensionAccumulator:
    """Running mean of per-frame pixel measurements, later scaled to cm."""
    values: list[float] = field(default_factory=list)

    def add(self, v: float) -> None:
        if v > 0 and math.isfinite(v):
            self.values.append(v)

    def median(self) -> float:
        return float(np.median(self.values)) if self.values else 0.0


@dataclass
class VideoResult:
    video: str
    duration_sec: float
    processed_frames: int
    blinks: int
    blink_rate_per_sec: float
    # All in cm.
    face_height: float          # forehead (landmark 10) to chin — measured
    face_width: float           # cheek-to-cheek (bizygomatic) — measured
    head_height: float          # derived: face_height * 1.30 (vertex-to-menton)
    ear_to_ear: float           # measured via MediaPipe Pose (bitragion)
    left_eye_width: float
    left_eye_height: float
    right_eye_width: float
    right_eye_height: float
    nose_length: float
    nose_width: float
    mouth_width: float
    mouth_height: float
    iris_px: float


def _pt(landmarks, idx: int, w: int, h: int) -> np.ndarray:
    lm = landmarks[idx]
    return np.array([lm.x * w, lm.y * h], dtype=np.float32)


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def eye_aspect_ratio(landmarks, indices: list[int], w: int, h: int) -> float:
    p1, p2, p3, p4, p5, p6 = [_pt(landmarks, i, w, h) for i in indices]
    # EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    vert = _dist(p2, p6) + _dist(p3, p5)
    horiz = _dist(p1, p4)
    return vert / (2.0 * horiz) if horiz > 0 else 0.0


def iris_diameter_px(landmarks, iris_indices: list[int], w: int, h: int) -> float:
    # Horizontal diameter: distance between landmarks 1 and 3 around the iris
    # (left/right extremes of the iris circle).
    left = _pt(landmarks, iris_indices[1], w, h)
    right = _pt(landmarks, iris_indices[3], w, h)
    return _dist(left, right)


def process_video(path: Path, sample_fps: float) -> VideoResult:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / src_fps if src_fps > 0 else 0.0

    frame_stride = max(1, int(round(src_fps / sample_fps)))

    face_model_path = ensure_face_model()
    face_options = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(face_model_path)),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    face_mesh = mp_vision.FaceLandmarker.create_from_options(face_options)

    pose_model_path = ensure_pose_model()
    pose_options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(pose_model_path)),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    pose = mp_vision.PoseLandmarker.create_from_options(pose_options)

    blink_count = 0
    closed_streak = 0

    iris_px = DimensionAccumulator()
    face_h_px = DimensionAccumulator()
    face_w_px = DimensionAccumulator()
    l_eye_w = DimensionAccumulator()
    l_eye_h = DimensionAccumulator()
    r_eye_w = DimensionAccumulator()
    r_eye_h = DimensionAccumulator()
    nose_len = DimensionAccumulator()
    nose_w = DimensionAccumulator()
    mouth_w = DimensionAccumulator()
    mouth_h = DimensionAccumulator()
    ear_px = DimensionAccumulator()

    processed = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_stride != 0:
            frame_idx += 1
            continue
        frame_idx += 1
        processed += 1

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int((frame_idx - 1) / src_fps * 1000) if src_fps > 0 else processed
        result = face_mesh.detect_for_video(mp_image, timestamp_ms)
        if not result.face_landmarks:
            # Losing the face resets the closed-eye streak so we don't
            # spuriously count a blink on re-acquisition.
            closed_streak = 0
            continue

        lms = result.face_landmarks[0]

        # --- Blink detection ---
        left_ear = eye_aspect_ratio(lms, LEFT_EYE_EAR, w, h)
        right_ear = eye_aspect_ratio(lms, RIGHT_EYE_EAR, w, h)
        ear = (left_ear + right_ear) / 2.0

        if ear < EAR_THRESHOLD:
            closed_streak += 1
        else:
            if closed_streak >= EAR_CONSECUTIVE_FRAMES:
                blink_count += 1
            closed_streak = 0

        # --- Dimensions (in pixels) ---
        iris_px.add(iris_diameter_px(lms, LEFT_IRIS, w, h))
        iris_px.add(iris_diameter_px(lms, RIGHT_IRIS, w, h))

        face_h_px.add(_dist(_pt(lms, FACE_TOP, w, h), _pt(lms, FACE_BOTTOM, w, h)))
        face_w_px.add(_dist(_pt(lms, FACE_LEFT, w, h), _pt(lms, FACE_RIGHT, w, h)))

        l_eye_w.add(_dist(_pt(lms, LEFT_EYE_CORNERS[0], w, h),
                          _pt(lms, LEFT_EYE_CORNERS[1], w, h)))
        l_eye_h.add(_dist(_pt(lms, LEFT_EYE_VERT[0], w, h),
                          _pt(lms, LEFT_EYE_VERT[1], w, h)))
        r_eye_w.add(_dist(_pt(lms, RIGHT_EYE_CORNERS[0], w, h),
                          _pt(lms, RIGHT_EYE_CORNERS[1], w, h)))
        r_eye_h.add(_dist(_pt(lms, RIGHT_EYE_VERT[0], w, h),
                          _pt(lms, RIGHT_EYE_VERT[1], w, h)))

        nose_len.add(_dist(_pt(lms, NOSE_TOP, w, h), _pt(lms, NOSE_BOTTOM, w, h)))
        nose_w.add(_dist(_pt(lms, NOSE_LEFT, w, h), _pt(lms, NOSE_RIGHT, w, h)))

        mouth_w.add(_dist(_pt(lms, MOUTH_LEFT, w, h), _pt(lms, MOUTH_RIGHT, w, h)))
        mouth_h.add(_dist(_pt(lms, MOUTH_TOP, w, h), _pt(lms, MOUTH_BOTTOM, w, h)))

        # --- Ear-to-ear via Pose (separate model, same frame) ---
        pose_result = pose.detect_for_video(mp_image, timestamp_ms)
        if pose_result.pose_landmarks:
            plms = pose_result.pose_landmarks[0]
            ear_px.add(_dist(_pt(plms, POSE_LEFT_EAR, w, h),
                             _pt(plms, POSE_RIGHT_EAR, w, h)))

    cap.release()
    face_mesh.close()
    pose.close()

    # Handle a trailing blink that never reopened before video end.
    if closed_streak >= EAR_CONSECUTIVE_FRAMES:
        blink_count += 1

    iris_median_px = iris_px.median()
    if iris_median_px <= 0:
        # No iris detected anywhere: cannot calibrate. Report pixels.
        scale = float("nan")
    else:
        scale = IRIS_DIAMETER_CM / iris_median_px   # cm per pixel

    def to_cm(acc: DimensionAccumulator) -> float:
        return acc.median() * scale if scale == scale else 0.0

    blink_rate = blink_count / duration if duration > 0 else 0.0
    face_height_cm = to_cm(face_h_px)

    return VideoResult(
        video=str(path),
        duration_sec=duration,
        processed_frames=processed,
        blinks=blink_count,
        blink_rate_per_sec=blink_rate,
        face_height=face_height_cm,
        face_width=to_cm(face_w_px),
        head_height=face_height_cm * HEAD_HEIGHT_RATIO,
        ear_to_ear=to_cm(ear_px),
        left_eye_width=to_cm(l_eye_w),
        left_eye_height=to_cm(l_eye_h),
        right_eye_width=to_cm(r_eye_w),
        right_eye_height=to_cm(r_eye_h),
        nose_length=to_cm(nose_len),
        nose_width=to_cm(nose_w),
        mouth_width=to_cm(mouth_w),
        mouth_height=to_cm(mouth_h),
        iris_px=iris_median_px,
    )


def discover_videos(videos_dir: Path) -> list[Path]:
    exts = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
    return sorted(p for p in videos_dir.rglob("*") if p.suffix.lower() in exts)


def print_report(r: VideoResult) -> None:
    h = int(r.duration_sec // 3600)
    m = int((r.duration_sec % 3600) // 60)
    s = int(r.duration_sec % 60)
    print(f"\n=== {r.video} ===")
    print(f"Duration: {h:02d}:{m:02d}:{s:02d}  |  frames analyzed: {r.processed_frames}")
    print(f"Blinks: {r.blinks}   Rate: {r.blink_rate_per_sec:.4f} blinks/sec "
          f"({r.blink_rate_per_sec * 60:.1f} blinks/min)")
    print(f"Calibration: iris = {r.iris_px:.1f} px -> {IRIS_DIAMETER_CM} cm")
    print("Dimensions (cm):")
    print(f"  Face     forehead-to-chin: {r.face_height:.2f}   cheek-to-cheek: {r.face_width:.2f}")
    print(f"  Head     head-to-chin (derived, x{HEAD_HEIGHT_RATIO}): {r.head_height:.2f}")
    print(f"  Head     ear-to-ear (Pose): {r.ear_to_ear:.2f}")
    print(f"  L eye    W: {r.left_eye_width:.2f}   H: {r.left_eye_height:.2f}")
    print(f"  R eye    W: {r.right_eye_width:.2f}   H: {r.right_eye_height:.2f}")
    print(f"  Nose     L: {r.nose_length:.2f}   W: {r.nose_width:.2f}")
    print(f"  Mouth    W: {r.mouth_width:.2f}   H: {r.mouth_height:.2f}")


def write_csv(results: Iterable[VideoResult], out_path: Path) -> None:
    fields = [
        "video", "duration_sec", "processed_frames", "blinks", "blink_rate_per_sec",
        "forehead_to_chin_cm", "cheek_to_cheek_cm",
        "head_height_cm_derived", "ear_to_ear_cm_pose",
        "left_eye_width_cm", "left_eye_height_cm",
        "right_eye_width_cm", "right_eye_height_cm",
        "nose_length_cm", "nose_width_cm",
        "mouth_width_cm", "mouth_height_cm",
        "iris_px",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        for r in results:
            writer.writerow([
                r.video, f"{r.duration_sec:.2f}", r.processed_frames,
                r.blinks, f"{r.blink_rate_per_sec:.6f}",
                f"{r.face_height:.3f}", f"{r.face_width:.3f}",
                f"{r.head_height:.3f}", f"{r.ear_to_ear:.3f}",
                f"{r.left_eye_width:.3f}", f"{r.left_eye_height:.3f}",
                f"{r.right_eye_width:.3f}", f"{r.right_eye_height:.3f}",
                f"{r.nose_length:.3f}", f"{r.nose_width:.3f}",
                f"{r.mouth_width:.3f}", f"{r.mouth_height:.3f}",
                f"{r.iris_px:.2f}",
            ])


def aggregate(results: list[VideoResult]) -> VideoResult:
    """Combine per-file results into a single session-level result.

    Blink rate uses total blinks / total duration. Dimensions use the median of
    per-file medians, which is reasonable when each file already contributes
    thousands of frames.
    """
    total_dur = sum(r.duration_sec for r in results)
    total_frames = sum(r.processed_frames for r in results)
    total_blinks = sum(r.blinks for r in results)
    rate = total_blinks / total_dur if total_dur > 0 else 0.0

    def med(attr: str) -> float:
        vals = [getattr(r, attr) for r in results if getattr(r, attr) > 0]
        return float(np.median(vals)) if vals else 0.0

    return VideoResult(
        video=f"<aggregate of {len(results)} files>",
        duration_sec=total_dur,
        processed_frames=total_frames,
        blinks=total_blinks,
        blink_rate_per_sec=rate,
        face_height=med("face_height"),
        face_width=med("face_width"),
        head_height=med("head_height"),
        ear_to_ear=med("ear_to_ear"),
        left_eye_width=med("left_eye_width"),
        left_eye_height=med("left_eye_height"),
        right_eye_width=med("right_eye_width"),
        right_eye_height=med("right_eye_height"),
        nose_length=med("nose_length"),
        nose_width=med("nose_width"),
        mouth_width=med("mouth_width"),
        mouth_height=med("mouth_height"),
        iris_px=med("iris_px"),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", type=Path, help="Path to a single video file.")
    src.add_argument("--videos-dir", type=Path, help="Directory of videos to process.")
    ap.add_argument("--sample-fps", type=float, default=5.0,
                    help="Frames per second to sample (default 5). Lower = faster.")
    ap.add_argument("--aggregate", action="store_true",
                    help="Also report a session-level total across all videos.")
    ap.add_argument("--out", type=Path, default=Path("results.csv"),
                    help="Output CSV path.")
    args = ap.parse_args()

    if args.video:
        videos = [args.video]
    else:
        videos = discover_videos(args.videos_dir)
        if not videos:
            raise SystemExit(f"No videos found in {args.videos_dir}")

    results = []
    for v in videos:
        print(f"Processing {v} ...")
        r = process_video(v, args.sample_fps)
        print_report(r)
        results.append(r)

    if args.aggregate and len(results) > 1:
        session = aggregate(results)
        print("\n" + "=" * 60)
        print("SESSION TOTAL")
        print("=" * 60)
        print_report(session)
        results.append(session)

    write_csv(results, args.out)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()