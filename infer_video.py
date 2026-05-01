"""
X3D-M Video Inference Pipeline
==============================
Loads a real video file, preprocesses it following the Kinetics-400 evaluation
protocol (temporal center-crop, spatial resize + center-crop, ImageNet/Kinetics
normalisation), and runs inference through the from-scratch X3D-M model using
the native C convolutional kernel.

Usage examples:
    # Single video
    python infer_video.py videos/writing.mp4

    # All videos in a folder
    python infer_video.py videos/

    # With profiling
    python infer_video.py videos/brushing_teeth.mp4 --profile

    # Show the 16 sampled frames (saved as a grid image)
    python infer_video.py videos/diving.mp4 --save-frames

    # Custom settings
    python infer_video.py videos/biking.mp4 --num-frames 16 --sampling-rate 6.0 --top-k 10
"""

from __future__ import annotations
import argparse
import time
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np

# ── X3D-M scratch imports ──────────────────────────────────────────
from scratch.models.x3d_m import X3D_M
from scratch.load_weights import load_pretrained_numpy
from scratch.ops.conv3d import set_conv3d_method, get_conv3d_method, VALID_METHODS

# ── Constants ──────────────────────────────────────────────────────
# These match PyTorchVideo / SlowFast's X3D-M evaluation protocol.
NUM_FRAMES    = 16          # T dimension — the model always expects 16 frames
SAMPLING_FPS  = 6.0         # Effective sampling rate: 30 fps / stride 5 = 6 fps
SHORT_SIDE    = 256         # Resize shorter side to 256 before cropping
CROP_SIZE     = 224         # Center-crop to 224 x 224

# Kinetics-400 / PyTorchVideo normalization (applied AFTER scaling to [0,1])
MEAN = np.array([0.45, 0.45, 0.45], dtype=np.float32)
STD  = np.array([0.225, 0.225, 0.225], dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════
#  STEP 1 — Read video metadata
# ═══════════════════════════════════════════════════════════════════

def get_video_info(path: str) -> Tuple[float, int, int, int]:
    """
    Open the video and return (fps, total_frames, width, height).
    We need fps to compute the correct temporal stride, and total_frames
    to know how much video we have to work with.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    fps          = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return fps, total_frames, width, height


# ═══════════════════════════════════════════════════════════════════
#  STEP 2 — Compute which frame indices to sample
# ═══════════════════════════════════════════════════════════════════

def compute_frame_indices(
    total_frames: int,
    fps: float,
    num_frames: int = NUM_FRAMES,
    target_fps: float = SAMPLING_FPS,
) -> List[int]:
    """
    Determine exactly which raw frame indices to read from the video.

    The logic:
    1. Compute stride = round(native_fps / target_fps).
       This adapts to any source FPS so we always sample at ~6 fps effective.
       - 30 fps video → stride 5  (every 5th frame)
       - 24 fps video → stride 4
       - 60 fps video → stride 10
       - 15 fps video → stride 2 (but note: fewer than ideal raw frames)

    2. The span of raw frames we need is: (num_frames - 1) * stride + 1.
       For 16 frames at stride 5 that's 76 frames (≈2.53 s at 30 fps).

    3. If the video is long enough, we CENTER the span in the video
       (standard evaluation protocol). If the video is too short,
       we reduce the stride until the frames fit, or as a last resort
       sample uniformly across whatever frames exist.

    Returns a list of `num_frames` integer indices into the raw video.
    """
    stride = max(1, round(fps / target_fps))
    span   = (num_frames - 1) * stride + 1

    if total_frames >= span:
        # ── Normal case: video is long enough ──
        # Center the sampling window in the video.
        start = (total_frames - span) // 2
        indices = [start + i * stride for i in range(num_frames)]

    elif total_frames >= num_frames:
        # ── Video is shorter than ideal but has at least 16 frames ──
        # Shrink the stride so all 16 frames fit.
        stride = (total_frames - 1) // (num_frames - 1)
        span   = (num_frames - 1) * stride + 1
        start  = (total_frames - span) // 2
        indices = [start + i * stride for i in range(num_frames)]

    else:
        # ── Very short video (< 16 frames) ──
        # Sample uniformly and repeat the last frame to fill 16 slots.
        indices = list(range(total_frames))
        while len(indices) < num_frames:
            indices.append(total_frames - 1)

    return indices


# ═══════════════════════════════════════════════════════════════════
#  STEP 3 — Read the selected frames from the video file
# ═══════════════════════════════════════════════════════════════════

def read_frames(path: str, indices: List[int]) -> List[np.ndarray]:
    """
    Read specific frames from the video by seeking to each index.

    Returns a list of BGR uint8 images (OpenCV native format).
    We convert to RGB in the preprocessing step.
    """
    cap = cv2.VideoCapture(path)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            # If seeking fails (common near end of file), duplicate the last
            # successfully read frame.
            if frames:
                frames.append(frames[-1].copy())
            else:
                raise RuntimeError(f"Failed to read frame {idx} from {path}")
        else:
            frames.append(frame)
    cap.release()
    return frames


# ═══════════════════════════════════════════════════════════════════
#  STEP 4 — Spatial preprocessing (resize + center crop)
# ═══════════════════════════════════════════════════════════════════

def resize_short_side(frame: np.ndarray, short_side: int = SHORT_SIDE) -> np.ndarray:
    """
    Resize the frame so that its shorter side equals `short_side` pixels,
    keeping the aspect ratio intact.

    Example: a 1920×1080 frame → shorter side is 1080, scale factor = 256/1080
             → new size ≈ 456×256.  A 640×480 frame → 342×256.

    We use BILINEAR interpolation (cv2.INTER_LINEAR) which is the standard
    for evaluation.  During training you'd use random-area crops instead,
    but for inference / demo, bilinear resize + center crop is correct.
    """
    h, w = frame.shape[:2]
    if h <= w:
        # Height is the short side
        new_h = short_side
        new_w = int(round(w * short_side / h))
    else:
        # Width is the short side
        new_w = short_side
        new_h = int(round(h * short_side / w))
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def center_crop(frame: np.ndarray, crop_size: int = CROP_SIZE) -> np.ndarray:
    """
    Take the center crop_size × crop_size patch from the frame.

    After resizing the short side to 256, the frame is at least 256 on
    both sides.  We cut a 224×224 window from the exact center, discarding
    16 pixels of border on the short side and more on the long side.

    This keeps the main subject (usually centered in Kinetics videos)
    and removes edge noise.
    """
    h, w = frame.shape[:2]
    top  = (h - crop_size) // 2
    left = (w - crop_size) // 2
    return frame[top : top + crop_size, left : left + crop_size]


# ═══════════════════════════════════════════════════════════════════
#  STEP 5 — Pixel normalization and tensor assembly
# ═══════════════════════════════════════════════════════════════════

def preprocess_frames(frames: List[np.ndarray]) -> np.ndarray:
    """
    Convert a list of 16 BGR uint8 frames into the (1, 3, 16, 224, 224)
    float32 tensor the model expects.

    Per-frame pipeline:
        1. BGR → RGB               (OpenCV loads BGR; model expects RGB)
        2. Resize shorter side → 256
        3. Center crop → 224×224
        4. uint8 [0,255] → float32 [0.0, 1.0]
        5. Normalize: (pixel − mean) / std
           mean = [0.45, 0.45, 0.45],  std = [0.225, 0.225, 0.225]
           (These are the standard Kinetics-400 values used by PyTorchVideo.)

    After processing all frames:
        6. Stack → (16, 224, 224, 3)       — T, H, W, C
        7. Transpose → (3, 16, 224, 224)   — C, T, H, W
        8. Add batch dim → (1, 3, 16, 224, 224)
    """
    processed = []
    for frame in frames:
        # 1. BGR → RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 2. Resize shorter side to 256
        frame = resize_short_side(frame, SHORT_SIDE)

        # 3. Center crop 224×224
        frame = center_crop(frame, CROP_SIZE)

        # 4. Scale to [0, 1]
        frame = frame.astype(np.float32) / 255.0

        # 5. Normalize with mean and std
        #    Shape of frame is (224, 224, 3), MEAN/STD are (3,),
        #    broadcasting handles the rest.
        frame = (frame - MEAN) / STD

        processed.append(frame)

    # 6. Stack: list of (224, 224, 3) → (16, 224, 224, 3)
    clip = np.stack(processed, axis=0)

    # 7. Transpose: (T, H, W, C) → (C, T, H, W)
    clip = clip.transpose(3, 0, 1, 2)

    # 8. Add batch dimension: (C, T, H, W) → (1, C, T, H, W)
    clip = np.expand_dims(clip, axis=0)

    return clip


# ═══════════════════════════════════════════════════════════════════
#  STEP 6 — Load Kinetics-400 labels
# ═══════════════════════════════════════════════════════════════════

def load_labels(path: str = "kinetics400_labels.txt") -> List[str]:
    """
    Load the 400 class names from the text file.
    Each line is one class, order matches model output indices.
    """
    with open(path, "r") as f:
        labels = [line.strip() for line in f if line.strip()]
    assert len(labels) == 400, f"Expected 400 labels, got {len(labels)}"
    return labels


# ═══════════════════════════════════════════════════════════════════
#  STEP 7 — Softmax and top-k prediction
# ═══════════════════════════════════════════════════════════════════

def softmax(logits: np.ndarray) -> np.ndarray:
    """
    Convert raw logits to probabilities.

    The model outputs 400 raw scores (logits). Softmax normalizes them
    into a probability distribution that sums to 1.0, making the numbers
    interpretable as confidence percentages.

    We subtract the max for numerical stability (prevents overflow in exp).
    """
    x = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def top_k_predictions(
    logits: np.ndarray,
    labels: List[str],
    k: int = 5,
) -> List[Tuple[str, float]]:
    """
    Return the top-k (label, probability) pairs from model output.
    """
    probs = softmax(logits[0])             # (400,)
    top_indices = np.argsort(probs)[::-1][:k]
    return [(labels[i], float(probs[i])) for i in top_indices]


# ═══════════════════════════════════════════════════════════════════
#  STEP 8 — (Optional) Save sampled frames as a debug grid
# ═══════════════════════════════════════════════════════════════════

def save_frame_grid(
    frames: List[np.ndarray],
    output_path: str,
    cols: int = 4,
) -> None:
    """
    Save the 16 sampled frames as a 4×4 grid image for visual inspection.
    This lets you verify that the correct frames were sampled and the
    spatial crop looks reasonable.
    """
    n = len(frames)
    rows = (n + cols - 1) // cols

    # Resize all frames to a common small size for the grid
    thumb_h, thumb_w = 160, 160
    thumbs = []
    for f in frames:
        # Convert BGR→RGB for display, resize short side, center crop
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        rgb = resize_short_side(rgb, thumb_h)
        rgb = center_crop(rgb, min(thumb_h, thumb_w))
        rgb = cv2.resize(rgb, (thumb_w, thumb_h))
        thumbs.append(rgb)

    # Pad with black frames if needed
    while len(thumbs) < rows * cols:
        thumbs.append(np.zeros((thumb_h, thumb_w, 3), dtype=np.uint8))

    grid_rows = []
    for r in range(rows):
        row_imgs = thumbs[r * cols : (r + 1) * cols]
        grid_rows.append(np.concatenate(row_imgs, axis=1))
    grid = np.concatenate(grid_rows, axis=0)

    # Save as BGR for OpenCV
    grid_bgr = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, grid_bgr)
    print(f"  Frame grid saved: {output_path}")


# ═══════════════════════════════════════════════════════════════════
#  Main inference pipeline
# ═══════════════════════════════════════════════════════════════════

def infer_video(
    video_path: str,
    model: X3D_M,
    labels: List[str],
    top_k: int = 5,
    save_frames: bool = False,
    target_fps: float = SAMPLING_FPS,
) -> List[Tuple[str, float]]:
    """
    End-to-end inference on a single video file.

    1. Read video metadata (fps, frame count, resolution)
    2. Compute temporal sampling indices
    3. Read the selected frames
    4. Preprocess → (1, 3, 16, 224, 224) tensor
    5. Forward pass through X3D-M
    6. Decode predictions
    """
    video_name = Path(video_path).stem
    print(f"\n{'─' * 60}")
    print(f"  VIDEO: {Path(video_path).name}")
    print(f"{'─' * 60}")

    # ── 1. Video metadata ──
    fps, total_frames, w, h = get_video_info(video_path)
    duration = total_frames / fps if fps > 0 else 0
    print(f"  Resolution : {w}×{h}")
    print(f"  FPS        : {fps:.2f}")
    print(f"  Frames     : {total_frames}")
    print(f"  Duration   : {duration:.2f}s")

    # ── 2. Temporal sampling ──
    indices = compute_frame_indices(total_frames, fps, NUM_FRAMES, target_fps)
    stride = indices[1] - indices[0] if len(indices) > 1 else 1
    effective_fps = fps / stride if stride > 0 else fps
    clip_duration = (indices[-1] - indices[0]) / fps if fps > 0 else 0
    print(f"  Stride     : {stride}  (every {stride}th raw frame)")
    print(f"  Eff. FPS   : {effective_fps:.1f} fps")
    print(f"  Clip span  : frames {indices[0]}–{indices[-1]}  ({clip_duration:.2f}s)")

    # ── 3. Read frames ──
    t0 = time.perf_counter()
    frames = read_frames(video_path, indices)
    read_ms = (time.perf_counter() - t0) * 1000
    print(f"  Read time  : {read_ms:.1f} ms")

    # ── 3b. (Optional) save frame grid for visual debugging ──
    if save_frames:
        grid_path = f"frame_grid_{video_name}.png"
        save_frame_grid(frames, grid_path)

    # ── 4. Preprocess ──
    t0 = time.perf_counter()
    tensor = preprocess_frames(frames)
    prep_ms = (time.perf_counter() - t0) * 1000
    print(f"  Preprocess : {prep_ms:.1f} ms")
    print(f"  Tensor     : {tensor.shape}  dtype={tensor.dtype}")
    print(f"  Value range: [{tensor.min():.3f}, {tensor.max():.3f}]")

    # ── 5. Forward pass ──
    print(f"\n  Running inference (method={get_conv3d_method()})...")
    t0 = time.perf_counter()
    logits = model.forward(tensor)
    infer_ms = (time.perf_counter() - t0) * 1000
    print(f"  Inference  : {infer_ms:.1f} ms  ({infer_ms/1000:.2f}s)")

    # ── 6. Decode predictions ──
    predictions = top_k_predictions(logits, labels, top_k)

    print(f"\n  Top-{top_k} predictions:")
    for rank, (label, prob) in enumerate(predictions, 1):
        bar = "█" * int(prob * 40)
        print(f"    {rank}. {label:<35s} {prob*100:6.2f}%  {bar}")

    return predictions


# ═══════════════════════════════════════════════════════════════════
#  CLI entry point
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="X3D-M Video Inference — from-scratch, no PyTorch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python infer_video.py videos/writing.mp4
  python infer_video.py videos/                       # all .mp4 in folder
  python infer_video.py videos/diving.mp4 --save-frames
  python infer_video.py videos/biking.mp4 --method native --top-k 10
        """,
    )
    parser.add_argument(
        "input",
        type=str,
        help="Path to a video file (.mp4) or a directory containing videos",
    )
    parser.add_argument(
        "--weights", "-w",
        type=str,
        default="weights/x3d_m_kinetics400.npz",
        help="Path to pretrained .npz weights (default: weights/x3d_m_kinetics400.npz)",
    )
    parser.add_argument(
        "--labels", "-l",
        type=str,
        default="kinetics400_labels.txt",
        help="Path to Kinetics-400 label file",
    )
    parser.add_argument(
        "--method", "-m",
        type=str,
        default="native",
        choices=list(VALID_METHODS),
        help="Convolution method (default: native — C backend)",
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=5,
        help="Number of top predictions to show (default: 5)",
    )
    parser.add_argument(
        "--sampling-rate",
        type=float,
        default=SAMPLING_FPS,
        help=f"Target sampling FPS (default: {SAMPLING_FPS})",
    )
    parser.add_argument(
        "--save-frames",
        action="store_true",
        help="Save a 4×4 grid of the 16 sampled frames for visual inspection",
    )

    args = parser.parse_args()

    # ── Set convolution method ──
    set_conv3d_method(args.method)

    # ── Resolve input path(s) ──
    input_path = Path(args.input)
    if input_path.is_dir():
        video_paths = sorted(input_path.glob("*.mp4"))
        if not video_paths:
            print(f"No .mp4 files found in {input_path}")
            return
        print(f"Found {len(video_paths)} video(s) in {input_path}/")
    elif input_path.is_file():
        video_paths = [input_path]
    else:
        print(f"Error: {args.input} is not a valid file or directory")
        return

    # ── Load model ──
    print("=" * 60)
    print("X3D-M Video Inference Pipeline")
    print(f"  Conv method : {get_conv3d_method()}")
    print(f"  Weights     : {args.weights}")
    print(f"  Sampling FPS: {args.sampling_rate}")
    print("=" * 60)

    print("\nLoading model and weights...")
    t0 = time.perf_counter()
    model = X3D_M(num_classes=400)
    load_pretrained_numpy(model, args.weights)
    model.eval()
    load_ms = (time.perf_counter() - t0) * 1000
    print(f"  Model loaded in {load_ms:.1f} ms")

    # ── Load labels ──
    labels = load_labels(args.labels)

    # ── Run inference on each video ──
    all_results = {}
    for vp in video_paths:
        preds = infer_video(
            str(vp), model, labels,
            top_k=args.top_k,
            save_frames=args.save_frames,
            target_fps=args.sampling_rate,
        )
        all_results[vp.name] = preds

    # ── Summary ──
    if len(video_paths) > 1:
        print(f"\n{'=' * 60}")
        print("SUMMARY")
        print(f"{'=' * 60}")
        for name, preds in all_results.items():
            top_label, top_prob = preds[0]
            print(f"  {name:<25s} → {top_label} ({top_prob*100:.1f}%)")


if __name__ == "__main__":
    main()
