import os
import cv2
import numpy as np
import shutil
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


def check_video(video_path):
    """
    Check an individual video: supports .webm and common formats.
    """
    try:
        # Explicitly specify FFMPEG backend for better .webm support
        cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)

        if not cap.isOpened():
            return video_path, False, "Codec error or unsupported format"

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            # Some .webm files cannot retrieve total frame count; attempt to read one frame directly
            ret, frame = cap.read()
            if not ret:
                return video_path, False, "Empty video or header error"
            total_frames = 1  # At least one frame exists

        # Sampling check: Start, Middle, and End
        check_indices = [0, total_frames // 2, max(0, total_frames - 1)]

        for idx in check_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()

            if not ret or frame is None:
                continue  # Some formats do not support seeking; try the next index

            # Check for numerical anomalies (NaN/Inf)
            if not np.isfinite(frame).all():
                return video_path, False, "Contains NaN or Inf"

            # Check for extreme pixel values (standard range: 0-255)
            if np.max(frame) > 255 or np.min(frame) < 0:
                return video_path, False, "Pixel values out of range"

            # Check for "dead" frames (pure black or extremely low variance)
            # Video preprocessing usually performs (x - mean) / std; zero variance leads to NaN
            if np.std(frame) < 1e-3:
                return video_path, False, "Static/Zero variance frame (Potential NaN source)"

        cap.release()
        return video_path, True, "OK"

    except Exception as e:
        return video_path, False, f"Runtime error: {str(e)}"


def clean_dataset(video_dir, quarantine_dir="quarantine_vids"):
    """Check for corrupted files in the dataset and move them to the quarantine directory."""
    # Supported extensions
    extensions = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.webm')
    all_videos = []
    for root, _, files in os.walk(video_dir):
        for f in files:
            if f.lower().endswith(extensions):
                all_videos.append(os.path.join(root, f))

    print(f"🚀 Scan started | Target: {len(all_videos)} videos")

    if not os.path.exists(quarantine_dir):
        os.makedirs(quarantine_dir)

    # Parallel processing
    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap(check_video, all_videos), total=len(all_videos)))

    # Process results
    bad_vids = [r for r in results if not r[1]]

    if bad_vids:
        print(f"\n⚠️ Found {len(bad_vids)} abnormal videos! Moving to quarantine...")
        for path, _, reason in bad_vids:
            # Keep original filename; prevent overwriting due to identical names
            dest = os.path.join(quarantine_dir, os.path.basename(path))
            try:
                shutil.move(path, dest)
            except:
                pass
        print(f"✅ Quarantine complete. See bad_videos.log for details.")
    else:
        print("\n✨ No abnormal videos found. The dataset is clean!")

    # Logging
    with open("bad_videos.log", "w") as f:
        for path, status, msg in results:
            if not status:
                f.write(f"{path} | {msg}\n")


if __name__ == "__main__":
    # Configure your paths
    # Original dataset path
    DATA_PATH = "/mnt/windows/Datasets/Something_Something_v2/20bn-something-something-v2"
    # Quarantine zone path
    QUARANTINE_PATH = "/mnt/windows/Datasets/Something_Something_v2/corrupted_data"

    # Start checking the dataset and handle corrupted files
    clean_dataset(DATA_PATH, QUARANTINE_PATH)