import argparse
import subprocess
from pathlib import Path
import pandas as pd

def run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n\nSTDERR:\n{p.stderr}")
    return p.stdout

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_id", required=True)
    ap.add_argument("--processed_dir", default="data/processed")
    ap.add_argument("--thumb_w", type=int, default=640)  # downscale for robustness/speed
    args = ap.parse_args()

    out_dir = Path(args.processed_dir) / args.video_id
    meta_path = out_dir / "metadata.json"
    moments_path = out_dir / "moments.parquet"
    if not meta_path.exists() or not moments_path.exists():
        raise FileNotFoundError("Need metadata.json and moments.parquet first.")

    import json
    meta = json.loads(meta_path.read_text())
    video_path = Path(meta["raw_video_path"])

    thumbs_dir = out_dir / "thumbs"
    thumbs_dir.mkdir(parents=True, exist_ok=True)

    moments = pd.read_parquet(moments_path)
    thumb_paths = []

    for row in moments.itertuples(index=False):
        mid = int(row.moment_id)
        ts = (float(row.start) + float(row.end)) / 2.0
        thumb_path = thumbs_dir / f"moment_{mid:05d}.jpg"

        if not thumb_path.exists():
            # robust JPEG extraction:
            # -ss before input for fast seek
            # scale down + force JPEG-friendly pixel format
            vf = f"scale={args.thumb_w}:-2,format=yuvj420p"
            run([
                "ffmpeg", "-y",
                "-ss", f"{ts:.3f}",
                "-i", str(video_path),
                "-frames:v", "1",
                "-vf", vf,
                "-q:v", "3",
                "-strict", "-2",
                str(thumb_path),
            ])

        thumb_paths.append(str(thumb_path))

    moments["thumb_path"] = thumb_paths
    moments.to_parquet(moments_path, index=False)
    print("Updated:", moments_path)
    if len(moments) == 0:
        print("No moments found (moments.parquet is empty). Skipping thumbnail extraction.")
        return
    print("Example thumb:", moments["thumb_path"].iloc[0])

if __name__ == "__main__":
    main()
