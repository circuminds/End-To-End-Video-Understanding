import argparse
import json
import subprocess
from pathlib import Path
import pandas as pd

def run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return p.returncode == 0, p.stderr

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def extract_thumb(video_path: Path, ts: float, out_path: Path, thumb_w: int) -> bool:
    # robust JPEG extraction
    vf = f"scale={thumb_w}:-2,format=yuvj420p"
    ok, _ = run([
        "ffmpeg", "-y",
        "-ss", f"{ts:.3f}",
        "-i", str(video_path),
        "-frames:v", "1",
        "-vf", vf,
        "-q:v", "3",
        "-strict", "-2",
        str(out_path),
    ])
    if ok and out_path.exists() and out_path.stat().st_size > 0:
        return True

    # PNG fallback (often succeeds when JPG encoder/pixfmt is picky)
    png_path = out_path.with_suffix(".png")
    ok2, _ = run([
        "ffmpeg", "-y",
        "-ss", f"{ts:.3f}",
        "-i", str(video_path),
        "-frames:v", "1",
        "-vf", f"scale={thumb_w}:-2",
        str(png_path),
    ])
    if ok2 and png_path.exists() and png_path.stat().st_size > 0:
        # If PNG succeeded, keep PNG and update caller via return True
        return True

    return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_id", required=True)
    ap.add_argument("--processed_dir", default="data/processed")
    ap.add_argument("--thumb_w", type=int, default=640)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.processed_dir) / args.video_id
    meta = json.loads((out_dir / "metadata.json").read_text())
    video_path = Path(meta["raw_video_path"])

    # duration may be string; convert safely
    dur = meta.get("ffprobe", {}).get("duration", None)
    duration = float(dur) if dur is not None else None
    if duration is None or duration <= 0:
        raise ValueError("Could not read valid duration from metadata.json")

    moments_path = out_dir / "moments.parquet"
    thumbs_dir = out_dir / "thumbs"
    thumbs_dir.mkdir(parents=True, exist_ok=True)

    moments = pd.read_parquet(moments_path)

    # point to expected jpg names (we may also generate png fallback if needed)
    moments["thumb_path"] = [
        str(thumbs_dir / f"moment_{int(mid):05d}.jpg")
        for mid in moments["moment_id"].astype(int).tolist()
    ]

    # regenerate missing (or all)
    targets = []
    for r in moments.itertuples(index=False):
        out_path = Path(r.thumb_path)
        if args.force or (not out_path.exists()):
            # midpoint, clamped to valid decode range
            raw_ts = (float(r.start) + float(r.end)) / 2.0
            ts = clamp(raw_ts, 0.0, max(0.0, duration - 0.25))
            targets.append((int(r.moment_id), ts, out_path))

    print(f"Video duration: {duration:.3f}s")
    print(f"Thumbs to create: {len(targets)}")

    ok_count = 0
    for mid, ts, out_path in targets:
        # try ts, then a bit earlier
        success = extract_thumb(video_path, ts, out_path, args.thumb_w)
        if not success:
            success = extract_thumb(video_path, max(0.0, ts - 1.0), out_path, args.thumb_w)

        if success:
            ok_count += 1
            # if PNG fallback was used, record that in thumb_path
            png_path = out_path.with_suffix(".png")
            if png_path.exists() and (not out_path.exists() or out_path.stat().st_size == 0):
                # switch to PNG
                moments.loc[moments["moment_id"] == mid, "thumb_path"] = str(png_path)
            print(f"[OK] moment {mid:02d} @ {ts:.3f}s")
        else:
            print(f"[FAIL] moment {mid:02d} @ {ts:.3f}s (and fallback)")

    moments.to_parquet(moments_path, index=False)
    print(f"Created {ok_count}/{len(targets)} thumbs")
    print("Updated moments:", moments_path)

if __name__ == "__main__":
    main()
