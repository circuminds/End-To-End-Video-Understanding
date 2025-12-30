import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

def run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n\nSTDERR:\n{p.stderr}")
    return p.stdout

def has_audio(video_path: Path) -> bool:
    out = run([
        "ffprobe", "-v", "error",
        "-select_streams", "a",
        "-show_entries", "stream=codec_type",
        "-of", "csv=p=0",
        str(video_path)
    ]).strip()
    return len(out) > 0

def get_duration(video_path: Path) -> float:
    out = run([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path)
    ]).strip()
    return float(out)


def sha1_of_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()[:12]

def ffprobe_metadata(video_path: Path) -> dict:
    out = run([
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,duration",
        "-of", "json",
        str(video_path),
    ])
    j = json.loads(out)
    stream = (j.get("streams") or [{}])[0]
    return {
        "width": stream.get("width"),
        "height": stream.get("height"),
        "r_frame_rate": stream.get("r_frame_rate"),
        "duration": stream.get("duration"),
    }

def extract_audio(video_path: Path, audio_out: Path):
    audio_out.parent.mkdir(parents=True, exist_ok=True)

    if has_audio(video_path):
        # Extract mono 16k wav
        run([
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vn",
            "-ac", "1",
            "-ar", "16000",
            str(audio_out)
        ])
        return

    # No audio stream → generate silent wav with same duration
    dur = max(0.1, get_duration(video_path))
    run([
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", "anullsrc=channel_layout=mono:sample_rate=16000",
        "-t", f"{dur:.3f}",
        str(audio_out)
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Path to an mp4/mkv video")
    ap.add_argument("--raw_dir", default="data/raw_videos")
    ap.add_argument("--processed_dir", default="data/processed")
    args = ap.parse_args()

    video_path = Path(args.video).expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    raw_dir = Path(args.raw_dir)
    processed_dir = Path(args.processed_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    video_id = sha1_of_file(video_path)
    out_dir = processed_dir / video_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Copy into raw_videos for reproducibility
    raw_copy = raw_dir / f"{video_id}{video_path.suffix.lower()}"
    if not raw_copy.exists():
        raw_copy.write_bytes(video_path.read_bytes())

    audio_path = out_dir / "audio.wav"
    if not audio_path.exists():
        extract_audio(raw_copy, audio_path)

    meta = {
        "video_id": video_id,
        "original_name": video_path.name,
        "raw_video_path": str(raw_copy),
        "audio_path": str(audio_path),
        "ffprobe": ffprobe_metadata(raw_copy),
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))

if __name__ == "__main__":
    main()
