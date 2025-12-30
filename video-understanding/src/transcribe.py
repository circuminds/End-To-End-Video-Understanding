import argparse
import json
from pathlib import Path
import pandas as pd
from faster_whisper import WhisperModel
from tqdm import tqdm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_id", required=True)
    ap.add_argument("--processed_dir", default="data/processed")
    ap.add_argument("--model", default="small")  # good default for quality/speed
    ap.add_argument("--device", default="auto")  # auto, cpu, cuda
    ap.add_argument("--compute_type", default="auto")  # auto, int8, float16
    ap.add_argument("--language", default=None)  # e.g., "en"
    ap.add_argument("--beam_size", type=int, default=5)
    ap.add_argument("--vad_filter", action="store_true")  # helps if lots of silence
    args = ap.parse_args()

    out_dir = Path(args.processed_dir) / args.video_id
    audio_path = out_dir / "audio.wav"
    if not audio_path.exists():
        raise FileNotFoundError(audio_path)

    transcript_json = out_dir / "transcript_segments.json"
    transcript_parquet = out_dir / "transcript_segments.parquet"

    # Load model
    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)

    segments, info = model.transcribe(
        str(audio_path),
        language=args.language,
        beam_size=args.beam_size,
        vad_filter=args.vad_filter,
        word_timestamps=False,  # we’ll add word-level later as an optional upgrade
    )

    rows = []
    for seg in segments:
        rows.append({
            "segment_id": int(seg.id),
            "start": float(seg.start),
            "end": float(seg.end),
            "text": seg.text.strip(),
        })

    meta = {
        "video_id": args.video_id,
        "audio_path": str(audio_path),
        "model": args.model,
        "device": args.device,
        "compute_type": args.compute_type,
        "language": info.language,
        "language_probability": float(info.language_probability),
        "duration": float(info.duration),
    }

    # Save
    transcript_json.write_text(json.dumps({"meta": meta, "segments": rows}, indent=2))
    df = pd.DataFrame(rows)
    df.to_parquet(transcript_parquet, index=False)

    print("Saved:")
    print(" -", transcript_json)
    print(" -", transcript_parquet)
    print("Detected language:", meta["language"], "p=", meta["language_probability"])
    print("Segments:", len(df))
    print("First 3 segments:\n", df.head(3).to_string(index=False))

if __name__ == "__main__":
    main()
