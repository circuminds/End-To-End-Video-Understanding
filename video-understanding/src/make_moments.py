import argparse
from pathlib import Path
import pandas as pd

def make_moments(df: pd.DataFrame, window_s: float, stride_s: float, min_chars: int):
    """
    Build moments by sliding a time window over transcript segments.
    Each moment contains transcript text whose segment timestamps overlap [t, t+window_s].
    """
    if df.empty:
        return pd.DataFrame(columns=["moment_id","start","end","text","n_segments"])

    t0 = float(df["start"].min())
    t1 = float(df["end"].max())

    moments = []
    moment_id = 0
    t = t0

    # Pre-sort for speed
    df = df.sort_values("start").reset_index(drop=True)

    i = 0
    n = len(df)

    while t < t1:
        start = t
        end = t + window_s

        # advance i to first segment that might overlap
        while i < n and float(df.loc[i, "end"]) < start:
            i += 1

        j = i
        texts = []
        count = 0
        # collect overlapping segments
        while j < n and float(df.loc[j, "start"]) <= end:
            seg_start = float(df.loc[j, "start"])
            seg_end = float(df.loc[j, "end"])
            if seg_end >= start:  # overlap
                txt = str(df.loc[j, "text"]).strip()
                if txt:
                    texts.append(txt)
                    count += 1
            j += 1

        text = " ".join(texts).strip()

        if len(text) >= min_chars:
            moments.append({
                "moment_id": moment_id,
                "start": float(start),
                "end": float(end),
                "text": text,
                "n_segments": int(count),
            })
            moment_id += 1

        t += stride_s

    return pd.DataFrame(moments)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_id", required=True)
    ap.add_argument("--processed_dir", default="data/processed")
    ap.add_argument("--window_s", type=float, default=20.0)   # good default
    ap.add_argument("--stride_s", type=float, default=10.0)   # overlap improves recall
    ap.add_argument("--min_chars", type=int, default=40)
    args = ap.parse_args()

    out_dir = Path(args.processed_dir) / args.video_id
    transcript_path = out_dir / "transcript_segments.parquet"
    if not transcript_path.exists():
        raise FileNotFoundError(transcript_path)

    df = pd.read_parquet(transcript_path)
    moments = make_moments(df, args.window_s, args.stride_s, args.min_chars)

    out_path = out_dir / "moments.parquet"
    moments.to_parquet(out_path, index=False)

    print("Saved:", out_path)
    print("Moments:", len(moments))
    print("Preview:\n", moments.head(5).to_string(index=False))

if __name__ == "__main__":
    main()
