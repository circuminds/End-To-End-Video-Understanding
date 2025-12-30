import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import faiss

def fmt_ts(seconds: float) -> str:
    s = int(seconds)
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:02d}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_id", required=True)
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--processed_dir", default="data/processed")
    args = ap.parse_args()

    out_dir = Path(args.processed_dir) / args.video_id
    emb_dir = out_dir / "embeddings"

    moments_path = out_dir / "moments.parquet"
    idx_path = emb_dir / "faiss_fused.index"
    q_path = emb_dir / "query_fused.npy"

    if not moments_path.exists(): raise FileNotFoundError(moments_path)
    if not idx_path.exists(): raise FileNotFoundError(idx_path)
    if not q_path.exists(): raise FileNotFoundError(q_path)

    moments = pd.read_parquet(moments_path)
    index = faiss.read_index(str(idx_path))
    qv = np.load(q_path).astype(np.float32)

    scores, ids = index.search(qv, args.top_k)

    print("\nTop fused results:")
    for rank, (idx, score) in enumerate(zip(ids[0], scores[0]), start=1):
        row = moments.iloc[int(idx)]
        snippet = str(row["text"][:180]).replace("\n", " ").strip()
        print(f"\n#{rank}  score={float(score):.4f}  {fmt_ts(float(row['start']))}–{fmt_ts(float(row['end']))}")
        print("thumb:", row.get("thumb_path", ""))
        print("text:", snippet + ("..." if len(str(row["text"])) > 180 else ""))

if __name__ == "__main__":
    main()
