import argparse
from pathlib import Path
import numpy as np
import faiss

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_id", required=True)
    ap.add_argument("--processed_dir", default="data/processed")
    ap.add_argument("--index_type", default="hnsw", choices=["flat", "hnsw"])
    ap.add_argument("--hnsw_m", type=int, default=32)
    args = ap.parse_args()

    out_dir = Path(args.processed_dir) / args.video_id
    X_path = out_dir / "embeddings" / "text_clip.npy"
    if not X_path.exists():
        raise FileNotFoundError(X_path)

    X = np.load(X_path).astype(np.float32)
    d = X.shape[1]

    # We use cosine similarity by doing inner product on normalized vectors
    if args.index_type == "flat":
        index = faiss.IndexFlatIP(d)
    else:
        index = faiss.IndexHNSWFlat(d, args.hnsw_m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 200
        index.hnsw.efSearch = 64

    index.add(X)

    idx_path = out_dir / "embeddings" / "faiss_text_clip.index"
    faiss.write_index(index, str(idx_path))
    print("Saved index:", idx_path)
    print("Vectors:", index.ntotal, "Dim:", d)

if __name__ == "__main__":
    main()
