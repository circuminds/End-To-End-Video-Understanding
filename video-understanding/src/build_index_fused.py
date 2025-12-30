import argparse
from pathlib import Path
import numpy as np
import faiss

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_id", required=True)
    ap.add_argument("--processed_dir", default="data/processed")
    ap.add_argument("--emb_name", default="fused_a1.0_b1.0.npy")
    ap.add_argument("--index_type", default="hnsw", choices=["flat", "hnsw"])
    ap.add_argument("--hnsw_m", type=int, default=32)
    args = ap.parse_args()

    emb_dir = Path(args.processed_dir) / args.video_id / "embeddings"
    X_path = emb_dir / args.emb_name
    if not X_path.exists():
        raise FileNotFoundError(X_path)

    X = np.load(X_path).astype(np.float32)
    d = X.shape[1]

    if args.index_type == "flat":
        index = faiss.IndexFlatIP(d)
    else:
        index = faiss.IndexHNSWFlat(d, args.hnsw_m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 200
        index.hnsw.efSearch = 64

    index.add(X)

    idx_path = emb_dir / "faiss_fused.index"
    faiss.write_index(index, str(idx_path))
    print("Saved index:", idx_path)
    print("Vectors:", index.ntotal, "Dim:", d)

if __name__ == "__main__":
    main()
