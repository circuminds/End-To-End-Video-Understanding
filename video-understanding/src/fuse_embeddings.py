import argparse
from pathlib import Path
import numpy as np

def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, eps)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_id", required=True)
    ap.add_argument("--processed_dir", default="data/processed")
    ap.add_argument("--alpha", type=float, default=1.0, help="weight for text")
    ap.add_argument("--beta", type=float, default=1.0, help="weight for image")
    args = ap.parse_args()

    out_dir = Path(args.processed_dir) / args.video_id / "embeddings"
    text_path = out_dir / "text_clip.npy"
    img_path = out_dir / "image_clip.npy"
    if not text_path.exists(): raise FileNotFoundError(text_path)
    if not img_path.exists(): raise FileNotFoundError(img_path)

    T = np.load(text_path).astype(np.float32)
    I = np.load(img_path).astype(np.float32)
    if T.shape[0] != I.shape[0]:
        raise ValueError("Text and image embeddings must have same number of rows")

    # weighted concat
    X = np.concatenate([args.alpha * T, args.beta * I], axis=1)
    X = l2_normalize(X)

    out_path = out_dir / f"fused_a{args.alpha}_b{args.beta}.npy"
    np.save(out_path, X)
    print("Saved fused embeddings:", out_path)
    print("Shape:", X.shape)

if __name__ == "__main__":
    main()
