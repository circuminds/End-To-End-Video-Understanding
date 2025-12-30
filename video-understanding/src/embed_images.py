import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import open_clip
from PIL import Image
from tqdm import tqdm


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, eps)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_id", required=True)
    ap.add_argument("--processed_dir", default="data/processed")
    ap.add_argument("--model", default="ViT-B-32")
    ap.add_argument("--pretrained", default="openai")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    out_dir = Path(args.processed_dir) / args.video_id
    moments_path = out_dir / "moments.parquet"
    if not moments_path.exists():
        raise FileNotFoundError(moments_path)

    moments = pd.read_parquet(moments_path)
    if "thumb_path" not in moments.columns:
        raise ValueError("moments.parquet must contain thumb_path. Run extract_thumbs.py first.")

    thumbs = moments["thumb_path"].astype(str).tolist()

    # Auto-fix missing thumbs (robust for video-end timestamps)
    missing = [p for p in thumbs if not Path(p).exists()]
    if missing:
        print(f"[embed_images] Missing {len(missing)} thumbnails. Repairing...")
        import subprocess
        subprocess.run(
            ["python", "src/fix_missing_thumbs.py", "--video_id", args.video_id, "--force"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        # Reload after repair
        moments = pd.read_parquet(moments_path)
        thumbs = moments["thumb_path"].astype(str).tolist()
        still_missing = [p for p in thumbs if not Path(p).exists()]
        if still_missing:
            raise FileNotFoundError(f"Still missing thumbnails after repair. Example: {still_missing[0]}")


    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained
    )
    model = model.to(device).eval()

    feats = []
    bs = args.batch_size

    for i in tqdm(range(0, len(thumbs), bs), desc="Embedding images"):
        batch_paths = thumbs[i:i+bs]
        imgs = []
        for p in batch_paths:
            img = Image.open(p).convert("RGB")
            imgs.append(preprocess(img))
        x = torch.stack(imgs, dim=0).to(device)
        v = model.encode_image(x)
        v = v / v.norm(dim=-1, keepdim=True)
        feats.append(v.cpu().numpy().astype(np.float32))

    X = np.vstack(feats)
    X = l2_normalize(X)

    emb_dir = out_dir / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)
    out_npy = emb_dir / "image_clip.npy"
    np.save(out_npy, X)
    print("Saved image embeddings:", out_npy)
    print("Shape:", X.shape)

if __name__ == "__main__":
    main()
