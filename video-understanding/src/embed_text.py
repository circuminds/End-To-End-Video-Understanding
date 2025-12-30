import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import open_clip
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
    ap.add_argument("--pretrained", default="openai")  # good default
    ap.add_argument("--device", default="auto")        # auto/cpu/cuda
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
    texts = moments["text"].fillna("").astype(str).tolist()

    model, _, _ = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained
    )
    tokenizer = open_clip.get_tokenizer(args.model)
    model = model.to(device)
    model.eval()

    embeds = []
    bs = args.batch_size
    for i in tqdm(range(0, len(texts), bs), desc="Embedding text"):
        batch = texts[i:i+bs]
        tokens = tokenizer(batch).to(device)
        feat = model.encode_text(tokens)
        feat = feat / feat.norm(dim=-1, keepdim=True)  # normalize in torch
        embeds.append(feat.cpu().numpy().astype(np.float32))

    X = np.vstack(embeds)
    X = l2_normalize(X)  # ensure normalized

    emb_dir = out_dir / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)
    out_npy = emb_dir / "text_clip.npy"
    np.save(out_npy, X)

    print("Saved text embeddings:", out_npy)
    print("Shape:", X.shape)


if __name__ == "__main__":
    main()
