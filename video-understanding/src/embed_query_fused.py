import argparse
from pathlib import Path
import numpy as np
import torch
import open_clip

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_id", required=True)
    ap.add_argument("--query", required=True)
    ap.add_argument("--processed_dir", default="data/processed")
    ap.add_argument("--model", default="ViT-B-32")
    ap.add_argument("--pretrained", default="openai")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=1.0)
    args = ap.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model, _, _ = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    tokenizer = open_clip.get_tokenizer(args.model)
    model = model.to(device).eval()

    tokens = tokenizer([args.query]).to(device)
    t = model.encode_text(tokens)
    t = t / t.norm(dim=-1, keepdim=True)
    t = t.cpu().numpy().astype(np.float32)  # (1, d)

    # fused vector = [alpha*T, beta*I]
    # query has no image, so I is zeros
    zeros = np.zeros_like(t)
    q = np.concatenate([args.alpha * t, args.beta * zeros], axis=1)

    # normalize
    q = q / np.linalg.norm(q, axis=1, keepdims=True)

    out_dir = Path(args.processed_dir) / args.video_id / "embeddings"
    out_path = out_dir / "query_fused.npy"
    np.save(out_path, q)
    print("Saved fused query embedding:", out_path, "shape=", q.shape)

if __name__ == "__main__":
    main()
