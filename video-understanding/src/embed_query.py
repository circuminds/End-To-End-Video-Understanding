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
    args = ap.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model, _, _ = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    tokenizer = open_clip.get_tokenizer(args.model)
    model = model.to(device).eval()

    tokens = tokenizer([args.query]).to(device)
    v = model.encode_text(tokens)
    v = v / v.norm(dim=-1, keepdim=True)
    v = v.cpu().numpy().astype(np.float32)  # (1, d)

    out_dir = Path(args.processed_dir) / args.video_id / "embeddings"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "query.npy"
    np.save(out_path, v)
    print("Saved query embedding:", out_path, "shape=", v.shape)

if __name__ == "__main__":
    main()
