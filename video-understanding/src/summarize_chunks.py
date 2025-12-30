import argparse
import json
from pathlib import Path
import pandas as pd
import requests

def fmt_ts(seconds: float) -> str:
    s = int(seconds)
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:02d}"

def ollama_chat(model: str, prompt: str, host: str = "http://localhost:11434") -> str:
    url = f"{host}/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a precise summarization assistant. Do not hallucinate details not present."},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {
            "temperature": 0.2,
        }
    }
    r = requests.post(url, json=payload, timeout=600)
    r.raise_for_status()
    return r.json()["message"]["content"].strip()

def build_time_chunks(moments: pd.DataFrame, chunk_s: float):
    moments = moments.sort_values("start").reset_index(drop=True)
    t0 = float(moments["start"].min())
    t1 = float(moments["end"].max())
    chunks = []
    cstart = t0
    while cstart < t1:
        cend = cstart + chunk_s
        mask = (moments["start"] < cend) & (moments["end"] > cstart)
        part = moments[mask]
        text = " ".join(part["text"].astype(str).tolist()).strip()
        if len(text) > 0:
            chunks.append((cstart, cend, text))
        cstart = cend
    return chunks

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_id", required=True)
    ap.add_argument("--processed_dir", default="data/processed")
    ap.add_argument("--chunk_s", type=float, default=90.0)   # 60–120 is good
    ap.add_argument("--model", default="qwen2.5:7b-instruct")
    ap.add_argument("--ollama_host", default="http://localhost:11434")
    args = ap.parse_args()

    out_dir = Path(args.processed_dir) / args.video_id
    moments_path = out_dir / "moments.parquet"
    if not moments_path.exists():
        raise FileNotFoundError(moments_path)

    moments = pd.read_parquet(moments_path)
    chunks = build_time_chunks(moments, args.chunk_s)

    chunk_summaries = []
    for i, (cs, ce, text) in enumerate(chunks):
        prompt = f"""
Summarize the following transcript segment from {fmt_ts(cs)} to {fmt_ts(ce)}.

Return JSON with keys:
- title: short title (max 8 words)
- bullets: 3-6 bullet points
- keywords: 5-10 keywords (single words or short phrases)
- quotes: up to 2 short exact phrases from the text if useful (optional)

TEXT:
{text}
"""
        ans = ollama_chat(args.model, prompt, host=args.ollama_host)
        chunk_summaries.append({
            "chunk_id": i,
            "start": float(cs),
            "end": float(ce),
            "summary_raw": ans,
        })
        print(f"chunk {i} {fmt_ts(cs)}–{fmt_ts(ce)} done")

    # Build final summary from chunk summaries (hierarchical)
    joined = "\n\n".join([f"[{fmt_ts(c['start'])}-{fmt_ts(c['end'])}]\n{c['summary_raw']}" for c in chunk_summaries])
    final_prompt = f"""
You are given chunk summaries for a video. Produce:
1) Overall summary (6-10 bullets)
2) 3-6 chapters with: title, start_time, end_time, what it covers (bullets)
3) A short TL;DR (2-3 sentences)

Return JSON with keys: tldr, overall_bullets, chapters.

CHUNK SUMMARIES:
{joined}
"""
    final = ollama_chat(args.model, final_prompt, host=args.ollama_host)

    out = {
        "video_id": args.video_id,
        "chunk_s": args.chunk_s,
        "model": args.model,
        "chunks": chunk_summaries,
        "final_raw": final,
    }

    out_path = out_dir / "summaries.json"
    out_path.write_text(json.dumps(out, indent=2))
    print("Saved:", out_path)

if __name__ == "__main__":
    main()
