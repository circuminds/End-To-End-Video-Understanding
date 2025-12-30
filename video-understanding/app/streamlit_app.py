import json
import os
import shutil
import subprocess
import time
import hashlib
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
import faiss


# -----------------------------
# Utilities
# -----------------------------
def run_cmd(cmd, cwd: Optional[Path] = None):
    """Run a command and stream output into the app."""
    p = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )
    lines = []
    for line in p.stdout:
        lines.append(line)
        yield line
    p.wait()
    if p.returncode != 0:
        raise RuntimeError("Command failed:\n" + "".join(lines))


def fmt_ts(seconds: float) -> str:
    s = int(seconds)
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:02d}"


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def load_meta(processed_dir: Path, video_id: str) -> dict:
    return json.loads((processed_dir / video_id / "metadata.json").read_text())


def build_clip(video_path: Path, start_s: float, end_s: float, out_path: Path) -> Path:
    """
    Builds a short playable mp4 clip around [start_s, end_s].
    Returns the actual clip path created.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dur = max(0.2, end_s - start_s)

    def run_ffmpeg(cmd):
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return p.returncode, (p.stdout or "") + "\n" + (p.stderr or "")

    # Attempt 1: H.264 + AAC (best browser compatibility)
    cmd1 = [
        "ffmpeg", "-y",
        "-ss", f"{start_s:.3f}",
        "-i", str(video_path),
        "-t", f"{dur:.3f}",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        str(out_path),
    ]
    rc, out = run_ffmpeg(cmd1)

    # Validate output
    if rc == 0 and out_path.exists() and out_path.stat().st_size > 10_000:
        return out_path

    # Attempt 2: stream copy (fast; works if source already browser-compatible)
    copy_path = out_path.with_name(out_path.stem + "_copy.mp4")
    cmd2 = [
        "ffmpeg", "-y",
        "-ss", f"{start_s:.3f}",
        "-i", str(video_path),
        "-t", f"{dur:.3f}",
        "-c", "copy",
        "-movflags", "+faststart",
        str(copy_path),
    ]
    rc2, out2 = run_ffmpeg(cmd2)
    if rc2 == 0 and copy_path.exists() and copy_path.stat().st_size > 10_000:
        return copy_path

    # Attempt 3: fallback without audio (avoids AAC issues)
    fallback_path = out_path.with_name(out_path.stem + "_fallback.mp4")
    cmd3 = [
        "ffmpeg", "-y",
        "-ss", f"{start_s:.3f}",
        "-i", str(video_path),
        "-t", f"{dur:.3f}",
        "-vf", "scale=1280:-2",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-an",
        "-movflags", "+faststart",
        str(fallback_path),
    ]
    rc3, out3 = run_ffmpeg(cmd3)
    if rc3 == 0 and fallback_path.exists() and fallback_path.stat().st_size > 10_000:
        return fallback_path

    raise RuntimeError(
        "ffmpeg failed to create a playable clip.\n\n"
        "Attempt 1 (x264+aac) output:\n" + out[-6000:] + "\n\n"
        "Attempt 2 (copy) output:\n" + out2[-6000:] + "\n\n"
        "Attempt 3 (fallback no-audio) output:\n" + out3[-6000:]
    )


def processed_video_ids(processed_dir: Path):
    if not processed_dir.exists():
        return []
    return sorted([p.name for p in processed_dir.glob("*") if p.is_dir()])


def file_save_to(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def run_step(step_name: str, cmd: list, cwd: Path, status, progress, step_idx: int, total_steps: int):
    """Run one pipeline step with high-level status updates (no live logs)."""
    status.update(label=f"⏳ {step_name}", state="running")
    progress.progress(step_idx / total_steps)

    p = subprocess.run(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if p.returncode != 0:
        status.update(label=f"❌ {step_name} failed", state="error")
        raise RuntimeError(f"{step_name} failed.\n\n--- Output ---\n{p.stdout[-12000:]}")
    status.update(label=f"✅ {step_name}", state="complete")
    return p.stdout


def get_latest_video_id(processed_dir: Path):
    if not processed_dir.exists():
        return None
    dirs = [p for p in processed_dir.glob("*") if p.is_dir()]
    if not dirs:
        return None
    return max(dirs, key=lambda p: p.stat().st_mtime).name


def _safe_under_data_dir(path: Path) -> bool:
    p = path.resolve()
    return "data" in p.parts


def wipe_all_data(processed_dir: Path, raw_dir: Path):
    """
    Remove all previously processed videos and raw uploads.
    Only deletes within these two directories.
    """
    for d in [processed_dir, raw_dir]:
        if not _safe_under_data_dir(d):
            raise RuntimeError(f"Safety stop: refusing to wipe non-data directory: {d.resolve()}")

    if processed_dir.exists():
        shutil.rmtree(processed_dir, ignore_errors=True)
    if raw_dir.exists():
        shutil.rmtree(raw_dir, ignore_errors=True)

    processed_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Clear session state pointers
    for k in [
        "last_video_id", "search_results", "last_clip_path", "last_clip_label",
        "pipeline_logs", "last_uploaded_sig"
    ]:
        if k in st.session_state:
            del st.session_state[k]


def file_signature(uploaded) -> str:
    """
    Fast signature to detect a new upload without hashing the whole file.
    name + size + hash(first/last 4KB)
    """
    name = uploaded.name
    size = uploaded.size
    data = uploaded.getvalue()
    head = data[:4096]
    tail = data[-4096:] if len(data) > 4096 else data
    h = hashlib.sha1(head + tail).hexdigest()[:12]
    return f"{name}|{size}|{h}"


def ensure_fresh_session(processed_dir: Path, raw_dir: Path):
    """
    On a brand new Streamlit session (tab reopened / refresh / server restart),
    wipe any leftover data from previous session.
    """
    marker = processed_dir / ".session_marker"
    ensure_dir(processed_dir)
    ensure_dir(raw_dir)

    # Always initialize safely
    sess = st.session_state
    sess.setdefault("session_marker", str(time.time()))

    # First-run per session: if marker exists from previous session, wipe and replace
    if not sess.get("_fresh_session_checked", False):
        if marker.exists():
            wipe_all_data(processed_dir, raw_dir)

        # ensure key still exists after wipe_all_data (it clears session keys)
        sess.setdefault("session_marker", str(time.time()))
        marker.write_text(sess.get("session_marker", str(time.time())))
        sess["_fresh_session_checked"] = True



# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="Video Understanding (Local)", layout="wide")
st.title("🎥 Video Understanding System (Upload → Process → Search → Summarize)")

ROOT = Path(".").resolve()

PROCESSED_DIR = Path("data/processed")
processed_dir = PROCESSED_DIR

RAW_DIR = Path("data/raw_videos")
raw_dir = RAW_DIR

ensure_dir(processed_dir)
ensure_dir(raw_dir)

# Wipe data when app is reopened / new session starts
ensure_fresh_session(processed_dir, raw_dir)

st.sidebar.markdown("### Upload a video")
uploaded = st.sidebar.file_uploader("Upload mp4/mkv", type=["mp4", "mkv", "mov", "webm"])

uploaded_path = None
if uploaded is not None:
    sig = file_signature(uploaded)

    # If upload changed, wipe everything and start fresh
    if st.session_state.get("last_uploaded_sig") != sig:
        wipe_all_data(processed_dir, raw_dir)
        st.session_state["last_uploaded_sig"] = sig
        st.sidebar.info("New video detected → cleared previous data.")

    # Save uploaded file after wipe
    up_name = uploaded.name.replace(" ", "_")
    uploaded_path = raw_dir / f"__uploaded__{up_name}"
    file_save_to(uploaded_path, uploaded.getvalue())
    st.sidebar.success(f"Saved upload: {uploaded_path.name}")

st.sidebar.markdown("---")
st.sidebar.markdown("### Parameters")
use_multimodal = True
run_summaries = True

ollama_model = "qwen2.5:7b-instruct"
chunk_s = st.sidebar.number_input("Summary chunk size (seconds)", min_value=30, max_value=300, value=90, step=10)
thumb_w = st.sidebar.number_input("Thumbnail width", min_value=320, max_value=1280, value=640, step=80)

# -----------------------------
# Pipeline Runner UI
# -----------------------------
st.subheader("Pipeline")

pipeline_col1, pipeline_col2 = st.columns([2, 1], vertical_alignment="top")

with pipeline_col2:
    st.markdown("**Requirements on your machine**")
    st.markdown("- `ffmpeg` installed")
    st.markdown("- Python deps installed (already done)")
    st.markdown("- For summaries: `ollama serve` running")

    st.markdown("---")
    mode = st.radio("Select Mode", ["Search", "Summaries"], horizontal=True, index=0)

with pipeline_col1:
    if uploaded_path is None:
        st.info("Upload a video from the sidebar to process it.")
    else:
        if st.button("🚀 Process uploaded video"):
            try:
                with st.spinner("Processing video… this may take a while for long videos."):
                    status = st.status("Starting pipeline…", expanded=False)
                    progress = st.progress(0)

                    # Ensure clean (upload handler already wiped, but keep deterministic)
                    sig = file_signature(uploaded)
                    if st.session_state.get("last_uploaded_sig") != sig:
                        wipe_all_data(processed_dir, raw_dir)
                        st.session_state["last_uploaded_sig"] = sig

                        # re-save upload after wiping raw_dir
                        up_name = uploaded.name.replace(" ", "_")
                        uploaded_path = raw_dir / f"__uploaded__{up_name}"
                        file_save_to(uploaded_path, uploaded.getvalue())

                    st.session_state["last_video_id"] = None

                    steps = [
                        ("Ingest video", ["python", "src/ingest.py", "--video", str(uploaded_path)]),
                        ("Transcribe audio (Whisper)", ["python", "src/transcribe.py", "--video_id", "__VID__", "--model", "small", "--vad_filter"]),
                        ("Create moments", ["python", "src/make_moments.py", "--video_id", "__VID__", "--window_s", "20", "--stride_s", "10"]),
                        ("Extract thumbnails", ["python", "src/extract_thumbs.py", "--video_id", "__VID__", "--thumb_w", str(int(thumb_w))]),
                        ("Fix missing thumbnails", ["python", "src/fix_missing_thumbs.py", "--video_id", "__VID__", "--force"]),
                        ("Embed text", ["python", "src/embed_text.py", "--video_id", "__VID__"]),
                        ("Build text index", ["python", "src/build_index.py", "--video_id", "__VID__"]),
                    ]

                    if use_multimodal:
                        steps += [
                            ("Embed images", ["python", "src/embed_images.py", "--video_id", "__VID__", "--batch_size", "8"]),
                            ("Fuse embeddings", ["python", "src/fuse_embeddings.py", "--video_id", "__VID__", "--alpha", "1.0", "--beta", "1.0"]),
                            ("Build fused index", ["python", "src/build_index_fused.py", "--video_id", "__VID__"]),
                        ]

                    if run_summaries:
                        steps += [
                            ("Generate summaries (Ollama)", [
                                "python", "src/summarize_chunks.py",
                                "--video_id", "__VID__",
                                "--chunk_s", str(float(chunk_s)),
                                "--model", ollama_model,
                                #"--ollama_host", "http://host.docker.internal:11434",
                            ])
                        ]

                    total_steps = len(steps)

                    # Step 1: Ingest
                    status.update(label="⏳ Ingest video", state="running")
                    progress.progress(0 / total_steps)

                    out = subprocess.run(
                        steps[0][1],
                        cwd=str(ROOT),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                    if out.returncode != 0:
                        status.update(label="❌ Ingest video failed", state="error")
                        raise RuntimeError(f"Ingest failed.\n\n--- Output ---\n{out.stdout[-12000:]}")

                    newest = max(processed_dir.glob("*"), key=lambda p: p.stat().st_mtime)
                    vid = newest.name
                    st.session_state["last_video_id"] = vid
                    status.update(label=f"✅ Ingest complete (video_id={vid})", state="complete")
                    progress.progress(1 / total_steps)

                    # Remaining steps
                    step_outputs = []
                    for idx, (name, cmd) in enumerate(steps[1:], start=2):
                        cmd = [vid if x == "__VID__" else x for x in cmd]
                        out_text = run_step(name, cmd, ROOT, status, progress, idx, total_steps)
                        step_outputs.append((name, out_text))

                    progress.progress(1.0)
                    status.update(label="✅ Pipeline complete", state="complete")
                    st.success(f"Done! Processed video_id: {vid}")

                    with st.expander("Show debug output (optional)"):
                        for name, text in step_outputs:
                            st.markdown(f"**{name}**")
                            st.code(text[-8000:])

            except Exception as e:
                st.error(str(e))

# Stop if nothing processed yet
existing_ids = processed_video_ids(processed_dir)
if not existing_ids:
    st.stop()

# Active video_id: last in session, else latest on disk
video_id = st.session_state.get("last_video_id") or get_latest_video_id(processed_dir)
if video_id is None:
    st.info("No processed video yet. Upload a video and click **Process uploaded video**.")
    st.stop()

vid_dir = processed_dir / video_id
moments_path = vid_dir / "moments.parquet"
emb_dir = vid_dir / "embeddings"
summ_path = vid_dir / "summaries.json"

if not moments_path.exists():
    st.warning(f"Missing {moments_path}. Process the uploaded video first.")
    st.stop()

moments = pd.read_parquet(moments_path)

# Choose index
fused_idx = emb_dir / "faiss_fused.index"
text_idx = emb_dir / "faiss_text_clip.index"
index_path = fused_idx if fused_idx.exists() else text_idx

if not index_path.exists():
    st.warning(f"Missing index {index_path}. Run embeddings + index build.")
    st.stop()

index = faiss.read_index(str(index_path))
meta = load_meta(processed_dir, video_id)
video_path = Path(meta["raw_video_path"])


# Query embedding via subprocess (avoids OpenMP clashes)
def embed_query_to_npy(query: str, fused: bool) -> np.ndarray:
    if fused:
        cmd = ["python", "src/embed_query_fused.py", "--video_id", video_id, "--query", query]
        out_npy = emb_dir / "query_fused.npy"
    else:
        cmd = ["python", "src/embed_query.py", "--video_id", video_id, "--query", query]
        out_npy = emb_dir / "query.npy"

    subprocess.run(cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=True)
    return np.load(out_npy).astype(np.float32)


# -----------------------------
# Search UI
# -----------------------------
if mode == "Search":
    st.subheader(f"Video: {meta.get('original_name', video_id)}")

    c1, c2 = st.columns([3, 1])
    with c1:
        query = st.text_input("Query", "explain most interesting part from video")
    with c2:
        top_k = st.number_input("Top K", min_value=1, max_value=30, value=5, step=1)

    if st.button("🔎 Search"):
        fused = (index_path.name == "faiss_fused.index")
        qv = embed_query_to_npy(query, fused=fused)
        scores, ids = index.search(qv, int(top_k))

        st.markdown("### Results")
        for rank, (idx, score) in enumerate(zip(ids[0], scores[0]), start=1):
            row = moments.iloc[int(idx)]
            start_s = float(row["start"])
            end_s = float(row["end"])
            thumb_path = row.get("thumb_path", None)
            text = str(row["text"])

            with st.container(border=True):
                left, right = st.columns([1, 3])
                with left:
                    if thumb_path and Path(thumb_path).exists():
                        st.image(thumb_path, caption=f"#{rank} score={float(score):.4f}", use_container_width=True)
                    else:
                        st.write(f"#{rank} score={float(score):.4f}")

                with right:
                    st.markdown(f"**Time:** {fmt_ts(start_s)} – {fmt_ts(end_s)}")
                    st.write(text[:450] + ("..." if len(text) > 450 else ""))

                    clip_dir = vid_dir / "clips"
                    clip_path = clip_dir / f"hit_{rank:02d}_{int(start_s)}_{int(end_s)}.mp4"
                    clip_start = max(0.0, start_s - 3.0)
                    clip_end = end_s + 3.0

                    if st.button(f"▶️ Make & Play Clip (Result #{rank})", key=f"clip_{rank}"):
                        try:
                            with st.spinner("Creating clip…"):
                                actual_clip = build_clip(video_path, clip_start, clip_end, clip_path)
                            st.success(f"Clip created: {actual_clip.name} ({actual_clip.stat().st_size/1e6:.1f} MB)")
                            st.video(str(actual_clip))
                        except Exception as e:
                            st.error("Clip creation failed.")
                            st.code(str(e))

# -----------------------------
# Summaries UI (Improved)
# -----------------------------
else:
    st.subheader(f"Summaries — video: {meta.get('original_name', video_id)}")

    if not summ_path.exists():
        st.warning(f"Missing {summ_path}. Run summaries from the pipeline section.")
        st.stop()

    summ = json.loads(summ_path.read_text())

    # ---- Top summary header ----
    colA, colB, colC = st.columns([2, 1, 1])
    with colA:
        st.markdown("### 🧾 Final Summary")
    with colB:
        show_raw = st.toggle("Show raw JSON", key="show_raw_json", value=False)
    with colC:
        st.download_button(
            "Download summaries.json",
            data=summ_path.read_bytes(),
            file_name="summaries.json",
            mime="application/json",
            use_container_width=True,
        )

     # ---- Raw JSON (optional) ----
    if show_raw:
        st.markdown("---")
        st.markdown("### 🧩 Raw summaries.json")
        st.json(summ)

    final_raw = (summ.get("final_raw") or "").strip()
    if final_raw:
        # Make it readable: preserve paragraphs
        st.write(final_raw)
    else:
        st.info("No final summary found in summaries.json (final_raw is empty).")

    st.markdown("---")

    # ---- Key takeaways (heuristic) ----
    st.markdown("### ✅ Key Takeaways")
    takeaways = summ.get("takeaways", None)

    if isinstance(takeaways, list) and takeaways:
        for t in takeaways[:12]:
            st.markdown(f"- {t}")
    else:
        # Derive takeaways from chunk summaries if not provided by backend
        chunks = summ.get("chunks", [])
        bullets = []
        for c in chunks:
            text = (c.get("summary_raw") or "").strip()
            if not text:
                continue
            # grab first non-empty sentence-ish line
            first_line = next((ln.strip("-• ").strip()
                               for ln in text.splitlines()
                               if ln.strip()), "")
            if first_line:
                bullets.append(first_line)

        if bullets:
            for b in bullets[:8]:
                st.markdown(f"- {b}")
        else:
            st.caption("No takeaways available yet. (You can enhance summarize_chunks.py to output takeaways.)")

    st.markdown("---")

    # ---- Timeline / Chapters ----
    st.markdown("### ⏱️ Timeline (Chapters)")
    chunks = summ.get("chunks", [])
    if not chunks:
        st.info("No chunk summaries found.")
        st.stop()

    # Controls
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        q = st.text_input("Search within summaries", "")
    with c2:
        compact = st.toggle("Compact view", value=False)
    with c3:
        max_show = st.number_input("Max chunks", min_value=5, max_value=200, value=50, step=5)

    # Filter chunks by query
    def chunk_text(c):
        return ((c.get("summary_raw") or "") + "\n" + (c.get("summary_clean") or "")).lower()

    filtered = []
    if q.strip():
        ql = q.strip().lower()
        for c in chunks:
            if ql in chunk_text(c):
                filtered.append(c)
    else:
        filtered = chunks

    if not filtered:
        st.warning("No chunks match your search.")
        st.stop()

    # Display chunks
    for i, c in enumerate(filtered[: int(max_show)]):
        start = float(c.get("start", 0))
        end = float(c.get("end", 0))
        title = c.get("title") or f"Chunk {c.get('chunk_id', i)}  {fmt_ts(start)}–{fmt_ts(end)}"

        body = (c.get("summary_clean") or c.get("summary_raw") or "").strip()
        if not body:
            body = "(empty chunk summary)"

        if compact:
            # compact: show as a single line card
            with st.container(border=True):
                st.markdown(f"**{fmt_ts(start)}–{fmt_ts(end)}**  ·  {body[:220]}{'...' if len(body) > 220 else ''}")
        else:
            with st.expander(title, expanded=(i == 0 and not q.strip())):
                st.caption(f"{fmt_ts(start)}–{fmt_ts(end)}")
                st.write(body)

                # Optional: make clip from this chunk
                clip_dir = vid_dir / "clips"
                clip_path = clip_dir / f"chunk_{int(start)}_{int(end)}.mp4"
                if st.button(f"▶️ Make & Play Clip ({fmt_ts(start)}–{fmt_ts(end)})", key=f"chunk_clip_{video_id}_{i}"):
                    try:
                        with st.spinner("Creating clip…"):
                            actual_clip = build_clip(video_path, start, end, clip_path)
                        st.video(str(actual_clip))
                    except Exception as e:
                        st.error("Clip creation failed.")
                        st.code(str(e))
