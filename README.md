# 🎥 End-to-End Video Understanding System  
**Upload → Process → Search → Summarize (Fully Local, No Paid APIs)**

This project is a **full-stack, end-to-end video understanding system** that transforms long videos into **searchable, summarizable, and explorable knowledge** using **only open-source models and local inference**.

It demonstrates how to build a **real-world multimodal AI system** combining video processing, speech recognition, embeddings, vector search, and LLM-based summarization—wrapped in a clean interactive UI.

---

## ✨ What This System Can Do

- Upload videos (up to **2GB**)
- Automatically segment videos into meaningful moments
- Perform **semantic search** across video content
- Generate **human-readable summaries and timelines**
- Play precise video clips for search results and summary chunks
- Run **entirely locally** (no OpenAI / paid APIs)

---

## 🔍 Key Features

### 🎞️ Video Understanding Pipeline
- Video ingestion & metadata extraction
- Audio extraction (handles **silent videos** gracefully)
- Speech-to-text using **faster-whisper**
- Moment segmentation with adaptive windowing
- Thumbnail extraction
- Text embeddings (CLIP)
- Image embeddings (CLIP)
- Multimodal embedding fusion
- FAISS-based semantic indexing

### 🔎 Semantic Search
- Natural-language search across video moments
- Ranked results with:
  - timestamps
  - transcript snippets
  - thumbnails
  - playable video clips

### 🧠 Summarization
- Chunk-level summaries with timestamps
- Final high-level summary
- Timeline / chapter-style navigation
- Search within summaries
- Optional clip playback per summary chunk
- Raw JSON available for debugging (toggleable)

### 🧱 Engineering & Product Quality
- Fully local inference (privacy-first)
- Robust to:
  - silent videos
  - very short videos
  - session restarts
- Dockerized for reproducible hosting
- Automatic cleanup between uploads
- Streamlit-based interactive UI

---

## 🏗️ High-Level Architecture

<img width="346" height="600" alt="image" src="https://github.com/user-attachments/assets/620a5134-bb81-4369-8be9-0b4f102f992f" />


---

## 🧰 Tech Stack

### Core
- **Python 3.10**
- **Streamlit** – UI
- **Docker** – containerized deployment

### Machine Learning
- **faster-whisper** – speech-to-text
- **open-clip-torch** – text & image embeddings
- **FAISS** – vector similarity search
- **Ollama** – local LLM for summarization

### Media Processing
- **ffmpeg / ffprobe** – video & audio processing

### Data & Utilities
- NumPy, Pandas
- Parquet, JSON

---

## 📁 Project Structure

<img width="450" height="590" alt="image" src="https://github.com/user-attachments/assets/58955337-b086-4b99-8bf0-513a91533fe8" />


---

## 🚀 Running Locally (Without Docker)

### 1️⃣ Prerequisites
- Python 3.9+
- `ffmpeg` installed
- Ollama installed (for summaries)

Start Ollama:
```bash
ollama serve

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Run the app
streamlit run app/streamlit_app.py

Open in browser:
http://localhost:8501


🐳 Running with Docker (Recommended)
1️⃣ Build Docker image

docker build -t video-understanding .

2️⃣ Run container

docker run --rm \
  -p 8501:8501 \
  -v "$PWD/data:/app/data" \
  video-understanding

Open in browser:
http://localhost:8501

The mounted data/ directory persists outputs but is automatically cleaned by the app between sessions.

🧠 Ollama Configuration (Summaries)

When running in Docker, the app connects to Ollama on the host via:

http://host.docker.internal:11434

Make sure Ollama is running:
ollama serve



##Privacy & Data Handling

- No external paid API
- All processing happens locally
- Uploaded videos are automatically deleted when:
      - a new video is uploaded
      - a new session starts
- Designed for single-user / demo / research usage

## ⚠️ Known Limitations

- CPU-only by default (GPU support can be added)
- Long videos take time to process
- Ollama must be running for summaries
- Single-user pipeline (no job queue)


## 📈 Future Improvements

- GPU-enabled Docker image
- Background job queue (Celery / Redis)
Multi-user support
Automatic chapter title generation
Evaluation metrics (ROUGE, human eval)
Export summaries to PDF / Markdown
Interactive timeline scrubber

## 🙌 Acknowledgements

OpenAI CLIP
Whisper / faster-whisper
FAISS
Ollama
Streamlit


