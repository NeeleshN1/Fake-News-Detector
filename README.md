# NLP Fake News Detector

A live-news web app that rates news headlines **real or fake** in real time, powered by a
fine-tuned **DistilBERT** transformer running in **C++ via ONNX Runtime**.

The front end is a **TikTok-style vertical feed** of live news (NewsAPI.org); every headline is
scored by the model and shown with a real/fake badge. You can also paste your own headline to
analyze it.

> The model detects the **linguistic/stylistic signal** of fake/clickbait headlines (a signal, not a
> fact-check of any specific claim).

## Repository layout

```
backend/   C++ inference engine (WordPiece tokenizer + ONNX Runtime) + FastAPI + training pipeline
ui/        Next.js 16 (React 19, Tailwind v4) TikTok-style live-news feed
```

See [`backend/README.md`](backend/README.md) for the model architecture, build, and retraining steps.

## Architecture

```
Browser (feed)
   └─ GET /api/news  (Next.js server route)
        ├─ NewsAPI.org top-headlines        (key server-side only)
        └─ POST /predict_batch  ─▶  FastAPI  ─▶  ./fake_news (C++)  ─▶  ONNX Runtime  ─▶  P(fake)
```

- **Model:** DistilBERT fine-tuned on a balanced mix of fake/real headline datasets (WELFake,
  GonzaloA) plus diverse real headlines (AG News, HuffPost) so modern multi-domain news isn't
  over-flagged. Exported to ONNX and **INT8-quantized** (~67 MB) for fast, small-footprint inference.
- **Inference:** a hand-written C++ WordPiece tokenizer (matching `distilbert-base-uncased`) feeds
  ONNX Runtime's C++ API. ~92% test accuracy.
- **API:** FastAPI exposes `POST /predict` (one headline) and `POST /predict_batch` (a page of
  headlines, one model load).

## Run locally

**Backend** (needs ONNX Runtime — see `backend/README.md`):
```
cd backend
uvicorn api:app --host 127.0.0.1 --port 8000
```

**Frontend:**
```
cd ui
npm install
# create .env.local with your NewsAPI key:
#   NEWSAPI_KEY=...           (server-side)
#   NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
#   MODEL_API_BASE=http://127.0.0.1:8000
npm run dev   # http://localhost:3000
```

---

Built by **Neelesh Nayak**.
