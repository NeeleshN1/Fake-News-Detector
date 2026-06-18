# NLP Fake News Headline Detector — Backend

This repository contains the **C++ inference backend** for a fake-news headline analysis system,
exposed via a FastAPI service and deployed using Docker.

The backend analyzes the **linguistic/stylistic structure** of a headline and outputs a continuous
*fake-likelihood score* between 0 and 1 (`0` → real-news patterns, `1` → fake/clickbait patterns).
This score is a **signal**, not a factual judgment of any specific claim.

---

## Architecture

- **Model:** fine-tuned **DistilBERT** (transformer), trained in PyTorch.
- **Inference:** runs entirely in **C++ via ONNX Runtime** — the model is exported to ONNX and
  **INT8-quantized** (~67 MB, fits small free-tier hosts). A hand-written C++ **WordPiece
  tokenizer** (`src/tokenizer.cpp`) reproduces `distilbert-base-uncased` so no Python is needed at
  inference time.
- **API Layer:** FastAPI (`api.py`) shells out to the C++ binary and returns
  `{ "label": str, "confidence": float }` where `confidence = P(fake)`.
- **Deployment:** Docker + Render (the Dockerfile fetches a prebuilt ONNX Runtime for Linux).

```
headline ──▶ FastAPI /predict ──▶ ./fake_news (C++) ──▶ WordPiece tokenize ──▶ ONNX Runtime ──▶ P(fake)
```

The previous bag-of-words logistic-regression model collapsed on out-of-vocabulary words (most modern
headlines), scoring almost everything "fake". The transformer generalizes via subword tokenization +
pretraining, producing well-calibrated scores (mean P(fake) ≈ 0.09 on real test headlines, ≈ 0.93 on
fake), at ~94% test accuracy / 0.94 F1.

---

## Layout

```
include/                C++ headers (tokenizer, onnx_model, classifier)
src/                    C++ sources (main, classifier, tokenizer, onnx_model)
model/                  headline.int8.onnx, vocab.txt, config.json   (deployed artifacts)
training/               Python pipeline (data prep, fine-tune, export, parity test)
third_party/onnxruntime ONNX Runtime (not committed; see Build)
CMakeLists.txt          builds `fake_news` against ONNX Runtime
Dockerfile              Render/Docker image
api.py                  FastAPI wrapper
```

---

## Build (local, Windows or Linux)

1. Download a prebuilt **ONNX Runtime** release that matches your OS from
   https://github.com/microsoft/onnxruntime/releases and unpack it so that
   `third_party/onnxruntime/{include,lib}` exists (or pass `-DORT_ROOT=/path/to/onnxruntime`).
2. Configure + build:
   ```
   cmake -S . -B build -DORT_ROOT=third_party/onnxruntime
   cmake --build build --config Release
   ```
3. Copy the binary (and on Windows the ONNX Runtime DLLs) next to `model/`:
   ```
   cp build/Release/fake_news.exe ./           # Windows
   cp third_party/onnxruntime/lib/*.dll ./      # Windows
   # Linux: cp build/fake_news ./ ; ensure libonnxruntime.so is on LD_LIBRARY_PATH
   ```
4. Run:
   ```
   ./fake_news "Some news headline to score"
   #   -> REAL 0.0385   (label + P(fake))
   ./fake_news --ids "headline"   # debug: print tokenizer input_ids
   ```

## Run the API

```
pip install fastapi uvicorn
uvicorn api:app --host 127.0.0.1 --port 8000
curl -X POST "http://127.0.0.1:8000/predict?headline=Some%20headline"
```

---

## Retrain / regenerate the model

Requires a Python env with `torch transformers datasets scikit-learn onnx onnxruntime pandas`
(GPU recommended). From `backend/`:

```
python training/prepare_data.py       # builds data/processed/{train,val,test}.csv (balanced, 1=fake)
python training/train_transformer.py  # fine-tunes DistilBERT -> model/transformer/
python training/export_onnx.py        # -> model/headline.int8.onnx, vocab.txt, config.json
python training/parity_test.py        # verifies the C++ binary matches the Python reference
```

---

Built and maintained by **Neelesh Nayak**.
