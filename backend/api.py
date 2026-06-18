# Backend API for headline prediction
# Runs the C++ model (transformer / ONNX Runtime) and returns predictions to the frontend.
# Response shape is unchanged: { "label": str, "confidence": float } where confidence = P(fake).

import os
import subprocess

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Resolve the binary + working dir relative to this file so it works no matter where
# uvicorn is launched (the C++ side loads model/ relative to its cwd).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BINARY = os.path.join(BASE_DIR, "fake_news.exe" if os.name == "nt" else "fake_news")


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(headline: str):
    result = subprocess.run(
        [BINARY, headline],
        capture_output=True,
        text=True,
        cwd=BASE_DIR,
    )

    output = result.stdout.strip().split()
    if len(output) < 2:
        raise HTTPException(
            status_code=500,
            detail=f"inference failed: {result.stderr.strip() or result.stdout.strip()}",
        )

    return {
        "label": output[0],
        "confidence": float(output[1]),
    }


class BatchIn(BaseModel):
    headlines: list[str]


@app.post("/predict_batch")
def predict_batch(payload: BatchIn):
    """Score many headlines in a single C++ process invocation (one model load)."""
    headlines = payload.headlines
    if not headlines:
        return []

    # one headline per line; flatten any embedded newlines so lines stay 1:1
    stdin = "\n".join(h.replace("\r", " ").replace("\n", " ") for h in headlines)
    result = subprocess.run(
        [BINARY, "--batch"],
        input=stdin,
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=BASE_DIR,
    )

    lines = result.stdout.splitlines()
    if len(lines) < len(headlines):
        raise HTTPException(
            status_code=500,
            detail=f"batch inference failed: {result.stderr.strip() or result.stdout.strip()}",
        )

    out = []
    for ln in lines[: len(headlines)]:
        parts = ln.split("\t")
        if len(parts) >= 2:
            out.append({"label": parts[0], "confidence": float(parts[1])})
        else:
            out.append({"label": "UNDETERMINED", "confidence": 0.5})
    return out
