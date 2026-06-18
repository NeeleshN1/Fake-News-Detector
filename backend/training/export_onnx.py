"""
export_onnx.py — export the fine-tuned DistilBERT to ONNX, INT8-quantize it, and emit
the artifacts the C++ inference engine needs.

Produces (in backend/model/):
    headline.onnx        fp32 ONNX graph (inputs: input_ids, attention_mask; output: logits)
    headline.int8.onnx   dynamic INT8 quantized graph (~1/4 size, fits Render free tier)
    vocab.txt            WordPiece vocab (token per line; line number == id)
    config.json          tokenizer/runtime params for the C++ side

DistilBERT has NO token_type_ids, so the graph takes exactly two int64 inputs.
"""

import os
import json
import shutil
import numpy as np
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
SRC = os.path.join(BACKEND, "model", "transformer")
MODEL_DIR = os.path.join(BACKEND, "model")

FP32 = os.path.join(MODEL_DIR, "headline.onnx")
INT8 = os.path.join(MODEL_DIR, "headline.int8.onnx")
VOCAB_OUT = os.path.join(MODEL_DIR, "vocab.txt")
CONFIG_OUT = os.path.join(MODEL_DIR, "config.json")

MAX_LEN = 64
PARITY = [
    "Federal Reserve holds interest rates steady amid inflation concerns",
    "FIFA awards Donald Trump Golden Boot before 2026 World Cup even begins",
    "BREAKING: Scientists confirm the moon is made entirely of cheese",
    "Maduro arrives in New York to face federal charges",
]


def softmax_fake(logits):
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    p = e / e.sum(axis=-1, keepdims=True)
    return p[:, 1]


class LogitsWrapper(torch.nn.Module):
    """Return the raw logits tensor so torch.onnx.export emits a clean 'logits' output."""
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids, attention_mask):
        return self.m(input_ids=input_ids, attention_mask=attention_mask).logits


def main():
    tok = AutoTokenizer.from_pretrained(SRC)
    model = AutoModelForSequenceClassification.from_pretrained(SRC).eval().cpu()
    wrapped = LogitsWrapper(model).eval()

    # ---- export fp32 ONNX ----
    enc = tok("placeholder headline text", return_tensors="pt", max_length=MAX_LEN,
              truncation=True, padding="max_length")
    inputs = (enc["input_ids"], enc["attention_mask"])
    torch.onnx.export(
        wrapped, inputs, FP32,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "logits": {0: "batch"},
        },
        opset_version=17, do_constant_folding=True,
    )
    print(f"wrote {FP32} ({os.path.getsize(FP32)/1e6:.1f} MB)", flush=True)

    # ---- dynamic INT8 quantization ----
    quantize_dynamic(FP32, INT8, weight_type=QuantType.QInt8)
    print(f"wrote {INT8} ({os.path.getsize(INT8)/1e6:.1f} MB)", flush=True)

    # ---- vocab + config for C++ ----
    # v5 fast tokenizers may not save legacy vocab.txt; reconstruct it from the vocab
    # (one token per line; line index == token id, which is what the C++ tokenizer expects).
    vocab = tok.get_vocab()  # token -> id
    id2tok = {i: t for t, i in vocab.items()}
    with open(VOCAB_OUT, "w", encoding="utf-8") as f:
        for i in range(len(id2tok)):
            f.write(id2tok[i] + "\n")
    print(f"wrote {VOCAB_OUT} ({len(id2tok)} tokens)", flush=True)

    config = {
        "model_type": "distilbert",
        "onnx_model": "headline.int8.onnx",
        "vocab_file": "vocab.txt",
        "max_len": MAX_LEN,
        "do_lower_case": bool(getattr(tok, "do_lower_case", True)),
        "strip_accents": True,           # BERT uncased basic tokenizer strips accents
        "unk_token": tok.unk_token,
        "cls_token": tok.cls_token,
        "sep_token": tok.sep_token,
        "pad_token": tok.pad_token,
        "unk_id": tok.unk_token_id,
        "cls_id": tok.cls_token_id,
        "sep_id": tok.sep_token_id,
        "pad_id": tok.pad_token_id,
        "wordpiece_prefix": "##",
        "id2label": {0: "REAL", 1: "FAKE"},
        "fake_label_index": 1,
    }
    with open(CONFIG_OUT, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"wrote {CONFIG_OUT}", flush=True)

    # ---- parity check: torch vs fp32 onnx vs int8 onnx ----
    print("\n================ PARITY (P(fake)) ================")
    sess_fp32 = ort.InferenceSession(FP32, providers=["CPUExecutionProvider"])
    sess_int8 = ort.InferenceSession(INT8, providers=["CPUExecutionProvider"])
    print(f"{'torch':>8} {'onnx32':>8} {'onnx8':>8}   headline")
    for text in PARITY:
        e = tok(text, return_tensors="pt", max_length=MAX_LEN, truncation=True)
        with torch.no_grad():
            p_torch = torch.softmax(model(**e).logits, dim=-1)[0, 1].item()
        feed = {"input_ids": e["input_ids"].numpy().astype(np.int64),
                "attention_mask": e["attention_mask"].numpy().astype(np.int64)}
        p32 = softmax_fake(sess_fp32.run(None, feed)[0])[0]
        p8 = softmax_fake(sess_int8.run(None, feed)[0])[0]
        print(f"{p_torch:8.3f} {p32:8.3f} {p8:8.3f}   {text}")
    print("\n(int8 may differ from torch by a few hundredths — that is expected.)")


if __name__ == "__main__":
    main()
