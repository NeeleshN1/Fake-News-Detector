"""
parity_test.py — verify the C++ inference path matches the Python reference.

Two checks:
  1) TOKENIZER parity: C++ `fake_news --ids "<text>"` input_ids == HuggingFace tokenizer ids.
  2) PROBABILITY parity: C++ `fake_news "<text>"` P(fake) ~= Python onnxruntime P(fake).

Run after export_onnx.py and after copying the built exe + dlls + model/ into backend/.
"""

import os
import sys
import subprocess
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
MODEL_DIR = os.path.join(BACKEND, "model")
EXE = os.path.join(BACKEND, "fake_news.exe" if os.name == "nt" else "fake_news")
TOK_SRC = os.path.join(MODEL_DIR, "transformer")
ONNX = os.path.join(MODEL_DIR, "headline.int8.onnx")
MAX_LEN = 64

TESTS = [
    "Federal Reserve holds interest rates steady amid inflation concerns",
    "FIFA awards Donald Trump Golden Boot before 2026 World Cup even begins",
    "BREAKING: Scientists confirm the moon is made entirely of cheese!!!",
    "Maduro arrives in New York to face federal charges",
    "Carney will meet with Ukraine's allies in Paris as peace talks intensify",
    "Café owner's naïve résumé fools the entire town",   # accents/punctuation edge case
    "U.S. stocks edge higher as tech shares rebound",
]


def cpp_ids(text):
    out = subprocess.run([EXE, "--ids", text], cwd=BACKEND, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"fake_news --ids failed: {out.stderr}")
    return [int(x) for x in out.stdout.split()]


def cpp_prob(text):
    out = subprocess.run([EXE, text], cwd=BACKEND, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"fake_news failed: {out.stderr}")
    return float(out.stdout.split()[1])


def softmax_fake(logits):
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return (e / e.sum(axis=-1, keepdims=True))[:, 1]


def main():
    if not os.path.exists(EXE):
        sys.exit(f"missing exe: {EXE} (build + copy it into backend/ first)")
    tok = AutoTokenizer.from_pretrained(TOK_SRC if os.path.isdir(TOK_SRC) else "distilbert-base-uncased")
    sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])

    print("================ TOKENIZER PARITY ================")
    tok_ok = True
    for t in TESTS:
        hf = tok(t, truncation=True, max_length=MAX_LEN)["input_ids"]
        cc = cpp_ids(t)
        match = (hf == cc)
        tok_ok &= match
        print(f"[{'OK ' if match else 'BAD'}] {t}")
        if not match:
            print(f"      HF : {hf}")
            print(f"      C++: {cc}")

    print("\n================ PROBABILITY PARITY ================")
    print(f"{'py_onnx':>8} {'cpp':>8} {'diff':>7}   headline")
    prob_ok = True
    for t in TESTS:
        ids = tok(t, truncation=True, max_length=MAX_LEN)["input_ids"]
        feed = {"input_ids": np.array([ids], dtype=np.int64),
                "attention_mask": np.ones((1, len(ids)), dtype=np.int64)}
        p_py = float(softmax_fake(sess.run(None, feed)[0])[0])
        p_cc = cpp_prob(t)
        diff = abs(p_py - p_cc)
        prob_ok &= diff < 1e-2
        print(f"{p_py:8.3f} {p_cc:8.3f} {diff:7.4f}   {t}")

    print("\nRESULT:",
          "tokenizer", "PASS" if tok_ok else "FAIL",
          "| probability", "PASS (<1e-2)" if prob_ok else "FAIL")
    sys.exit(0 if (tok_ok and prob_ok) else 1)


if __name__ == "__main__":
    main()
