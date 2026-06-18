"""
prepare_data.py — build a clean, balanced headline dataset for fake-news classification.

Output label convention (matches the UI's fake-likelihood score):
    label 1 = FAKE
    label 0 = REAL

Sources (both use 0=fake / 1=real on the Hub, so we FLIP to our convention):
    - davanstrien/WELFake          (primary; 72k news titles, balanced)
    - GonzaloA/fake_news           (supplementary news titles; deduped against WELFake)

Writes data/processed/{train,val,test}.csv with columns: headline,label
and prints verification samples per FINAL label so polarity can be eyeballed.
"""

import os
import re
import sys
import pandas as pd

from datasets import load_dataset, concatenate_datasets
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
OUT_DIR = os.path.join(BACKEND, "data", "processed")
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_STATE = 42
TOTAL_CAP = 60000          # cap balanced corpus for fast fine-tuning on a 4GB GPU
MIN_WORDS = 3
MAX_WORDS = 40

# Each source: (hf_name, title_col_candidates, label_col_candidates, flip)
# flip=True means the source uses 1=real/0=fake, so we invert to our 1=fake/0=real.
# Verified empirically (the two sources use OPPOSITE conventions):
#   WELFake  -> raw 1=fake, 0=real  (no flip)
#   GonzaloA -> raw 0=fake, 1=real  (flip)
SOURCES = [
    ("davanstrien/WELFake", ["title", "Title"], ["label", "Label"], False),
    ("GonzaloA/fake_news",  ["title", "Title"], ["label", "Label"], True),
]

# Real-only sources (every row is REAL, label 0). These broaden the REAL class beyond
# 2016 political wire copy so modern, punchy, multi-domain headlines (tech/sports/health/
# entertainment) are not mistaken for fake. (hf_name, text_col_candidates, cap)
REAL_SOURCES = [
    ("sh0416/ag_news",               ["title"],    40000),  # world/sports/business/sci-tech
    ("heegyu/news-category-dataset", ["headline"], 50000),  # modern, diverse (HuffPost)
]

_SOURCE_TAG = re.compile(r"\s*[\(\[]?\s*(reuters|ap|afp|breitbart|cnn)\s*[\)\]]?\s*$", re.I)
_WS = re.compile(r"\s+")


def _pick(colnames, candidates):
    for c in candidates:
        if c in colnames:
            return c
    return None


def load_source(hf_name, title_cands, label_cands, flip):
    """Load one HF dataset, return DataFrame[headline,label,source] in our convention."""
    print(f"\n--- loading {hf_name} ---", flush=True)
    ds = load_dataset(hf_name)
    parts = [ds[s] for s in ds.keys()]
    merged = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
    cols = merged.column_names
    tcol = _pick(cols, title_cands)
    lcol = _pick(cols, label_cands)
    if tcol is None or lcol is None:
        raise ValueError(f"{hf_name}: could not find title/label cols in {cols}")

    df = merged.to_pandas()[[tcol, lcol]].rename(columns={tcol: "headline", lcol: "label"})
    # raw label distribution (Hub convention)
    print(f"{hf_name}: raw rows={len(df)}, raw label counts (Hub convention)="
          f"{df['label'].value_counts().to_dict()}", flush=True)

    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["headline", "label"])
    df["label"] = df["label"].astype(int)
    if flip:
        df["label"] = 1 - df["label"]   # -> 1=fake, 0=real
    df["source"] = hf_name
    return df


def load_real_source(hf_name, text_cands, cap):
    """Load a real-only dataset, return DataFrame[headline,label=0,source]."""
    print(f"\n--- loading {hf_name} (real-only) ---", flush=True)
    ds = load_dataset(hf_name)
    parts = [ds[s] for s in ds.keys()]
    merged = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
    tcol = _pick(merged.column_names, text_cands)
    if tcol is None:
        raise ValueError(f"{hf_name}: no text col in {merged.column_names}")

    df = merged.to_pandas()[[tcol]].rename(columns={tcol: "headline"}).dropna(subset=["headline"])
    if cap and len(df) > cap:
        df = df.sample(n=cap, random_state=RANDOM_STATE)
    df["label"] = 0
    df["source"] = hf_name
    print(f"{hf_name}: real rows={len(df)}", flush=True)
    return df


def clean_headline(h):
    if not isinstance(h, str):
        return None
    h = h.strip().strip('"').strip()
    h = _SOURCE_TAG.sub("", h)          # drop trailing "(Reuters)" / "- Breitbart" style tags
    h = _WS.sub(" ", h).strip()
    n = len(h.split())
    if n < MIN_WORDS or n > MAX_WORDS:
        return None
    return h


def main():
    frames = []
    for spec in SOURCES:
        try:
            frames.append(load_source(*spec))
        except Exception as e:
            print(f"!! skipping {spec[0]}: {e}", flush=True)
    for spec in REAL_SOURCES:
        try:
            frames.append(load_real_source(*spec))
        except Exception as e:
            print(f"!! skipping {spec[0]}: {e}", flush=True)

    if not frames:
        print("FATAL: no sources loaded", file=sys.stderr)
        sys.exit(1)

    df = pd.concat(frames, ignore_index=True)
    print(f"\ncombined rows={len(df)}", flush=True)

    # clean
    df["headline"] = df["headline"].map(clean_headline)
    df = df.dropna(subset=["headline"])

    # dedup (case-insensitive)
    df["_key"] = df["headline"].str.lower()
    before = len(df)
    df = df.drop_duplicates(subset="_key").drop(columns="_key")
    print(f"after clean+dedup: {len(df)} rows (removed {before - len(df)} dups/invalid)", flush=True)
    print(f"label balance (1=fake,0=real): {df['label'].value_counts().to_dict()}", flush=True)

    # balance 50/50 (and optionally cap) by sampling each class to the same size.
    # NOTE: avoid groupby().apply() — pandas 3.0 drops the grouping column from it.
    per = min(df["label"].value_counts().min(), TOTAL_CAP // 2)

    def take(label):
        return df[df["label"] == label].sample(n=per, random_state=RANDOM_STATE)

    df = (pd.concat([take(0), take(1)])
            .sample(frac=1.0, random_state=RANDOM_STATE)
            .reset_index(drop=True))
    print(f"\nbalanced corpus: {len(df)} rows, "
          f"counts={df['label'].value_counts().to_dict()}", flush=True)

    # stratified split 80/10/10
    train, tmp = train_test_split(df, test_size=0.20, stratify=df["label"],
                                  random_state=RANDOM_STATE)
    val, test = train_test_split(tmp, test_size=0.50, stratify=tmp["label"],
                                 random_state=RANDOM_STATE)

    for name, part in [("train", train), ("val", val), ("test", test)]:
        path = os.path.join(OUT_DIR, f"{name}.csv")
        part[["headline", "label"]].to_csv(path, index=False)
        print(f"wrote {path}: {len(part)} rows, counts={part['label'].value_counts().to_dict()}",
              flush=True)

    # ---- polarity verification (eyeball these) ----
    print("\n================ VERIFY POLARITY ================")
    for lab, name in [(1, "FAKE (label=1)"), (0, "REAL (label=0)")]:
        print(f"\n[{name}] examples:")
        for h in train[train["label"] == lab]["headline"].head(6).tolist():
            print("   ", h)
    print("\nIf label=1 rows look fake/sensational and label=0 look like real reporting, polarity is correct.")


if __name__ == "__main__":
    main()
