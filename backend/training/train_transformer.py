"""
train_transformer.py — fine-tune DistilBERT for binary fake-news headline classification.

Label convention: 1 = FAKE, 0 = REAL.  Model outputs P(fake) = softmax(logits)[1],
which is exactly the "fake-likelihood score" the UI consumes.

Outputs the best model + tokenizer to backend/model/transformer/ and prints
accuracy / F1 / confusion matrix + a real-vs-fake calibration check.
"""

import os
import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, confusion_matrix,
    classification_report,
)

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
DATA = os.path.join(BACKEND, "data", "processed")
OUT = os.path.join(BACKEND, "model", "transformer")
os.makedirs(OUT, exist_ok=True)

MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 64
EPOCHS = 3
BATCH = 16
GRAD_ACCUM = 2
LR = 2e-5

SMOKE = [
    # obviously real (measured wire-style) -> expect LOW score
    ("Federal Reserve holds interest rates steady amid inflation concerns", "real?"),
    ("Carney will meet with Ukraine's allies in Paris as peace talks intensify", "real?"),
    ("Maduro arrives in New York to face federal charges", "real?"),
    # obviously fake / sensational -> expect HIGH score
    ("FIFA awards Donald Trump Golden Boot before 2026 World Cup even begins", "fake?"),
    ("BREAKING: Scientists confirm the moon is made entirely of cheese", "fake?"),
    ("Obama signs executive order banning the national anthem at sporting events", "fake?"),
]


def load_split(name):
    df = pd.read_csv(os.path.join(DATA, f"{name}.csv"))
    df = df.dropna(subset=["headline", "label"])
    df["label"] = df["label"].astype(int)
    return Dataset.from_pandas(df[["headline", "label"]], preserve_index=False)


def main():
    print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
          "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    ds = {s: load_split(s) for s in ["train", "val", "test"]}

    def tokenize(batch):
        return tok(batch["headline"], truncation=True, max_length=MAX_LEN)

    ds = {s: d.map(tokenize, batched=True, remove_columns=["headline"]) for s, d in ds.items()}
    collator = DataCollatorWithPadding(tokenizer=tok)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2,
        id2label={0: "REAL", 1: "FAKE"}, label2id={"REAL": 0, "FAKE": 1},
    )

    def compute_metrics(eval_pred):
        logits = eval_pred.predictions
        labels = eval_pred.label_ids
        if isinstance(logits, tuple):
            logits = logits[0]
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds),
            "precision": precision_score(labels, preds),
            "recall": recall_score(labels, preds),
        }

    args = TrainingArguments(
        output_dir=os.path.join(HERE, "_hf_out"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LR,
        per_device_train_batch_size=BATCH,
        per_device_eval_batch_size=64,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        warmup_ratio=0.1,
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="none",
        save_total_limit=1,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["val"],
        processing_class=tok,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
    )

    trainer.train()

    # ---- test-set evaluation ----
    print("\n================ TEST METRICS ================", flush=True)
    pred = trainer.predict(ds["test"])
    logits = pred.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]  # P(fake)
    y = pred.label_ids
    yhat = (probs >= 0.5).astype(int)

    print("accuracy:", round(accuracy_score(y, yhat), 4),
          "| f1:", round(f1_score(y, yhat), 4))
    print("\nclassification report (1=fake,0=real):\n",
          classification_report(y, yhat, target_names=["REAL", "FAKE"]))
    print("confusion matrix [rows=true REAL,FAKE][cols=pred REAL,FAKE]:\n",
          confusion_matrix(y, yhat))

    # ---- calibration check: this is the direct test that the "everything-fake" bug is gone ----
    print("\n================ CALIBRATION ================")
    print(f"mean P(fake) on REAL test headlines: {probs[y == 0].mean():.3f}  (want << 0.5)")
    print(f"mean P(fake) on FAKE test headlines: {probs[y == 1].mean():.3f}  (want >> 0.5)")

    # ---- smoke test on hand-picked headlines ----
    print("\n================ SMOKE TEST ================")
    model.eval()
    dev = model.device
    with torch.no_grad():
        for text, hint in SMOKE:
            enc = tok(text, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(dev)
            p = torch.softmax(model(**enc).logits, dim=-1)[0, 1].item()
            verdict = "FAKE" if p >= 0.5 else "REAL"
            print(f"  P(fake)={p:.3f} -> {verdict:4} [{hint}]  {text}")

    # ---- save ----
    trainer.save_model(OUT)
    tok.save_pretrained(OUT)
    print(f"\nsaved best model + tokenizer to {OUT}", flush=True)


if __name__ == "__main__":
    main()
