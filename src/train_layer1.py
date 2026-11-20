"""Layer 1: classify each dialogue act as Actionable item / Decision (1) or General discussion (0).

Compares three approaches on the same held-out test meetings (Lab 3, sections 6 and 8):
  rules     keyword rules: the "simple rule-based logic" Lab 3 argued is not enough
  tfidf     TF-IDF + logistic regression: a classical statistical baseline
  roberta   fine-tuned DistilRoBERTa with the previous utterances as context (transfer learning)

The decision threshold is tuned on the dev split to reach the Lab 4 target of >= 90% of action items
captured, favouring recall as Lab 3 section 8.3 specifies. Test meetings are never used for tuning.

Usage:  python src/train_layer1.py [--skip-roberta] [--epochs 3]
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"
TARGET_ITEM_RECALL = 0.90  # Lab 4: capture at least 90% of human-verified action items and decisions
CONTEXT_TURNS = 2
NEG_RATIO = 8  # sampled negatives per positive when fine-tuning the transformer

KEYWORDS = re.compile(
    r"\b(?:will|we'll|i'll|you'll|going to|gonna|need to|have to|should|must|let's|lets|decide[ds]?|decision|"
    r"agree[ds]?|deadline|by (?:monday|tuesday|wednesday|thursday|friday|tomorrow|next)|next (?:meeting|time)|"
    r"action|task|assign(?:ed)?|responsible|in charge)\b",
    re.I,
)


def load():
    df = pd.read_csv(DATA / "dialogue_acts.csv")
    df["text"] = df["text"].astype(str)
    df["native_english"] = (df.native_language == "English").map({True: "yes", False: "no"})
    # Previous turns in the meeting give the classifier conversational context.
    df["context"] = df.groupby("meeting")["text"].transform(
        lambda s: pd.concat([s.shift(k) for k in range(CONTEXT_TURNS, 0, -1)], axis=1)
        .fillna("").agg(" ".join, axis=1).str.strip())
    summ = pd.read_csv(DATA / "summary_sentences.csv").fillna({"linked_das": ""})
    summ = summ[summ.linked_das != ""]
    return df, summ


def item_recall(passed_ids: set, summ: pd.DataFrame) -> float:
    """Share of human-written action/decision sentences with at least one supporting DA passed to Layer 2."""
    caught = summ.linked_das.str.split(";").apply(lambda das: any(d in passed_ids for d in das))
    return float(caught.mean())


def pick_threshold(scores, df, summ):
    """Highest threshold on dev that still captures >= TARGET_ITEM_RECALL of action items/decisions."""
    best = None
    for t in np.unique(np.round(scores, 4))[::-1]:
        if item_recall(set(df.da_id[scores >= t]), summ) >= TARGET_ITEM_RECALL:
            best = float(t)
            break
    return best if best is not None else float(scores.min())


def evaluate(name, scores, threshold, df, summ):
    pred = (scores >= threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(df.label, pred, average="binary", zero_division=0)
    words = df.text.str.split().str.len()
    passed = set(df.da_id[pred == 1])
    out = {
        "model": name, "threshold": round(threshold, 4),
        "da_precision": round(p, 4), "da_recall": round(r, 4), "da_f1": round(f1, 4),
        "pr_auc": round(average_precision_score(df.label, scores), 4),
        "item_recall_all": round(item_recall(passed, summ), 4),
        "item_recall_actions": round(item_recall(passed, summ[summ.type == "actions"]), 4),
        "item_recall_decisions": round(item_recall(passed, summ[summ.type == "decisions"]), 4),
        "share_of_das_passed": round(pred.mean(), 4),
        "share_of_words_passed": round(words[pred == 1].sum() / words.sum(), 4),
    }
    groups = {}
    for col in ["sex", "native_english", "role"]:
        for val, g in df.assign(pred=pred).groupby(col):
            if g.label.sum() >= 20:  # too few positives makes a group's recall meaningless
                groups[f"{col}={val}"] = {"positives": int(g.label.sum()),
                                          "recall": round(float(g.pred[g.label == 1].mean()), 4)}
    out["recall_by_group"] = groups
    return out, pred


def run_rules(df):
    return df.text.str.contains(KEYWORDS).astype(float).to_numpy()


def run_tfidf(train, others):
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    X = vec.fit_transform(train.context + " || " + train.text)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0).fit(X, train.label)
    return [clf.predict_proba(vec.transform(d.context + " || " + d.text))[:, 1] for d in others]


def run_roberta(train, dev, test, epochs, base="distilbert/distilroberta-base"):
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForSequenceClassification.from_pretrained(base, num_labels=2).to(device)

    def batches(d, shuffle, bs=64):
        enc = tok(list(d.text), list(d.context), truncation=True, max_length=128, padding="max_length",
                  return_tensors="pt")
        ds = torch.utils.data.TensorDataset(enc["input_ids"], enc["attention_mask"], torch.tensor(d.label.values))
        return DataLoader(ds, batch_size=bs, shuffle=shuffle)

    # Positives are under 2% of utterances; training on all of them would collapse to "always 0".
    # Keep every positive plus a random 1:NEG_RATIO sample of negatives. The threshold is tuned on the
    # untouched dev split afterwards, so the shifted class balance does not bias the reported results.
    pos = train[train.label == 1]
    neg = train[train.label == 0].sample(n=min(len(pos) * NEG_RATIO, (train.label == 0).sum()), random_state=42)
    train = pd.concat([pos, neg]).sample(frac=1, random_state=42)
    print(f"  fine-tuning on {len(pos)} positives + {len(neg)} sampled negatives", flush=True)
    loss_fn = torch.nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    train_dl = batches(train, True, 32)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * len(train_dl) * epochs), len(train_dl) * epochs)

    def score(d):
        model.eval()
        out = []
        with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
            for ids, mask, _ in batches(d, False, 256):
                logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
                out.append(torch.softmax(logits.float(), -1)[:, 1].cpu().numpy())
        return np.concatenate(out)

    best_ap, best_state = -1, None
    for epoch in range(epochs):
        model.train()
        for step, (ids, mask, y) in enumerate(train_dl):
            with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
                logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
            loss = loss_fn(logits.float(), y.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            if step % 100 == 0:
                print(f"  epoch {epoch + 1} step {step}/{len(train_dl)} loss {loss.item():.4f}", flush=True)
        ap = average_precision_score(dev.label, score(dev))
        print(f"  epoch {epoch + 1}: dev PR-AUC {ap:.4f}", flush=True)
        if ap > best_ap:  # keep the epoch that does best on dev, never on test
            best_ap, best_state = ap, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    MODELS.mkdir(exist_ok=True)
    model.save_pretrained(MODELS / "layer1"); tok.save_pretrained(MODELS / "layer1")
    return score(dev), score(test)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-roberta", action="store_true")
    ap.add_argument("--epochs", type=int, default=3)
    args = ap.parse_args()

    df, summ = load()
    train, dev, test = (df[df.split == s].reset_index(drop=True) for s in ("train", "dev", "test"))
    summ_dev, summ_test = summ[summ.split == "dev"], summ[summ.split == "test"]
    print({s: (len(d), int(d.label.sum())) for s, d in [("train", train), ("dev", dev), ("test", test)]})

    scores = {"rules": (run_rules(dev), run_rules(test))}
    scores["tfidf"] = tuple(run_tfidf(train, [dev, test]))
    if not args.skip_roberta:
        scores["roberta"] = run_roberta(train, dev, test, args.epochs)

    RESULTS.mkdir(exist_ok=True)
    report, preds = [], test[["meeting", "da_id", "speaker", "start", "end", "text", "label"]].copy()
    for name, (s_dev, s_test) in scores.items():
        threshold = 0.5 if name == "rules" else pick_threshold(s_dev, dev, summ_dev)
        res, pred = evaluate(name, s_test, threshold, test, summ_test)
        report.append(res)
        preds[f"{name}_score"], preds[f"{name}_pass"] = s_test, pred
        print(json.dumps(res, indent=2))
    (RESULTS / "layer1_metrics.json").write_text(json.dumps(report, indent=2))
    preds.to_csv(RESULTS / "layer1_test_predictions.csv", index=False)
    cols = ["model", "da_precision", "da_recall", "da_f1", "pr_auc", "item_recall_all", "share_of_words_passed"]
    print(pd.DataFrame(report)[cols].to_string(index=False))


if __name__ == "__main__":
    main()
