"""Measure the precision filter (review_filter.py) on a saved Layer 2 run, without a GPU.

Re-scores the items in results/layer2_items_<tag>.csv and compares "everything drafted" with "shown by default".

  items per meeting      what the reviewer has to read
  items citing a ref.    share of items that cite a line AMI annotators linked to a summary item (a lower bound
                         on precision: AMI summaries list main outcomes, not every commitment)
  reference lines cited  how many of those annotated lines are still cited (a recall check)
  coverage               share of AMI summary items found, as in eval_layer2.py (needs data/processed/)

Usage:  python src/eval_filter.py [--tag roberta_ollama]
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from layer2 import Item
from review_filter import POSITIVE, SHOW_AT, assess

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def load(tag):
    items = pd.read_csv(RESULTS / f"layer2_items_{tag}.csv").fillna({"evidence_das": "", "problems": "", "owner": "",
                                                                    "deadline": ""})
    preds = pd.read_csv(RESULTS / "layer1_test_predictions.csv")
    return items, preds


def rescore(items, preds, show_at=SHOW_AT):
    """Rebuild each meeting's utterance list from the saved Layer 1 predictions and re-run the filter."""
    out = []
    for meeting, g in items.groupby("meeting"):
        m = preds[preds.meeting == meeting].sort_values("start", kind="stable").reset_index(drop=True)
        pos = {d: i for i, d in enumerate(m.da_id)}
        utts = [{"speaker": r.speaker, "text": str(r.text), "start": r.start} for r in m.itertuples()]
        scores = m.roberta_score.tolist()
        objs = []
        for r in g.itertuples():
            ev = [f"U{pos[d]}" for d in str(r.evidence_das).split(";") if d in pos]
            objs.append(Item(type=r.type, description=str(r.description), owner=r.owner or None,
                             deadline=r.deadline or None, evidence=ev, confidence=r.confidence,
                             # owner_not_in_meeting in older saved runs came from a validator bug fixed since
                             # (and speakers here are AMI letters, not roles), so it is not re-applied
                             problems=[p for p in str(r.problems).split(";") if p and p != "owner_not_in_meeting"]))
            objs[-1].row = r.Index
        for it in assess(objs, utts, scores, show_at):
            out.append({"row": it.row, "score": it.score, "hidden": it.hidden, "reasons": "; ".join(it.reasons)})
    return items.join(pd.DataFrame(out).set_index("row"))


def metrics(items, preds, summ, label):
    ref = set(preds.da_id[preds.label == 1])
    das = items.evidence_das.str.split(";")
    cites_ref = das.apply(lambda ds: any(d in ref for d in ds))
    cited = {d for ds in das for d in ds if d in ref}
    r = {"set": label, "items": len(items), "items_per_meeting": round(len(items) / items.meeting.nunique(), 1),
         "decisions": int((items.type == "decision").sum()), "actions": int((items.type == "action").sum()),
         "items_citing_a_reference_line": round(cites_ref.mean(), 4),
         "reference_lines_cited": len(cited), "reference_lines_total": len(ref)}
    if summ is not None:
        keys = {(m, d) for m, ds in zip(items.meeting, das) for d in ds}
        found = summ.apply(lambda s: any((s.meeting, d) in keys for d in s.linked_das.split(";")), axis=1)
        r["coverage_all"] = round(found.mean(), 4)
        r["coverage_actions"] = round(found[summ.type == "actions"].mean(), 4)
        r["coverage_decisions"] = round(found[summ.type == "decisions"].mean(), 4)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="roberta_ollama")
    ap.add_argument("--show-at", type=int, default=SHOW_AT)
    args = ap.parse_args()
    items, preds = load(args.tag)
    summ_path = ROOT / "data" / "processed" / "summary_sentences.csv"
    summ = None
    if summ_path.exists():
        summ = pd.read_csv(summ_path).fillna({"linked_das": ""})
        summ = summ[(summ.split == "test") & (summ.linked_das != "") & summ.meeting.isin(items.meeting.unique())]
    scored = rescore(items, preds, args.show_at)
    before = metrics(scored, preds, summ, "all drafted (before)")
    after = metrics(scored[~scored.hidden], preds, summ, f"shown, score >= {args.show_at} (after)")
    reasons = scored[scored.hidden].reasons.fillna("").str.split("; ").explode()
    reasons = reasons.str.replace(r"^repeats: .*", "repeat", regex=True)
    hidden_reasons = reasons[(reasons != "") & ~reasons.isin(POSITIVE)].value_counts().to_dict()
    report = {"tag": args.tag, "show_at": args.show_at, "before": before, "after": after,
              "why_hidden_items_scored_low": hidden_reasons}
    scored.to_csv(RESULTS / f"filter_items_{args.tag}.csv", index=False)
    (RESULTS / f"filter_metrics_{args.tag}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
