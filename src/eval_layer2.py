"""Evaluate the full Layer 1 -> Layer 2 pipeline on the held-out AMI test meetings.

Reference: the human-written "actions" and "decisions" sentences in each meeting's AMI summary,
with the dialogue acts annotators linked them to.

  coverage        a human item counts as found if an extracted item of any type cites one of its linked DAs
  owner accuracy  for found action items whose human sentence names a role, does the extracted owner match?
  grounding       share of extracted items that fail an automatic check (invalid evidence, unknown owner, ...)
  cost            prompt tokens and seconds per meeting, with Layer 1 filtering versus the whole transcript

Usage:  python src/eval_layer2.py [--layer1 roberta|tfidf] [--no-filter]
"""
import argparse
import json
import os
import random
import re
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from layer2 import make_extractor

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CHUNK = 60  # flagged utterances per model call, keeps each answer short enough to finish
ROLE_WORDS = {"project manager": "PM", "industrial designer": "ID", "user interface": "UI", "interface designer": "UI",
              "marketing": "ME"}


def role_in_sentence(text):
    found = {code for words, code in ROLE_WORDS.items() if words in text.lower()}
    return found.pop() if len(found) == 1 else None


def run_meeting(ex, m_df, use_filter, layer1):
    m_df = m_df.sort_values("start", kind="stable").reset_index(drop=True)
    utts = [{"speaker": r.role if isinstance(r.role, str) else r.speaker, "text": r.text, "da_id": r.da_id,
             "start": r.start} for r in m_df.itertuples()]
    flagged = [i for i, p in enumerate(m_df[f"{layer1}_pass"]) if p] if use_filter else list(range(len(utts)))
    items, errors, tokens, t0 = [], [], 0, time.perf_counter()
    for c in range(0, len(flagged), CHUNK):
        out = ex.extract(utts, flagged[c:c + CHUNK])
        items += out["items"]; errors += out["errors"]; tokens += out["prompt_tokens"]
    return utts, items, errors, tokens, time.perf_counter() - t0, len(flagged)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer1", default="roberta")
    ap.add_argument("--no-filter", action="store_true", help="send every utterance (cost comparison)")
    ap.add_argument("--split", default="test", choices=["dev", "test"], help="tune prompts on dev, report on test")
    ap.add_argument("--meetings", type=int, default=None, help="only the first N meetings (quick runs)")
    args = ap.parse_args()
    backend = os.environ.get("ECHOAI_BACKEND", "ollama")
    tag = ("nofilter" if args.no_filter else args.layer1) + ("_ollama" if backend == "ollama" else "")
    tag += "_dev" if args.split == "dev" else ""

    das = pd.read_csv(ROOT / "data" / "processed" / "dialogue_acts.csv")
    if args.split == "test":
        preds = pd.read_csv(RESULTS / "layer1_test_predictions.csv")
        test = das[das.split == "test"].merge(preds[["da_id"] + [c for c in preds if c.endswith("_pass")]], on="da_id")
    else:  # dev predictions are not saved by train_layer1.py, so score them with the saved Layer 1 model
        from pipeline import Layer1
        l1, test = Layer1(), das[das.split == "dev"].sort_values(["meeting", "start"], kind="stable").copy()
        test["roberta_pass"] = 0
        for m, g in test.groupby("meeting"):
            s = l1.score([{"text": t} for t in g.text.astype(str)])
            test.loc[g.index, "roberta_pass"] = [int(x >= l1.threshold) for x in s]
        del l1
    if args.meetings:
        test = test[test.meeting.isin(sorted(test.meeting.unique())[:args.meetings])]
    summ = pd.read_csv(ROOT / "data" / "processed" / "summary_sentences.csv").fillna({"linked_das": ""})
    summ = summ[(summ.split == args.split) & (summ.linked_das != "") & summ.meeting.isin(test.meeting.unique())]

    ex = make_extractor()
    rows, per_meeting = [], []
    for meeting, m_df in test.groupby("meeting"):
        utts, items, errors, tokens, secs, n_flag = run_meeting(ex, m_df, not args.no_filter, args.layer1)
        per_meeting.append({"meeting": meeting, "utterances": len(utts), "sent_to_llm": n_flag, "prompt_tokens": tokens,
                            "seconds": round(secs, 1), "items": len(items), "json_errors": len(errors)})
        print(per_meeting[-1], flush=True)
        for it in items:
            ev_idx = [int(e[1:]) for e in it.evidence if re.fullmatch(r"U\d+", e) and int(e[1:]) < len(utts)]
            rows.append({"meeting": meeting, **asdict(it), "problems": ";".join(it.problems),
                         "evidence_das": ";".join(utts[i]["da_id"] for i in ev_idx),
                         "evidence_text": " | ".join(f"{utts[i]['speaker']}: {utts[i]['text']}" for i in ev_idx),
                         "start": utts[ev_idx[0]]["start"] if ev_idx else None})
    items = pd.DataFrame(rows)
    items["evidence"] = items.evidence.apply(";".join)
    items.to_csv(RESULTS / f"layer2_items_{tag}.csv", index=False)

    # Coverage and owner accuracy against the human summaries.
    cited = {}
    for r in items.itertuples():
        for d in str(r.evidence_das).split(";"):
            cited.setdefault((r.meeting, d), []).append(r)
    found, owner_hits, owner_total = [], 0, 0
    for s in summ.itertuples():
        matches = [it for d in s.linked_das.split(";") for it in cited.get((s.meeting, d), [])]
        found.append(bool(matches))
        role = role_in_sentence(s.text)
        if s.type == "actions" and matches and role:
            owner_total += 1
            owner_hits += any(role in [o.strip() for o in str(it.owner).split(",")] for it in matches)
    summ = summ.assign(found=found)
    reference_das = {(s.meeting, d) for s in summ.itertuples() for d in s.linked_das.split(";")}
    supported = items.apply(lambda r: any((r.meeting, d) in reference_das for d in str(r.evidence_das).split(";")), axis=1)
    pm = pd.DataFrame(per_meeting)
    report = {
        "condition": tag,
        "meetings": len(pm), "extracted_items": len(items),
        "coverage_all": round(summ.found.mean(), 4),
        "coverage_actions": round(summ[summ.type == "actions"].found.mean(), 4),
        "coverage_decisions": round(summ[summ.type == "decisions"].found.mean(), 4),
        "reference_items": len(summ),
        "owner_accuracy_actions": round(owner_hits / owner_total, 4) if owner_total else None,
        "owner_checked": owner_total,
        "items_citing_a_reference_line": round(supported.mean(), 4),
        "items_failing_auto_checks": round((items.problems != "").mean(), 4),
        "auto_check_failures": items.problems[items.problems != ""].str.split(";").explode().value_counts().to_dict(),
        "json_errors": int(pm.json_errors.sum()),
        "share_of_utterances_sent": round(pm.sent_to_llm.sum() / pm.utterances.sum(), 4),
        "prompt_tokens_per_meeting": round(pm.prompt_tokens.mean()),
        "seconds_per_meeting": round(pm.seconds.mean(), 1),
    }
    (RESULTS / f"layer2_metrics_{tag}.json").write_text(json.dumps({"summary": report, "per_meeting": per_meeting}, indent=2))
    print(json.dumps(report, indent=2))

    if not args.no_filter and args.split == "test":
        # Sample for the team's manual review: measures the hallucination rate (Lab 4 target: under 2%).
        random.seed(7)
        sample = items.sample(min(50, len(items)), random_state=7)[
            ["meeting", "start", "type", "description", "owner", "deadline", "evidence_text"]]
        for col in ["supported_by_evidence (y/n)", "owner_correct (y/n/na)", "deadline_correct (y/n/na)", "reviewer", "notes"]:
            sample[col] = ""
        sample.to_csv(RESULTS / "manual_review_sample.csv", index=False)


if __name__ == "__main__":
    main()
