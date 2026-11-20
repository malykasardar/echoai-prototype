"""Build EchoAI's labelled dataset from the AMI Meeting Corpus manual annotations (v1.6.2, CC BY 4.0).

One row per dialogue act (DA), i.e. one utterance by one speaker. A DA is labelled
  label_action   = 1 if annotators linked it to an "actions" sentence in the meeting's abstractive summary
  label_decision = 1 if linked to a "decisions" sentence
  label          = 1 if either (Layer 1's "Actionable item / Decision" class, Lab 3 section 6.1)

Outputs (data/processed/):
  dialogue_acts.csv      every DA from meetings that have an abstractive summary
  summary_sentences.csv  the human-written action/decision sentences, with the DA ids they link to
                         (the reference for evaluating Layer 2 extraction)
"""
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
AMI = ROOT / "data" / "raw" / "ami"
OUT = ROOT / "data" / "processed"
NITE = "{http://nite.sourceforge.net/}"

# Standard AMI full-corpus partition by meeting series: test and dev series use different
# participants from training, so the models are scored on unseen speakers and teams.
DEV_SERIES = {"ES2011", "IS1008", "TS3004", "IB4001", "IB4002", "IB4003", "IB4004", "IB4010", "IB4011"}
TEST_SERIES = {"ES2004", "IS1009", "TS3003", "EN2002"}

HREF_RANGE = re.compile(r"id\(([^)]+)\)(?:\.\.id\(([^)]+)\))?")


def split_for(meeting: str) -> str:
    series = re.match(r"[A-Z]+\d+", meeting).group(0)
    if series in TEST_SERIES:
        return "test"
    if series in DEV_SERIES:
        return "dev"
    return "train"


def load_speakers() -> dict:
    """(meeting, agent letter) -> role and participant demographics."""
    people = {}
    for p in ET.parse(AMI / "corpusResources" / "participants.xml").getroot().iter("participant"):
        people[p.get(f"{NITE}id")] = {"sex": p.get("sex"), "native_language": p.get("native_language")}
    speakers = {}
    for m in ET.parse(AMI / "corpusResources" / "meetings.xml").getroot().iter("meeting"):
        for s in m.iter("speaker"):
            info = people.get(s.get("global_name"), {})
            speakers[(m.get("observation"), s.get("nxt_agent"))] = {
                "role": s.get("role"),
                "participant": s.get("global_name"),
                "sex": info.get("sex"),
                "native_language": info.get("native_language"),
            }
    return speakers


def load_da_types() -> dict:
    types = {}
    for el in ET.parse(AMI / "ontologies" / "da-types.xml").getroot().iter():
        if el.get(f"{NITE}id") and el.get("gloss"):
            types[el.get(f"{NITE}id")] = el.get("gloss")
    return types


def load_words(path: Path):
    """Return document-ordered element ids, id -> index, and per-element (text, start, end)."""
    order, info = [], {}
    for el in ET.parse(path).getroot():
        eid = el.get(f"{NITE}id")
        if eid is None:
            continue
        order.append(eid)
        text = el.text if el.tag == "w" else None
        info[eid] = (text, el.get("punc") == "true", el.get("starttime"), el.get("endtime"))
    return order, {e: i for i, e in enumerate(order)}, info


def da_text(href: str, order, index, info):
    m = HREF_RANGE.search(href)
    first, last = m.group(1), m.group(2) or m.group(1)
    ids = order[index[first]: index[last] + 1]
    out, starts, ends = "", [], []
    for eid in ids:
        text, punc, st, en = info[eid]
        if st:
            starts.append(float(st))
        if en:
            ends.append(float(en))
        if text is None:
            continue
        out += text if (punc or not out) else " " + text
    return out.strip(), (min(starts) if starts else None), (max(ends) if ends else None)


def load_summary(meeting: str):
    """Abstractive sentences of type actions/decisions, and DA id -> set of linked types."""
    sentences = {}
    root = ET.parse(AMI / "abstractive" / f"{meeting}.abssumm.xml").getroot()
    for section in root:
        if section.tag in ("actions", "decisions"):
            for s in section.iter("sentence"):
                sentences[s.get(f"{NITE}id")] = {"type": section.tag, "text": (s.text or "").strip(), "das": []}
    da_links = defaultdict(set)
    link_file = AMI / "extractive" / f"{meeting}.summlink.xml"
    if link_file.exists():
        for link in ET.parse(link_file).getroot().iter("summlink"):
            ptrs = {p.get("role"): HREF_RANGE.search(p.get("href")).group(1) for p in link.iter(f"{NITE}pointer")}
            sid, da = ptrs.get("abstractive"), ptrs.get("extractive")
            if sid in sentences and da:
                sentences[sid]["das"].append(da)
                da_links[da].add(sentences[sid]["type"])
    return sentences, da_links


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    speakers, da_types = load_speakers(), load_da_types()
    meetings = sorted(p.name.split(".")[0] for p in (AMI / "abstractive").glob("*.abssumm.xml"))
    rows, summ_rows = [], []
    for meeting in meetings:
        sentences, da_links = load_summary(meeting)
        for sid, s in sentences.items():
            summ_rows.append({"meeting": meeting, "split": split_for(meeting), "sentence_id": sid,
                              "type": s["type"], "text": s["text"], "linked_das": ";".join(s["das"])})
        for da_file in sorted((AMI / "dialogueActs").glob(f"{meeting}.*.dialog-act.xml")):
            agent = da_file.name.split(".")[1]
            order, index, info = load_words(AMI / "words" / f"{meeting}.{agent}.words.xml")
            spk = speakers.get((meeting, agent), {})
            for da in ET.parse(da_file).getroot().iter("dact"):
                child = da.find(f"{NITE}child")
                if child is None:
                    continue
                text, start, end = da_text(child.get("href"), order, index, info)
                if not text:
                    continue
                aspect = da.find(f"{NITE}pointer[@role='da-aspect']")
                da_id = da.get(f"{NITE}id")
                links = da_links.get(da_id, set())
                rows.append({
                    "meeting": meeting, "split": split_for(meeting), "da_id": da_id, "speaker": agent,
                    **spk, "start": start, "end": end, "text": text,
                    "da_type": da_types.get(HREF_RANGE.search(aspect.get("href")).group(1)) if aspect is not None else None,
                    "label_action": int("actions" in links), "label_decision": int("decisions" in links),
                    "label": int(bool(links)),
                })
    df = pd.DataFrame(rows).sort_values(["meeting", "start"], kind="stable")
    df.to_csv(OUT / "dialogue_acts.csv", index=False)
    pd.DataFrame(summ_rows).to_csv(OUT / "summary_sentences.csv", index=False)

    print(f"meetings: {df.meeting.nunique()}  dialogue acts: {len(df)}")
    print(df.groupby("split").agg(meetings=("meeting", "nunique"), das=("da_id", "size"),
                                  positive=("label", "sum"), pos_rate=("label", "mean")).round(4))
    print("summary sentences by type:\n", pd.DataFrame(summ_rows).groupby(["split", "type"]).size())


if __name__ == "__main__":
    main()
