"""Layer 2: turn Layer 1's flagged utterances into structured action items and decisions.

Runs an open instruction-tuned model locally on the GPU, so no meeting text leaves the laptop
(Lab 7 privacy controls). Every item must cite the utterance ids it came from; items are validated
against a fixed schema and against the meeting itself before they are shown (Lab 3 section 6.1).
"""
import json
import re
from dataclasses import dataclass, field

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
ROLES = {"PM": "Project Manager", "ID": "Industrial Designer", "UI": "User Interface Designer",
         "ME": "Marketing Expert"}

SYSTEM = """You extract action items and decisions from meeting transcripts for a meeting summary app.
You receive numbered utterances [U#] that a classifier flagged as possibly actionable, with nearby context.
Return ONLY a JSON object of this form:
{"items": [{"type": "action" | "decision",
            "description": "plain English, at most 15 words",
            "owner": "speaker label of the person responsible, e.g. PM, or null",
            "deadline": "deadline exactly as said, e.g. 'next meeting', or null",
            "evidence": ["U12", "U13"],
            "confidence": "high" | "low"}]}
Rules:
- action = someone commits to or is assigned future work. decision = the group explicitly agrees or settles
  something (e.g. "okay, let's go with that", "so we'll use rubber").
- Suggestions, opinions, questions, ideas and "it should have..." remarks are NOT decisions unless the group
  accepts them. If acceptance is unclear, set confidence to "low".
- Only include items clearly supported by the utterances you cite. Do not guess.
- Cite at most 4 utterances per item: the ones where the commitment or agreement is actually said.
- owner must be one of the speaker labels in the transcript, or null if nobody is clearly responsible.
- deadline must be null unless a time is actually stated.
- Merge repeats of the same item. If there are no items, return {"items": []}."""


@dataclass
class Item:
    type: str
    description: str
    owner: str | None
    deadline: str | None
    evidence: list
    confidence: str
    problems: list = field(default_factory=list)


def build_prompt(utts, flagged_idx, context=1):
    """utts: list of dicts with speaker label and text. Shows flagged utterances plus neighbours."""
    keep = sorted({j for i in flagged_idx for j in range(max(0, i - context), min(len(utts), i + context + 1))})
    lines, prev = [], None
    for i in keep:
        if prev is not None and i != prev + 1:
            lines.append("...")
        mark = "*" if i in flagged_idx else " "
        lines.append(f"{mark}[U{i}] {utts[i]['speaker']}: {utts[i]['text']}")
        prev = i
    speakers = sorted({u["speaker"] for u in utts})
    legend = ", ".join(f"{s} = {ROLES.get(s, s)}" for s in speakers)
    return (f"Speakers: {legend}\nFlagged utterances are marked with *.\n\n" + "\n".join(lines)), keep


def parse_and_validate(raw: str, utts, shown_idx):
    """Schema check plus grounding checks. Items that fail are kept but flagged, never silently fixed."""
    errors = []
    m = re.search(r"\{.*\}", raw, re.S)
    try:
        items_raw = json.loads(m.group(0))["items"] if m else None
        assert isinstance(items_raw, list)
    except Exception:
        # Output cut off or malformed: keep each complete item object rather than losing all of them.
        items_raw, dec = [], json.JSONDecoder()
        for pos in (p.start() for p in re.finditer(r'\{\s*"type"', raw)):
            try:
                items_raw.append(dec.raw_decode(raw, pos)[0])
            except json.JSONDecodeError:
                pass
        errors.append("truncated_json_recovered" if items_raw else "invalid_json")
    speakers = {u["speaker"] for u in utts}
    shown = {f"U{i}" for i in shown_idx}
    items = []
    for it in items_raw:
        if not isinstance(it, dict):
            continue
        item = Item(type=str(it.get("type", "")).lower(), description=str(it.get("description", "")).strip(),
                    owner=it.get("owner") or None, deadline=it.get("deadline") or None,
                    evidence=[str(e) for e in (it.get("evidence") or [])],
                    confidence=str(it.get("confidence", "low")).lower())
        if item.type not in ("action", "decision"):
            item.problems.append("bad_type")
        if not item.description:
            item.problems.append("empty_description")
        if not item.evidence:
            item.problems.append("no_evidence")
        if any(e not in shown for e in item.evidence):
            item.problems.append("evidence_not_in_input")  # cites an utterance it was never shown
        if item.owner and item.owner not in speakers:
            item.problems.append("owner_not_in_meeting")
        if item.deadline:
            cited = " ".join(utts[int(e[1:])]["text"] for e in item.evidence if e in shown).lower()
            if not any(w in cited for w in re.findall(r"[a-z0-9]+", item.deadline.lower()) if len(w) > 2):
                item.problems.append("deadline_not_in_evidence")
        if item.problems:
            item.confidence = "low"
        items.append(item)
    return items, errors


class Extractor:
    def __init__(self, model_id=MODEL_ID):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda")
        self.model.eval()

    def generate(self, user_msg, max_new_tokens=1536):
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_msg}]
        enc = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                           return_dict=True).to("cuda")
        n = enc["input_ids"].shape[1]
        with self.torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
        return self.tok.decode(out[0, n:], skip_special_tokens=True), int(n)

    def extract(self, utts, flagged_idx):
        prompt, shown = build_prompt(utts, flagged_idx)
        raw, n_tokens = self.generate(prompt)
        items, errors = parse_and_validate(raw, utts, shown)
        return {"items": items, "errors": errors, "raw": raw, "prompt_tokens": n_tokens}
