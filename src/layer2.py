"""Layer 2: turn Layer 1's flagged utterances into structured action items and decisions.

Runs an open instruction-tuned model locally on the GPU, so no meeting text leaves the laptop
(Lab 7 privacy controls). Every item must cite the utterance ids it came from; items are validated
against a fixed schema and against the meeting itself before they are shown (Lab 3 section 6.1).
"""
import json
import os
import re
from dataclasses import dataclass, field

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
ROLES = {"PM": "Project Manager", "ID": "Industrial Designer", "UI": "User Interface Designer",
         "ME": "Marketing Expert"}

SYSTEM_V2 = """You extract action items and decisions from meeting transcripts for a meeting summary app.
You receive numbered utterances [U#] that a classifier flagged as possibly actionable, with nearby context.
Return ONLY a JSON object of this form:
{"items": [{"type": "action" | "decision",
            "description": "plain English, at most 15 words",
            "owner": "speaker label(s) of who will DO the work, comma-separated if several, or null",
            "deadline": "a time stated in the cited utterances, copied word for word, or null",
            "evidence": ["U12", "U13"],
            "confidence": "high" | "low"}]}
Rules:
- action = someone commits to or is assigned future work. decision = the group settles something.
- A decision can be an explicit agreement ("okay, let's go with that") or a choice stated as settled that
  nobody objects to ("so it'll be yellow", "our target group is young people", "we'll use rubber").
- Open suggestions, questions and brainstorming ("what if...", "maybe we could...") are NOT decisions.
  If you are unsure whether the group accepted something, include it with confidence "low".
- Owner is the person who will DO the work, not the person who assigns it:
  "I'll do X" -> the speaker. "You'll do X" -> the person being addressed, if the transcript shows who that is.
  "We'll do X" -> null. If you cannot tell who is meant, set owner to null. Never default to the speaker.
- Deadline: copy a time phrase only if it appears in the utterances you cite. Most items have no deadline: null.
- Only include items clearly supported by the utterances you cite. Do not guess.
- Cite at most 4 utterances per item: the ones where the commitment or agreement is actually said.
- Merge repeats of the same item. If there are no items, return {"items": []}."""


# Prompt v3 (precision): v2 drafted ~25 items per meeting against ~6 in the human summary, mostly
# brainstormed features, findings read out from slides, and meeting logistics. v3 names those exclusions and
# asks for a short verbatim quote, which parse_and_validate checks against the cited lines.
SYSTEM_V3 = """You extract action items and decisions from meeting transcripts for a meeting summary app.
You receive numbered utterances [U#] that a classifier flagged as possibly actionable, with nearby context.
Most flagged utterances are NOT items: the classifier is tuned to miss nothing, so expect only a few real items.
Return ONLY a JSON object of this form:
{"items": [{"type": "action" | "decision",
            "description": "plain English, at most 15 words, starting with a verb for actions",
            "owner": "speaker label(s) of who will DO the work, comma-separated if several, or null",
            "deadline": "a time stated in the cited utterances, copied word for word, or null",
            "evidence": ["U12", "U13"],
            "quote": "the shortest exact words from one cited utterance that show the commitment or agreement",
            "confidence": "high" | "low"}]}
What counts:
- action = a specific person commits to, or is given, a concrete task to do after this point in the meeting
  ("I'll check the cost", "you'll do the components concept", "let's each send our sketches by Friday").
- decision = the group settles a choice: explicit agreement ("okay, let's go with that", "we decided on
  rubber") or a choice stated as final that nobody objects to ("so it'll be yellow").
What does NOT count (leave these out):
- Ideas, wishes and features someone lists or presents ("it should have a mute button", "maybe a flip top",
  "what if it glowed") unless the group then agrees to them.
- Questions, unless the answer settles the matter; then cite the answer too.
- Findings or facts read out ("research shows people want...", "the trend is fruit colours"): these inform a
  decision but are not one.
- Meeting logistics: when the next meeting is, going back to rooms, tool training, passing the pen,
  presenting slides now, questionnaires, jokes and small talk.
- Restating something already listed: merge repeats into one item.
Fields:
- Owner is the person who will DO the work, not the person who assigns it:
  "I'll do X" -> the speaker. "You'll do X" -> the person being addressed, if the transcript shows who that is.
  "We'll do X" -> null. If you cannot tell who is meant, set owner to null. Decisions have owner null.
- Deadline: copy a time phrase only if it appears in the utterances you cite. Most items have no deadline: null.
- quote must be copied exactly from a cited utterance, 3 to 12 words.
- confidence "high" only if the quote alone shows a clear commitment or agreement; otherwise "low".
- Cite at most 4 utterances per item. If there are no items, return {"items": []}."""


# v2 stays the default: on the dev split v3 cut items per meeting by half but found fewer reference items
# (coverage 76.5% -> 64.7%), see results/RESULTS.md. Set ECHOAI_PROMPT=v3 to use the stricter prompt.
PROMPT_VERSION = os.environ.get("ECHOAI_PROMPT", "v2")
SYSTEM = SYSTEM_V3 if PROMPT_VERSION == "v3" else SYSTEM_V2


@dataclass
class Item:
    type: str
    description: str
    owner: str | None
    deadline: str | None
    evidence: list
    confidence: str
    problems: list = field(default_factory=list)
    quote: str | None = None
    score: int = 0                      # 0-100, set by review_filter.assess
    reasons: list = field(default_factory=list)
    hidden: bool = False                # below the show threshold or a repeat; kept for the reviewer


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


def normalize_owner(owner):
    """The model sometimes writes "null" as text, or a role's full name instead of its label."""
    if owner is None or str(owner).strip().lower() in ("", "null", "none", "n/a"):
        return None
    full_names = {v.lower(): k for k, v in ROLES.items()}
    parts = [full_names.get(p.strip().lower(), p.strip()) for p in str(owner).split(",")]
    return ", ".join(parts)


def _norm(text):
    return " ".join(re.findall(r"[a-z0-9']+", str(text).lower()))


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
                    owner=normalize_owner(it.get("owner")), deadline=it.get("deadline") or None,
                    evidence=[str(e) for e in (it.get("evidence") or [])],
                    confidence=str(it.get("confidence", "low")).lower(),
                    quote=(str(it["quote"]).strip() or None) if it.get("quote") else None)
        if item.type not in ("action", "decision"):
            item.problems.append("bad_type")
        if not item.description:
            item.problems.append("empty_description")
        if not item.evidence:
            item.problems.append("no_evidence")
        if any(e not in shown for e in item.evidence):
            item.problems.append("evidence_not_in_input")  # cites an utterance it was never shown
        if item.owner and any(o.strip() not in speakers for o in str(item.owner).split(",")):
            item.problems.append("owner_not_in_meeting")
        if item.deadline:
            cited = " ".join(utts[int(e[1:])]["text"] for e in item.evidence if e in shown).lower()
            if not any(w in cited for w in re.findall(r"[a-z0-9]+", item.deadline.lower()) if len(w) > 2):
                item.problems.append("deadline_not_in_evidence")
        if item.quote:
            cited = _norm(" ".join(utts[int(e[1:])]["text"] for e in item.evidence if e in shown))
            q = _norm(item.quote).split()
            # Exact match, or nearly: transcripts are messy and the model may drop a filler word.
            if q and _norm(item.quote) not in cited and sum(w in cited.split() for w in q) / len(q) < 0.8:
                item.problems.append("quote_not_in_evidence")  # the model paraphrased or invented the quote
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


# JSON schema handed to Ollama so decoding can only produce output of the required shape.
ITEMS_SCHEMA = {
    "type": "object", "required": ["items"],
    "properties": {"items": {"type": "array", "items": {
        "type": "object", "required": ["type", "description", "owner", "deadline", "evidence", "confidence"],
        "properties": {
            "type": {"type": "string", "enum": ["action", "decision"]},
            "description": {"type": "string"},
            "owner": {"type": ["string", "null"]},
            "deadline": {"type": ["string", "null"]},
            "evidence": {"type": "array", "items": {"type": "string"}},
            "quote": {"type": "string"},
            "confidence": {"type": "string", "enum": ["high", "low"]}}}}}}
if PROMPT_VERSION != "v3":
    ITEMS_SCHEMA["properties"]["items"]["items"]["properties"].pop("quote")
else:
    ITEMS_SCHEMA["properties"]["items"]["items"]["required"].insert(5, "quote")


class OllamaExtractor(Extractor):
    """Same model and prompt served by Ollama (llama.cpp), which generates several times faster than
    plain transformers on a laptop GPU. Still fully local: Ollama listens on localhost only."""

    def __init__(self, model=None, host=None):
        import os
        self.model_name = model or os.environ.get("ECHOAI_OLLAMA_MODEL", "qwen3:4b-instruct-2507-q8_0")
        self.host = (host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        if not self.host.startswith("http"):
            self.host = "http://" + self.host

    def generate(self, user_msg, max_new_tokens=1536):
        import urllib.request
        body = {"model": self.model_name, "stream": False, "format": ITEMS_SCHEMA,
                "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_msg}],
                "options": {"temperature": 0, "num_predict": max_new_tokens, "num_ctx": 16384}}
        req = urllib.request.Request(f"{self.host}/api/chat", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=900) as r:
            out = json.loads(r.read())
        return out["message"]["content"], int(out.get("prompt_eval_count", 0))


def make_extractor():
    """ECHOAI_BACKEND=ollama (default) or transformers."""
    import os
    return OllamaExtractor() if os.environ.get("ECHOAI_BACKEND", "ollama") == "ollama" else Extractor()
