"""Precision filter: score every drafted item and hide the ones that are probably noise.

Layer 1 is tuned for recall (it passes ~19% of utterances so it misses as few items as possible), so Layer 2
sees a lot of brainstorming and small talk and drafts ~25 items per meeting against ~6 in AMI's human summary.
This step runs after Layer 2 and needs no GPU. Each item gets:

  score    0-100, how likely it is a real decision or action item (shown in the app as "confidence")
  reasons  short human-readable notes on what raised or lowered the score
  hidden   True when the score is below SHOW_AT, or the item repeats a stronger one

Hidden items are never deleted: the app keeps them behind a "show hidden items" switch so the meeting owner
can still rescue one (Lab 7 human oversight). The rules use only the words of the cited utterances, so every
decision the filter makes can be explained to the user.
"""
import re

SHOW_AT = 50  # items scoring below this are hidden by default

# What the cited utterances say -------------------------------------------------------------------------
DECIDED = re.compile(r"\b(decided|decision|agreed|agree on|settled|we'?ll go (with|for)|go(ing)? with|let'?s go|"
                     r"we chose|we'?ve chosen|final(ly|ise|ize)?|that'?s (it|settled|decided))\b")
COMMIT = re.compile(r"\b(i'?ll|i will|i'?m (going to|gonna)|i am going to|we'?ll|we will|we'?re (going to|gonna)|"
                    r"we are going to|you'?ll|you will|you'?re (going to|gonna)|let'?s|it'?ll be|it'?s (going to|"
                    r"gonna) be|(is|are) (going to|gonna)|will (do|be|have|send|make|check|look|work|put|bring)|"
                    r"have to|need to|must|by (monday|tuesday|wednesday|thursday|friday|tomorrow|next week))\b")
HEDGE = re.compile(r"\b(maybe|perhaps|might|possibly|what if|wondering|i (don'?t|do not) know|dunno|how about|"
                   r"is it possible|would you|should we|shouldn'?t we|can we|could we|do we|i guess|"
                   r"we could|you could|could have|suggest)\b")
AGREE = re.compile(r"^\W*(yeah|yes|yep|okay|ok|sure|agreed|fine|definitely|absolutely|exactly|right|good idea|"
                   r"sounds good|that'?s (good|fine|great|a good idea)|i think so|mm-hmm)\b")
REPORT = re.compile(r"\b(found|findings|survey|research (shows|says)|people (want|prefer|said)|trend(s|y)? "
                    r"(is|are|show)|consumers? (want|consider|prefer)|according to)\b")
# The meeting talking about itself: when it ends, tooling, rooms. Real follow-ups ("I'll send the minutes") pass.
LOGISTICS = re.compile(r"(tool training|next meeting (is|will be|in)|meeting (will be|is) in|back to (our|the|your) "
                       r"room|pass the pen|line width|whiteboard|white board|smart.?board|celebrat|questionnaire|"
                       r"export as picture|minutes (left|to go)|five minutes|ten minutes|thirty minutes|"
                       r"take a break|lunch)", re.I)

# Reasons that raised the score; every other reason lowered it or is a warning (used by the app).
POSITIVE = {"strong action/decision language", "explicit decision words", "commitment words", "someone else agreed"}
HARD_PROBLEMS = {"bad_type", "empty_description", "no_evidence", "evidence_not_in_input", "quote_not_in_evidence"}
STOP = set("a an the to of for and or in on with by is be are it its this that as at from into our we will "
           "remote control should use include design".split())


def _cited(item, utts):
    idx = [int(e[1:]) for e in item.evidence if re.fullmatch(r"U\d+", str(e)) and int(e[1:]) < len(utts)]
    return idx


def _words(text):
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP and len(w) > 2}


def score_item(item, utts, scores=None):
    """Returns (score, reasons). `scores` are Layer 1 probabilities per utterance, if available.

    The base is how strongly Layer 1 rated the cited lines (mean of the two best). On the AMI test meetings
    this is the best single predictor of whether an item is one the human summary also lists. Language rules
    then add or subtract points for things the classifier cannot see: who agreed, whether it was only a
    question or an idea, meeting logistics, and failed grounding checks.
    """
    idx = _cited(item, utts)
    lines = [utts[i] for i in idx]
    texts = [str(u["text"]).lower() for u in lines]
    joined = " ".join(texts)
    proposer = lines[0]["speaker"] if lines else None
    why = []

    if scores is not None and idx:
        best = sorted((scores[i] for i in idx), reverse=True)[:2]
        strength = sum(best) / len(best)
    else:
        strength = 0.5  # no Layer 1 scores available: neutral
    s = 20 + 60 * min(1.0, strength / 0.8)
    if strength >= 0.6:
        why.append("strong action/decision language")
    elif strength < 0.25:
        why.append("little action/decision language")

    decided, committed = bool(DECIDED.search(joined)), bool(COMMIT.search(joined))
    if decided:
        s += 10; why.append("explicit decision words")
    elif committed:
        s += 10; why.append("commitment words")
    others_agree = any(AGREE.search(t) for u, t in zip(lines, texts) if u["speaker"] != proposer)
    if others_agree:
        s += 10; why.append("someone else agreed")
    if not (decided or committed):
        if HEDGE.search(joined) and not others_agree:
            s -= 10; why.append("only a suggestion")
        if texts and all(t.rstrip().endswith("?") for u, t in zip(lines, texts) if u["speaker"] == proposer):
            s -= 15; why.append("only a question")
        if REPORT.search(joined):
            s -= 10; why.append("reports findings, not a choice")
    if LOGISTICS.search(item.description) or (LOGISTICS.search(joined) and not decided):
        s -= 25; why.append("meeting logistics")
    if len(idx) > 6:
        s -= 10; why.append("evidence spread over many lines")
    if item.confidence != "high":
        why.append("model unsure")  # shown to the reviewer, but not scored: on AMI it did not predict errors
    hard = HARD_PROBLEMS & set(item.problems)
    soft = set(item.problems) - hard
    s -= 10 * len(soft)
    if hard:
        s = min(s, 30)  # an item that cites nothing real, or misquotes the meeting, is never shown by default
    why += [f"check failed: {p.replace('_', ' ')}" for p in item.problems]
    return int(round(max(0, min(100, s)))), why


def assess(items, utts, scores=None, show_at=SHOW_AT):
    """Scores every item, then hides low scorers and near-duplicates. Returns items sorted as in the meeting."""
    for it in items:
        it.score, it.reasons = score_item(it, utts, scores)
        it.hidden = it.score < show_at
        if it.type == "decision":
            it.owner = None  # a decision belongs to the group; owners only make sense for actions
    kept = []
    for it in sorted(items, key=lambda x: -x.score):
        if it.hidden:
            continue
        w, ev = _words(it.description), set(it.evidence)
        for k in kept:
            kw, kev = _words(k.description), set(k.evidence)
            jacc = len(w & kw) / max(1, len(w | kw))
            ev_overlap = len(ev & kev) / max(1, min(len(ev), len(kev)))
            if jacc >= 0.5 or (ev_overlap >= 0.5 and jacc >= 0.2):
                it.hidden = True
                it.reasons.append(f"repeats: {k.description}")
                break
        else:
            kept.append(it)
    first = lambda it: min(_cited(it, utts) or [10 ** 9])
    return sorted(items, key=first)
