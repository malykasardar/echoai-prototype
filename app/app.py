"""EchoAI demo: the Meetings, Summary and My Tasks screens from Lab 1, running fully on this laptop.

Run:  streamlit run app/app.py
"""
import html
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from layer2 import make_extractor  # noqa: E402
from pipeline import Layer1, parse_transcript, run, transcribe  # noqa: E402
from review_filter import POSITIVE, SHOW_AT  # noqa: E402

st.set_page_config(page_title="EchoAI", page_icon="🎙️", layout="wide")
LOG = ROOT / "results" / "confirmation_log.csv"
HIGH_AT = 70  # confidence at or above this is "high"

# Chips and cards. Colours are translucent so they read on both the light and the dark Streamlit theme.
st.markdown("""<style>
.chip{display:inline-block;padding:1px 9px;margin:0 6px 4px 0;border-radius:999px;font-size:.8rem;font-weight:600;
      line-height:1.6;border:1px solid transparent;white-space:nowrap}
.c-dec{background:rgba(99,102,241,.14);color:#6366f1;border-color:rgba(99,102,241,.35)}
.c-act{background:rgba(16,185,129,.14);color:#0f9f6e;border-color:rgba(16,185,129,.35)}
.c-hi{background:rgba(16,185,129,.12);color:#0f9f6e}
.c-mid{background:rgba(245,158,11,.14);color:#c27c06}
.c-lo{background:rgba(148,163,184,.18);color:#7b8794}
.c-meta{background:rgba(148,163,184,.14);color:inherit;font-weight:500}
.c-warn{background:rgba(239,68,68,.12);color:#dc4a4a}
.c-ok{background:rgba(16,185,129,.16);color:#0f9f6e}
.c-no{background:rgba(239,68,68,.12);color:#dc4a4a}
.item-title{font-size:1.05rem;font-weight:650;margin:2px 0 6px 0}
.quote{border-left:3px solid rgba(148,163,184,.6);padding:2px 0 2px 10px;margin:6px 0 2px 0;opacity:.9;
       font-size:.92rem}
.quote .who{font-weight:600}
.quote .ts{font-family:monospace;opacity:.7;margin-right:6px}
.ctx{font-size:.88rem;margin:2px 0;opacity:.6}
.ctx.cited{opacity:1}
.why{font-size:.85rem;margin:1px 0}
</style>""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading models onto the GPU...")
def models():
    return Layer1(), make_extractor()


def fmt_time(secs):
    if secs is None or pd.isna(secs):
        return "--:--"
    secs = int(secs)
    return f"{secs // 60:02d}:{secs % 60:02d}"


def chip(text, cls):
    return f'<span class="chip {cls}">{html.escape(str(text))}</span>'


def conf_chip(score):
    if score >= HIGH_AT:
        return chip(f"High confidence · {score}", "c-hi")
    if score >= SHOW_AT:
        return chip(f"Medium confidence · {score}", "c-mid")
    return chip(f"Low confidence · {score}", "c-lo")


def cited_idx(it, utts):
    return [int(e[1:]) for e in it["evidence"] if str(e)[1:].isdigit() and int(e[1:]) < len(utts)]


def key_line(it, utts):
    """The cited utterance that best shows the item: the one holding the model's quote, else the first."""
    idx = cited_idx(it, utts)
    if not idx:
        return None
    q = (it.get("quote") or "").lower()
    for i in idx:
        if q and q[:30] in str(utts[i]["text"]).lower():
            return i
    return idx[0]


def ami_meetings():
    das = ROOT / "data" / "processed" / "dialogue_acts.csv"
    if not das.exists():
        return {}
    df = pd.read_csv(das)
    df = df[df.split == "test"].sort_values(["meeting", "start"])
    return {m: [{"speaker": r.role if isinstance(r.role, str) else r.speaker, "text": r.text, "start": r.start}
                for r in g.itertuples()] for m, g in df.groupby("meeting")}


def visible(m):
    return [it for it in m["items"] if not it.get("hidden")]


st.sidebar.title("🎙️ EchoAI")
st.sidebar.caption("Turns meetings into decisions and action items. Everything runs on this laptop's GPU; "
                   "nothing is sent online.")
screen = st.sidebar.radio("Screen", ["Meetings", "Summary", "My Tasks"],
                          captions=["Add and process a meeting", "Review what EchoAI found", "Your open action items"])

if "meetings" not in st.session_state:
    st.session_state.meetings = {}

# ---------------- Meetings ----------------
if screen == "Meetings":
    st.header("Meetings")
    src = st.radio("Add a meeting", ["Paste a transcript", "Upload audio", "AMI test meeting (unseen by the model)"],
                   horizontal=True)
    name, utts, audio_bytes, audio_ext = None, None, None, ".wav"
    if src == "Paste a transcript":
        name = st.text_input("Meeting name", "Team stand-up")
        text = st.text_area("One line per turn: `Speaker: what they said` (optional `[mm:ss]` at the start)", height=220,
                            placeholder="[00:12] Malyka: I'll send the report by Friday\n[00:15] Ethan: Sounds good.")
        if text.strip():
            utts = parse_transcript(text)
            st.caption(f"{len(utts)} turns from {len({u['speaker'] for u in utts})} speakers")
    elif src == "Upload audio":
        st.info("Consent check (Lab 7): only upload a recording if every participant agreed to it.")
        consent = st.checkbox("All participants consented to this recording being processed")
        up = st.file_uploader("Audio file", type=["wav", "mp3", "m4a"])
        name = st.text_input("Meeting name", up.name if up else "Recorded meeting")
        if up and consent:
            audio_bytes, audio_ext = up.getvalue(), Path(up.name).suffix.lower()
    else:
        ami = ami_meetings()
        if ami:
            name = st.selectbox("Meeting", list(ami))
            utts = ami[name]
        else:
            st.warning("Run src/prepare_ami.py first.")

    if st.button("Process meeting", type="primary", disabled=not (utts or audio_bytes)):
        layer1, extractor = models()
        if audio_bytes:
            with st.spinner("Transcribing with Whisper..."), tempfile.NamedTemporaryFile(suffix=audio_ext, delete=False) as f:
                f.write(audio_bytes)
                utts = transcribe(f.name)
        t0 = datetime.now()
        with st.status(f"Processing {len(utts)} utterances...", expanded=False) as status:
            status.write("Layer 1 is screening every utterance, Layer 2 is drafting items, then each item is scored.")
            scores, flagged, items = run(utts, layer1, extractor)
            status.update(label="Done", state="complete")
        st.session_state.meetings[name] = {
            "utts": utts, "scores": scores, "flagged": flagged, "audio": audio_bytes,
            "items": [{**vars(it), "status": "pending"} for it in items],
            "processed": datetime.now().strftime("%Y-%m-%d %H:%M"), "seconds": (datetime.now() - t0).total_seconds()}
        m = st.session_state.meetings[name]
        shown = visible(m)
        st.success(f"Done in {m['seconds']:.0f}s. Open the **Summary** screen to review.")
        c = st.columns(4)
        c[0].metric("Utterances", len(utts))
        c[1].metric("Flagged by Layer 1", len(flagged))
        c[2].metric("Items to review", len(shown))
        c[3].metric("Hidden as likely noise", len(items) - len(shown))

    if st.session_state.meetings:
        st.subheader("Processed meetings")
        st.dataframe(pd.DataFrame([{
            "Meeting": n, "Processed": m["processed"], "Utterances": len(m["utts"]),
            "Decisions": sum(i["type"] == "decision" for i in visible(m)),
            "Actions": sum(i["type"] == "action" for i in visible(m)),
            "Hidden as noise": sum(bool(i.get("hidden")) for i in m["items"]),
            "Awaiting review": sum(i["status"] == "pending" for i in visible(m))}
            for n, m in st.session_state.meetings.items()]), hide_index=True, width="stretch")

# ---------------- Summary ----------------
elif screen == "Summary":
    if not st.session_state.meetings:
        st.info("Process a meeting on the Meetings screen first.")
        st.stop()
    name = st.selectbox("Meeting", list(st.session_state.meetings))
    m = st.session_state.meetings[name]
    utts = m["utts"]
    shown = visible(m)
    n_hidden = len(m["items"]) - len(shown)
    st.header(name)
    st.caption(f"Processed {m['processed']} in {m['seconds']:.0f}s · ⚠️ AI-generated draft: nothing is shared until "
               "you confirm each item (Lab 7 human oversight).")
    if m["audio"]:
        st.audio(m["audio"])

    reviewed = sum(it["status"] != "pending" for it in shown)
    c = st.columns(4)
    c[0].metric("Decisions", sum(it["type"] == "decision" for it in shown))
    c[1].metric("Action items", sum(it["type"] == "action" for it in shown))
    c[2].metric("Reviewed", f"{reviewed} / {len(shown)}")
    c[3].metric("Hidden as likely noise", n_hidden, help=f"Items scoring below {SHOW_AT}/100, or repeating a stronger "
                "item. Turn on 'Show hidden items' to check them.")
    if shown:
        st.progress(reviewed / len(shown))

    f1, f2, f3 = st.columns([2, 2, 2], vertical_alignment="center")
    kind_filter = f1.segmented_control("Show", ["All", "Decisions", "Action items"], default="All",
                                       key=f"kind{name}") or "All"
    show_hidden = f2.toggle(f"Show hidden items ({n_hidden})", key=f"hid{name}")
    strong = [it for it in shown if it["score"] >= HIGH_AT and it["status"] == "pending"]
    if f3.button(f"Confirm all high-confidence ({len(strong)})", disabled=not strong, width="stretch"):
        for it in strong:
            it["status"] = "confirmed"
        st.rerun()

    speakers = sorted({u["speaker"] for u in utts}) + ["(unassigned)"]
    sections = [("decision", "Decisions"), ("action", "Action items")]
    if kind_filter != "All":
        sections = [s for s in sections if s[1] == kind_filter]
    for kind, title in sections:
        rows = [(i, it) for i, it in enumerate(m["items"])
                if it["type"] == kind and (show_hidden or not it.get("hidden"))]
        st.subheader(f"{title} ({len(rows)})")
        if not rows:
            st.caption("None found.")
        for i, it in rows:
            with st.container(border=True):
                c1, c2 = st.columns([3, 1.4])
                head = chip("Decision" if kind == "decision" else "Action", "c-dec" if kind == "decision" else "c-act")
                head += conf_chip(it["score"])
                if kind == "action":
                    head += chip(f"👤 {it['owner'] or 'No owner yet'}", "c-meta")
                    head += chip(f"📅 {it['deadline']}", "c-meta") if it["deadline"] else ""
                if it.get("hidden"):
                    head += chip("Hidden", "c-lo")
                if it["problems"]:
                    head += chip("⚠ check failed", "c-warn")
                status_cls = {"confirmed": "c-ok", "rejected": "c-no"}.get(it["status"])
                if status_cls:
                    head += chip(it["status"].capitalize(), status_cls)
                c1.markdown(head + f'<div class="item-title">{html.escape(it["description"])}</div>',
                            unsafe_allow_html=True)
                k = key_line(it, utts)
                if k is not None:
                    u = utts[k]
                    c1.markdown(f'<div class="quote"><span class="ts">{fmt_time(u["start"])}</span>'
                                f'<span class="who">{html.escape(str(u["speaker"]))}:</span> '
                                f'{html.escape(str(u["text"]))}</div>', unsafe_allow_html=True)
                with c1.expander("Evidence and why EchoAI scored it this way"):
                    idx = cited_idx(it, utts)
                    ctx = sorted({j for c in idx for j in (c - 1, c, c + 1) if 0 <= j < len(utts)})
                    for j in ctx:
                        u = utts[j]
                        cls = "ctx cited" if j in idx else "ctx"
                        st.markdown(f'<div class="{cls}"><span class="ts">{fmt_time(u["start"])}</span> '
                                    f'<b>{html.escape(str(u["speaker"]))}:</b> {html.escape(str(u["text"]))}</div>',
                                    unsafe_allow_html=True)
                    if it.get("reasons"):
                        st.markdown("".join(f'<div class="why">{"✅" if r in POSITIVE else "⚠️"} {html.escape(r)}</div>'
                                            for r in it["reasons"]), unsafe_allow_html=True)
                if kind == "action":
                    opts = speakers if not it["owner"] or it["owner"] in speakers else [it["owner"]] + speakers
                    cur = it["owner"] or "(unassigned)"
                    new = c2.selectbox("Owner", opts, index=opts.index(cur), key=f"o{name}{i}")
                    it["owner"] = None if new == "(unassigned)" else new
                    it["deadline"] = c2.text_input("Deadline", it["deadline"] or "", key=f"d{name}{i}",
                                                   placeholder="none stated") or None
                b1, b2 = c2.columns(2)
                if b1.button("✓ Confirm", key=f"c{name}{i}", disabled=it["status"] == "confirmed", width="stretch"):
                    it["status"] = "confirmed"
                    st.rerun()
                if b2.button("✕ Reject", key=f"r{name}{i}", disabled=it["status"] == "rejected", width="stretch"):
                    it["status"] = "rejected"
                    st.rerun()

    pending = [it for it in shown if it["status"] == "pending"]
    confirmed = [it for it in m["items"] if it["status"] == "confirmed"]
    st.divider()
    st.subheader("Share the summary")
    if pending:
        st.warning(f"{len(pending)} item(s) still need review before the summary can be shared.")
    else:
        lines = [f"Summary of {name} (drafted by EchoAI, confirmed by the meeting owner)", "", "Decisions:"]
        lines += [f"- {it['description']}" for it in confirmed if it["type"] == "decision"] or ["- none"]
        lines += ["", "Action items:"]
        lines += [f"- {it['owner'] or 'Unassigned'}: {it['description']}" + (f" (by {it['deadline']})" if it["deadline"] else "")
                  for it in confirmed if it["type"] == "action"] or ["- none"]
        draft = "\n".join(lines)
        st.text_area("Follow-up email / Slack draft", draft, height=220)
        d1, d2, _ = st.columns([1, 1, 3])
        d1.download_button("Download summary", draft, file_name=f"{name}_summary.txt", width="stretch")
        if d2.button("Log review results", width="stretch"):
            log = pd.DataFrame([{"meeting": name, "logged": datetime.now().isoformat(timespec="seconds"),
                                 "type": it["type"], "status": it["status"], "confidence": it["confidence"],
                                 "score": it["score"], "hidden": bool(it.get("hidden"))}
                                for it in m["items"]])
            log.to_csv(LOG, mode="a", header=not LOG.exists(), index=False)
            st.success("Logged. Rejection rate feeds the Lab 9 metric 'items rejected at confirmation'.")

# ---------------- My Tasks ----------------
else:
    st.header("My Tasks")
    actions = [(n, m, it) for n, m in st.session_state.meetings.items() for it in m["items"]
               if it["type"] == "action" and not it.get("hidden") and it["status"] != "rejected"]
    owners = sorted({o.strip() for _, _, it in actions if it["owner"] for o in it["owner"].split(",")})
    if not actions:
        st.info("No action items yet. Process a meeting, then review it on the Summary screen.")
        st.stop()
    me = st.selectbox("Show tasks for", owners + ["(unassigned)"])
    mine = [(n, m, it) for n, m, it in actions
            if (me == "(unassigned)" and not it["owner"])
            or (it["owner"] and me in [o.strip() for o in it["owner"].split(",")])]
    c = st.columns(3)
    c[0].metric("Open tasks", len(mine))
    c[1].metric("Confirmed", sum(it["status"] == "confirmed" for _, _, it in mine))
    c[2].metric("With a deadline", sum(bool(it["deadline"]) for _, _, it in mine))
    rows = []
    for n, m, it in mine:
        k = key_line(it, m["utts"])
        rows.append({"Task": it["description"], "Deadline": it["deadline"] or "", "Meeting": n,
                     "Said at": fmt_time(m["utts"][k]["start"]) if k is not None else "",
                     "Status": {"confirmed": "✅ Confirmed", "pending": "⏳ Awaiting review"}.get(it["status"], it["status"]),
                     "Confidence": it["score"]})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
        "Confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=100, format="%d")})
    st.caption("Only items you have not rejected are listed. Items awaiting review are drafts until confirmed.")
