"""EchoAI demo: the Meetings, Summary and My Tasks screens from Lab 1, running fully on this laptop.

Run:  streamlit run app/app.py
"""
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

st.set_page_config(page_title="EchoAI", page_icon="🎙️", layout="wide")
LOG = ROOT / "results" / "confirmation_log.csv"


@st.cache_resource(show_spinner="Loading models onto the GPU...")
def models():
    return Layer1(), make_extractor()


def fmt_time(secs):
    if secs is None or pd.isna(secs):
        return "--:--"
    secs = int(secs)
    return f"{secs // 60:02d}:{secs % 60:02d}"


def ami_meetings():
    das = ROOT / "data" / "processed" / "dialogue_acts.csv"
    if not das.exists():
        return {}
    df = pd.read_csv(das)
    df = df[df.split == "test"].sort_values(["meeting", "start"])
    return {m: [{"speaker": r.role if isinstance(r.role, str) else r.speaker, "text": r.text, "start": r.start}
                for r in g.itertuples()] for m, g in df.groupby("meeting")}


st.sidebar.title("🎙️ EchoAI")
st.sidebar.caption("Meeting summarizer · all processing runs locally on this laptop's GPU; nothing is sent online.")
screen = st.sidebar.radio("Screen", ["Meetings", "Summary", "My Tasks"])

if "meetings" not in st.session_state:
    st.session_state.meetings = {}

# ---------------- Meetings ----------------
if screen == "Meetings":
    st.header("Meetings")
    src = st.radio("Add a meeting", ["Paste a transcript", "Upload audio", "AMI test meeting (unseen by the model)"],
                   horizontal=True)
    name, utts, audio_bytes = None, None, None
    if src == "Paste a transcript":
        name = st.text_input("Meeting name", "Team stand-up")
        text = st.text_area("One line per turn: `Speaker: what they said` (optional `[mm:ss]` at the start)", height=220)
        if text.strip():
            utts = parse_transcript(text)
    elif src == "Upload audio":
        st.info("Consent check (Lab 7): only upload a recording if every participant agreed to it.")
        consent = st.checkbox("All participants consented to this recording being processed")
        up = st.file_uploader("Audio file", type=["wav", "mp3", "m4a"])
        name = st.text_input("Meeting name", up.name if up else "Recorded meeting")
        if up and consent:
            audio_bytes = up.getvalue()
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
            with st.spinner("Transcribing with Whisper..."), tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(audio_bytes)
                utts = transcribe(f.name)
        t0 = datetime.now()
        with st.spinner(f"Layer 1 screening {len(utts)} utterances, then Layer 2 extracting items..."):
            scores, flagged, items = run(utts, layer1, extractor)
        st.session_state.meetings[name] = {
            "utts": utts, "scores": scores, "flagged": flagged, "audio": audio_bytes,
            "items": [{**vars(it), "status": "pending"} for it in items],
            "processed": datetime.now().strftime("%Y-%m-%d %H:%M"), "seconds": (datetime.now() - t0).total_seconds()}
        st.success(f"Done in {st.session_state.meetings[name]['seconds']:.0f}s: {len(flagged)} of {len(utts)} "
                   f"utterances flagged by Layer 1, {len(items)} items drafted. Open the Summary screen to review.")

    if st.session_state.meetings:
        st.subheader("Processed meetings")
        st.dataframe(pd.DataFrame([{
            "Meeting": n, "Processed": m["processed"], "Utterances": len(m["utts"]),
            "Decisions": sum(i["type"] == "decision" for i in m["items"]),
            "Actions": sum(i["type"] == "action" for i in m["items"]),
            "Awaiting review": sum(i["status"] == "pending" for i in m["items"])}
            for n, m in st.session_state.meetings.items()]), hide_index=True, use_container_width=True)

# ---------------- Summary ----------------
elif screen == "Summary":
    if not st.session_state.meetings:
        st.info("Process a meeting on the Meetings screen first.")
        st.stop()
    name = st.selectbox("Meeting", list(st.session_state.meetings))
    m = st.session_state.meetings[name]
    st.header(name)
    st.caption("⚠️ AI-generated draft. Nothing is shared until you confirm each item (Lab 7 human oversight).")
    if m["audio"]:
        st.audio(m["audio"])
    speakers = sorted({u["speaker"] for u in m["utts"]}) + ["(unassigned)"]

    for kind, title in [("decision", "Decisions"), ("action", "Action items")]:
        st.subheader(title)
        rows = [(i, it) for i, it in enumerate(m["items"]) if it["type"] == kind]
        if not rows:
            st.write("None found.")
        for i, it in rows:
            flag = "🟢" if it["confidence"] == "high" else "🟡 check this"
            with st.container(border=True):
                c1, c2 = st.columns([3, 2])
                c1.markdown(f"**{it['description']}**  \n{flag}"
                            + (f"  \n⚠️ auto-check: {', '.join(it['problems'])}" if it["problems"] else ""))
                if kind == "action":
                    cur = it["owner"] if it["owner"] in speakers else "(unassigned)"
                    it["owner"] = c2.selectbox("Owner", speakers, index=speakers.index(cur), key=f"o{name}{i}")
                    it["deadline"] = c2.text_input("Deadline", it["deadline"] or "", key=f"d{name}{i}") or None
                with c1.expander("Source in the meeting"):
                    for e in it["evidence"]:
                        if e[1:].isdigit() and int(e[1:]) < len(m["utts"]):
                            u = m["utts"][int(e[1:])]
                            st.markdown(f"`{fmt_time(u['start'])}` **{u['speaker']}:** {u['text']}")
                b1, b2, _ = c2.columns([1, 1, 2])
                if b1.button("Confirm", key=f"c{name}{i}", disabled=it["status"] == "confirmed"):
                    it["status"] = "confirmed"
                if b2.button("Reject", key=f"r{name}{i}", disabled=it["status"] == "rejected"):
                    it["status"] = "rejected"
                c2.caption(f"Status: {it['status']}")

    pending = [it for it in m["items"] if it["status"] == "pending"]
    confirmed = [it for it in m["items"] if it["status"] == "confirmed"]
    st.divider()
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
        st.download_button("Download summary", draft, file_name=f"{name}_summary.txt")
        if st.button("Log review results"):
            log = pd.DataFrame([{"meeting": name, "logged": datetime.now().isoformat(timespec="seconds"),
                                 "type": it["type"], "status": it["status"], "confidence": it["confidence"]}
                                for it in m["items"]])
            log.to_csv(LOG, mode="a", header=not LOG.exists(), index=False)
            st.success("Logged. Rejection rate feeds the Lab 9 metric 'items rejected at confirmation'.")

# ---------------- My Tasks ----------------
else:
    st.header("My Tasks")
    owners = sorted({it["owner"] for m in st.session_state.meetings.values() for it in m["items"]
                     if it["type"] == "action" and it["owner"]})
    if not owners:
        st.info("No assigned action items yet.")
        st.stop()
    me = st.selectbox("Show tasks for", owners)
    rows = [{"Task": it["description"], "Deadline": it["deadline"] or "", "Meeting": n, "Status": it["status"],
             "Said at": fmt_time(m["utts"][int(it["evidence"][0][1:])]["start"]) if it["evidence"] and it["evidence"][0][1:].isdigit() else ""}
            for n, m in st.session_state.meetings.items() for it in m["items"]
            if it["type"] == "action" and it["owner"] == me and it["status"] != "rejected"]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
