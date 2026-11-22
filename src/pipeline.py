"""End-to-end EchoAI pipeline used by the demo app: audio or transcript -> Layer 1 -> Layer 2.

Speech-to-text uses Whisper locally. Whisper does not identify speakers, so uploaded audio is
labelled as a single speaker and owners are confirmed by the meeting owner in the app; this is
the manual-owner fallback planned in Lab 7, section 1.
"""
import json
import re
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
LINE = re.compile(r"^\s*(?:\[?(\d{1,2}:\d{2}(?::\d{2})?)\]?\s*)?([^:]{1,40}):\s*(.+)$")
CONTEXT_TURNS = 2  # must match train_layer1.py


def parse_transcript(text: str):
    """Lines like 'Malyka: I'll send the report by Friday' or '[03:12] PM: ...'."""
    utts = []
    for line in text.splitlines():
        m = LINE.match(line)
        if m:
            ts, speaker, said = m.groups()
            secs = None
            if ts:
                parts = [int(p) for p in ts.split(":")]
                secs = sum(p * 60 ** i for i, p in enumerate(reversed(parts)))
            utts.append({"speaker": speaker.strip(), "text": said.strip(), "start": secs})
        elif line.strip() and utts:
            utts[-1]["text"] += " " + line.strip()
    return utts


def transcribe(audio_path: str, model_id="openai/whisper-large-v3-turbo"):
    """large-v3-turbo: lower word error rate than whisper-small on AMI audio and twice as fast (results/asr_metrics.json).
    WAV is read directly; MP3/M4A need ffmpeg (installed in the Docker image)."""
    from transformers import pipeline
    asr = pipeline("automatic-speech-recognition", model=model_id, dtype=torch.float16, device="cuda")
    audio = audio_path
    if str(audio_path).lower().endswith(".wav"):
        from eval_asr import load_wav
        audio = load_wav(audio_path)
    out = asr(audio, return_timestamps=True, chunk_length_s=30, generate_kwargs={"language": "en"})
    del asr
    torch.cuda.empty_cache()
    return [{"speaker": "Speaker", "text": c["text"].strip(), "start": c["timestamp"][0]} for c in out["chunks"]
            if c["text"].strip()]


class Layer1:
    def __init__(self, path=ROOT / "models" / "layer1"):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path).to("cuda").eval()
        metrics = json.loads((ROOT / "results" / "layer1_metrics.json").read_text())
        self.threshold = next(m["threshold"] for m in metrics if m["model"] == "roberta")

    def score(self, utts):
        texts = [u["text"] for u in utts]
        ctx = [" ".join(texts[max(0, i - CONTEXT_TURNS):i]) for i in range(len(texts))]
        scores = []
        with torch.no_grad():
            for i in range(0, len(texts), 128):
                enc = self.tok(texts[i:i + 128], ctx[i:i + 128], truncation=True, max_length=128, padding=True,
                               return_tensors="pt").to("cuda")
                scores += torch.softmax(self.model(**enc).logits.float(), -1)[:, 1].tolist()
        return scores


def run(utts, layer1: Layer1, extractor, chunk=60):
    scores = layer1.score(utts)
    flagged = [i for i, s in enumerate(scores) if s >= layer1.threshold]
    items = []
    for c in range(0, len(flagged), chunk):
        items += extractor.extract(utts, flagged[c:c + chunk])["items"]
    return scores, flagged, items
