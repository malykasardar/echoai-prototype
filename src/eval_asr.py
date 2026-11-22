"""Speech-to-text accuracy: Whisper word error rate (WER) on AMI test-meeting audio (Lab 4 target: under 5%).

Reference = AMI's manual word-level transcript for all speakers, in time order.
Audio = AMI "Mix-Headset" recording (all headset microphones mixed), downloaded to data/audio/.
Reported twice: on every word, and with filler words (um, uh, mm-hmm...) removed from both sides,
since Whisper usually drops fillers that AMI transcribes and those are not errors a user would care about.

Usage:  python src/eval_asr.py ES2004a [--models openai/whisper-small openai/whisper-large-v3-turbo]
"""
import argparse
import json
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import jiwer
import torch
from transformers import pipeline

ROOT = Path(__file__).resolve().parents[1]
AMI = ROOT / "data" / "raw" / "ami"
FILLERS = {"um", "uh", "mm", "hmm", "mm-hmm", "uh-huh", "huh", "ah", "oh", "er", "erm", "eh", "hm"}


def reference(meeting):
    words = []
    for f in sorted((AMI / "words").glob(f"{meeting}.*.words.xml")):
        for w in ET.parse(f).getroot().iter("w"):
            if w.get("punc") != "true" and w.text and w.get("starttime"):
                words.append((float(w.get("starttime")), w.text))
    return " ".join(t for _, t in sorted(words))


def load_wav(path):
    """Read a WAV file without ffmpeg (AMI audio is 16 kHz PCM)."""
    from scipy.io import wavfile
    sr, data = wavfile.read(path)
    data = data.astype("float32") / (32768.0 if data.dtype.kind == "i" else 1.0)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return {"raw": data, "sampling_rate": sr}


def normalize(text, drop_fillers):
    text = text.lower().replace("_", "")  # AMI writes acronyms as T_V_
    text = re.sub(r"[^a-z0-9' -]", " ", text)
    toks = [t.strip("'-") for t in text.split()]
    return " ".join(t for t in toks if t and not (drop_fillers and t in FILLERS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("meeting")
    ap.add_argument("--models", nargs="+", default=["openai/whisper-small", "openai/whisper-large-v3-turbo"])
    args = ap.parse_args()
    audio = ROOT / "data" / "audio" / f"{args.meeting}.Mix-Headset.wav"
    ref = reference(args.meeting)
    results = []
    for model_id in args.models:
        asr = pipeline("automatic-speech-recognition", model=model_id, dtype=torch.float16, device="cuda")
        t0 = time.perf_counter()
        hyp = asr(load_wav(audio), chunk_length_s=30, batch_size=8, generate_kwargs={"language": "en"})["text"]
        secs = time.perf_counter() - t0
        del asr
        torch.cuda.empty_cache()
        res = {"meeting": args.meeting, "model": model_id, "seconds": round(secs, 1),
               "wer_all_words": round(jiwer.wer(normalize(ref, False), normalize(hyp, False)), 4),
               "wer_without_fillers": round(jiwer.wer(normalize(ref, True), normalize(hyp, True)), 4)}
        print(res, flush=True)
        results.append(res)
        (ROOT / "results" / f"asr_{args.meeting}_{model_id.split('/')[-1]}.txt").write_text(hyp, encoding="utf-8")
    out = ROOT / "results" / "asr_metrics.json"
    prev = json.loads(out.read_text()) if out.exists() else []
    prev = [r for r in prev if r["meeting"] != args.meeting] + results
    out.write_text(json.dumps(prev, indent=2))


if __name__ == "__main__":
    main()
