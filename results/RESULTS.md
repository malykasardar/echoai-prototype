# EchoAI results

All numbers are measured, not planned. Test results come from 12 AMI meetings (series ES2004, IS1009, TS3003)
whose speakers and teams never appear in training. Prompts and thresholds were tuned only on the 13 dev meetings
(ES2011, IS1008, TS3004, IB4003). Hardware: one laptop, NVIDIA RTX 5070 Laptop GPU (12 GB).

## Results against the targets set in Labs 3, 4 and 7

| Target (source) | Result | Met? | Evidence |
|---|---|---|---|
| Rules are not enough; ML is needed (Lab 3 §6) | Keyword rules catch 33% of items; fine-tuned model 94% | Yes | `layer1_metrics.json` |
| Layer 1 catches ≥ 90% of action items and decisions (Lab 4) | **94.4%** (68 of 72) | Yes | `layer1_metrics.json` |
| Layer 1 passes only 10–15% of the transcript to the LLM (Lab 3 §4.4) | 18.9% of utterances, 35.3% of words | Partly | `layer1_metrics.json` |
| Full pipeline captures ≥ 90% of items (Lab 4) | **68.1%** exact line, 77.8% within 3 utterances | No | `layer2_metrics_roberta_ollama.json` |
| Output always valid JSON (Lab 3 §6.1) | 302 items, 1 malformed response recovered | Yes | same |
| Hallucination under 2% (Lab 4) | 1.3% fail automatic grounding checks; manual review pending | Pending | `manual_review_sample.csv` |
| Summary within 30 s (Lab 4) | **41.8 s** average per meeting (15–64 s) | No | same |
| Word error rate under 5% on clear audio (Lab 4) | 26.0% on mixed-microphone meeting audio; 3.6% wrong words, 20.7% dropped words | No | `asr_metrics.json` |
| Error gap between speaker groups ≤ 5 points (Lab 7, R5) | 12-point Layer 1 recall gap, native vs non-native English | Alert fired | `layer1_metrics.json` |

## Layer 1: which utterances might contain an action item or decision

Test set: 8,646 utterances, 160 positive (1.85%). Thresholds chosen on dev to reach 90% item recall.

| Approach | Items caught | Utterance precision | Utterance recall | F1 | PR-AUC | Words sent on |
|---|---|---|---|---|---|---|
| Keyword rules | 33.3% | 0.044 | 0.250 | 0.075 | 0.025 | 21.4% |
| TF-IDF + logistic regression | 88.9% | 0.036 | 0.756 | 0.069 | 0.101 | 49.7% |
| **DistilRoBERTa, fine-tuned** | **94.4%** | 0.069 | 0.706 | 0.126 | 0.142 | 35.3% |

Precision per utterance is low because AMI annotators link each summary item to only a few of the utterances
that discuss it. Layer 1's job is recall: anything it drops never reaches Layer 2.

The first fine-tuning run collapsed to predicting "no" for everything (dev PR-AUC 0.018, equal to the positive
rate). Training on all positives plus a 1:8 sample of negatives fixed it (dev PR-AUC 0.165).

**Fairness (Lab 7, R5).** Recall by group on the test set:

| Group | Positives | Recall |
|---|---|---|
| Female | 85 | 69.4% |
| Male | 25 | 68.0% |
| Native English speaker | 76 | 64.5% |
| Non-native English speaker | 84 | 76.2% |
| Role PM / ID / UI / ME | 78 / 27 / 22 / 33 | 70.5% / 70.4% / 72.7% / 69.7% |

The native/non-native gap (11.7 points) exceeds the 5-point alert level set in Lab 7. Groups are small and whole
teams share a language background, so meeting style may explain part of the gap; this needs a larger test before
any conclusion about bias.

## Layer 2: extraction with a local LLM

Model: Qwen3-4B-Instruct-2507 (Apache 2.0), 8-bit, served by Ollama on the laptop GPU, with JSON-schema-constrained output.

**Prompt tuning on dev only** (13 meetings, 68 reference items):

| Dev metric | Prompt v1 | Prompt v2 |
|---|---|---|
| Items captured | 69.1% | 76.5% |
| Owner correct (action items) | 18% (2/11) | 67% (8/12) |
| Items failing automatic checks | 10.4% | 9.0% |

v1 copied the example deadline in the prompt ("next meeting") into unrelated items, and named the speaker as the
owner even when the speaker was assigning work to someone else. v2 fixed both.

**Final test run** (v2, 12 meetings, 72 reference items):

| Test metric | v1 (transformers) | v2 (Ollama, final) |
|---|---|---|
| Items captured, exact line | 68.1% | 68.1% |
| Items captured, within 3 utterances | 81.9% | 77.8% |
| Owner correct | 3/6 | 4/5 |
| Items failing automatic checks | 2.4% | 1.3% * |
| Seconds per meeting | 210.6 | **41.8** |

\* After fixing a validator bug found on this run: the model sometimes wrote the text "null" or a full role name
("Industrial Designer") as the owner, which the validator counted as an unknown owner (8.3% before the fix).

The dev improvement did not carry over to the test set. With 72 reference items one item is worth 1.4 points,
so differences of a few points are within noise. The test set was not used for any further tuning.

Only 20.5% of extracted items cite a line that AMI's summary also cites. AMI summaries list the main outcomes,
not every commitment, so this is a lower bound on precision; the manual review measures the real figure.

## Precision filter: fewer, better items for the reviewer (Nov 27)

Running the app showed the problem the test numbers hint at: Layer 2 drafted **25.2 items per meeting** while AMI's
human summaries list 6. Most extras were ideas someone floated, findings read off a slide, meeting logistics
("go back to the room") and repeats. Three changes target this:

1. **Precision filter** (`src/review_filter.py`, no GPU). Every item gets a 0–100 confidence score: the base is how
   strongly Layer 1 rated the cited lines, then language rules add points (decision or commitment words,
   someone else agreed) or take them off (only a suggestion, only a question, reported findings, meeting logistics,
   evidence spread over many lines). Failed grounding checks cap the score at 30. Items under 50 and repeats
   of a stronger item are hidden by default; the reviewer can still show them in the app.
2. **Prompt v3** (`src/layer2.py`, opt-in with `ECHOAI_PROMPT=v3`). Names the exclusions above and asks for a short
   verbatim quote, which the validator checks against the cited lines (`quote_not_in_evidence`). Tested on the
   GPU (below) it lost too many real items, so v2 stays the default.
3. **Review screen**: items show their confidence, owner, deadline and the key quote at a glance, with the
   context lines and the scoring reasons one click away, and a one-click confirm for high-confidence items.

**Filter measured on the saved final test run** (prompt v2, 12 meetings; `python src/eval_filter.py`):

| | All drafted (before) | Shown by default (after) | Change |
|---|---|---|---|
| Items per meeting the reviewer reads | 25.2 | **14.0** | −44% |
| Items citing a line AMI's summary also cites | 20.5% | **29.2%** | +8.7 points (+42% relative) |
| AMI reference lines still cited | 75 of 160 | 61 of 160 | −19% |

Why items were hidden (an item can have several reasons): little action/decision language 74, only a suggestion 31,
meeting logistics 16, evidence spread over many lines 9, only a question 5, reported findings 5, repeats 3,
failed grounding checks 3.

The trade-off is explicit: the default view drops some lines AMI annotators linked to a summary item, which is why
hidden items stay one switch away instead of being deleted. With the AMI summaries available, coverage of the 72
reference items drops from 68.1% to 59.7% in the default view (all items remain one switch away). The rules were
written after reading this test run's output, so these numbers are optimistic.

**GPU runs, Nov 27** (RTX 5060 Laptop 8 GB, qwen3:4b-instruct-2507-q8_0 via Ollama; about 13% of the model ran on
the CPU because the 16k context needs 7.3 GB, so times are slower than on the 12 GB laptop):

| | Dev, v2 (Nov 20) | Dev, v3 all | Dev, v3 shown | Test, v2 all | Test, v2 shown (filter) | Test, v3 all | Test, v3 shown |
|---|---|---|---|---|---|---|---|
| Items per meeting | 21.4 | 20.5 | 9.9 | 25.2 | 14.0 | 16.8 | 8.2 |
| Coverage of human items | 76.5% | 64.7% | 52.9% | 68.1% | 59.7% | 55.6% | 43.1% |
| Items citing a reference line | 21.2% | 22.5% | 29.5% | 20.5% | 29.2% | 23.8% | 36.4% |
| Seconds per meeting | n/a | 81.9 | | 41.8 | | 71.9 | |

Decision, made on dev as before: v3 found fewer human items on dev (64.7% vs 76.5%) without a matching gain, so
**v2 + the precision filter is the default**. It shows the reviewer 14 items per meeting instead of 25 and keeps
59.7% coverage in the default view and 68.1% with hidden items included. v3's quote check failed on 9–12% of its
items, mostly paraphrased quotes, which also argues against it for now.

## Does Layer 1 pay for itself? (Lab 3 ROI argument)

Same 4 test meetings (ES2004a–d, 15 reference items), Layer 2 run with and without Layer 1:

| | Without Layer 1 | With Layer 1 |
|---|---|---|
| Tokens read by the LLM | 84,633 | 30,775 (−64%) |
| Layer 2 time | 437.5 s | 162.0 s (−63%) |
| Items captured | 66.7% | 80.0% |

Filtering cuts LLM work by about two thirds and did not reduce quality on this sample.

## Serving speed

| Setup | Tokens/s | ES2004a end to end |
|---|---|---|
| Hugging Face transformers, bf16 | 15 | 170 s |
| Ollama (llama.cpp), 8-bit | 44.5 | 20.5 s |

## Speech to text

AMI ES2004a, 21 minutes, all headset microphones mixed into one track:

| Model | Time | WER, all words | WER, without fillers |
|---|---|---|---|
| whisper-small | 36.7 s | 30.6% | 27.5% |
| whisper-large-v3-turbo | 18.8 s | 29.2% | 26.0% |

Breakdown for large-v3-turbo: 3.6% substitutions (many are spelling variants such as "gonna" / "going to" and
"twenty five" / "25"), 20.7% deletions, 1.7% insertions. The words dropped are mostly overlapping short replies:
"yeah" 43 times, "okay" 26 times. Those replies are often how a group agrees to a decision, so separate
microphone tracks per speaker (as Zoom and Teams provide) are the main improvement for audio input.

## Limitations

- Small evaluation sets: 72 reference items on test, 68 on dev. Owner accuracy rests on 5 test items.
- AMI meetings are role-played design meetings from 2005; a real team's meetings may differ.
- Whisper does not identify speakers, so owners from uploaded audio are confirmed by the user in the app
  (the fallback planned in Lab 7).
- The hallucination rate needs the team's manual review to be confirmed.

## Deployment test (M5 evidence, Nov 22)

`docker compose up` on the RTX 5070 laptop: both containers started, the app answered on localhost:8501,
the Ollama container reused the locally downloaded model, and the app container saw the GPU. The full pipeline
run inside the app container on a 9-line sample stand-up returned the same 5 correct items as the local test
(4 actions with owners and 3 deadlines, 1 decision) in 65.2 s, including first-time model loading.
