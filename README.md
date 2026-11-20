# EchoAI prototype

MAI 104 group project, Seneca Polytechnic. Team: Malyka Sardar, Shivanjali Chandel, Mohmed Sajid Shaikh, Ethan Yitzhaky.

EchoAI turns a meeting transcript (or recording) into decisions and action items with owners and deadlines.
Every item cites the lines it came from, is flagged when uncertain, and must be confirmed by the meeting
owner before the summary can be shared. Everything runs locally on one laptop GPU; no meeting data is sent
to an online service.

## Pipeline

```
audio --Whisper--> transcript --Layer 1--> flagged utterances --Layer 2--> JSON items --checks--> review screen
                                (DistilRoBERTa                  (Qwen3-4B-Instruct,
                                 classifier)                     local)
```

| Step | What it does | File |
|---|---|---|
| Data | Builds a labelled dataset from the AMI Meeting Corpus (139 meetings, 110,795 utterances) | `src/prepare_ami.py` |
| Layer 1 | Flags utterances that may contain an action item or decision; compares keyword rules, TF-IDF and a fine-tuned transformer | `src/train_layer1.py` |
| Layer 2 | Extracts items as JSON from the flagged utterances; checks the schema, the cited evidence, the owner and the deadline | `src/layer2.py` |
| Evaluation | Scores the full pipeline against AMI's human-written summaries | `src/eval_layer2.py` |
| App | Meetings, Summary and My Tasks screens (Lab 1 design) | `app/app.py` |

## Results

Results are in `results/`. All numbers come from 12 AMI test meetings with speakers the models never saw.
They are summarised in `results/RESULTS.md`.

## Running it

Requires Python 3.12 and an NVIDIA GPU with at least 12 GB of memory.

```
python -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# 1. Data: download ami_public_manual_1.6.2.zip from
#    https://groups.inf.ed.ac.uk/ami/download/ and unzip it into data/raw/ami
python src/prepare_ami.py

# 2. Layer 1 (about 5 minutes on an RTX 5070 laptop GPU)
python src/train_layer1.py

# 3. Full pipeline evaluation (about 35 minutes)
python src/eval_layer2.py

# 4. Demo app
streamlit run app/app.py
```

## Data and licences

- AMI Meeting Corpus annotations v1.6.2, CC BY 4.0. Carletta, J. et al. (2005). The AMI Meeting Corpus:
  A pre-announcement. *Machine Learning for Multimodal Interaction*, 28–39.
- DistilRoBERTa (`distilbert/distilroberta-base`), Apache 2.0.
- Qwen3-4B-Instruct-2507, Apache 2.0.
- Whisper small (`openai/whisper-small`), MIT.

Meeting recordings and transcripts are never committed (see `.gitignore`), in line with the Lab 7 privacy plan.

## AI use

Code in this repository was written with the help of Claude (Anthropic) and reviewed by the team.
