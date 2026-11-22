# Contribution log

Each team member records their own work here (guidelines section 6: "evidence of balanced contribution").
Add rows yourself, in your own commits, so the git history shows who wrote what.
AI tools used for a piece of work are named in the last column (guidelines section 7).

| Date | Member | Role | What I did | Evidence (file, commit, board item, meeting) | AI tools used |
|---|---|---|---|---|---|
| Sep–Nov 2026 | Malyka Sardar | Project lead, Product Owner | Labs 1–10 lead author; project direction for the prototype (local-GPU design, Ollama, Docker, evaluation targets) | Labs 1–10; repo commits b442af9–8292f67 | Claude (writing, code generation), NotebookLM |
| | Shivanjali Chandel | AI Developer | | | |
| | Ethan Yitzhaky | Backend Developer | | | |
| | Mohmed Sajid Shaikh | Scrum Master, Frontend Developer | | | |

## Manual review of extracted items (hallucination rate)

`results/manual_review_sample.csv` holds 50 items drawn at random from the final system's test-set output.
Split by row number and fill in the empty columns for your rows, then commit the file:

| Rows | Reviewer |
|---|---|
| 1–13 | Shivanjali |
| 14–26 | Ethan |
| 27–38 | Mohmed |
| 39–50 | Malyka |

For each item, read `evidence_text` (what was actually said) and mark:
- `supported_by_evidence`: **y** if the item is a fair reading of what was said, **n** if it is invented or wrong.
- `owner_correct`: **y**, **n**, or **na** if no owner was given.
- `deadline_correct`: **y**, **n**, or **na** if no deadline was given.

## Stand-up records

| Date | Attendees | Recorded with consent of all (y/n) | Processed with EchoAI (y/n) | Notes file |
|---|---|---|---|---|
| | | | | |
