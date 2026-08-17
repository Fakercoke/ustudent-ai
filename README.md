# ustudent-ai

[github.com/Fakercoke/ustudent-ai](https://github.com/Fakercoke/ustudent-ai)

A course-enrolment assistant. A student writes one sentence in English or
Chinese; the service decides whether to look up handbook policy, query course
data, or complete an enrolment — and refuses when the handbook does not cover
the question.

This is the AI layer of a four-service system. The React front end and the
Spring Boot enrolment backend come with the course; **this repository is my
work**.

```
ustudent-frontend   React        :3000    student-facing UI
ustudent-backend    Spring Boot  :8080    courses, seats, enrolments, PostgreSQL
ustudent-ai         FastAPI      :8000    ← this repo
ustudent-postgres   PostgreSQL   :5432
```

---

## What it does

| Endpoint | What it is |
|---|---|
| `POST /agent-chat` | LangGraph ReAct agent with three tools and multi-turn memory. Routes policy questions to RAG, course questions to the backend, and can execute an enrolment. |
| `POST /rag-ask` | Retrieval-augmented Q&A over the handbook. Returns the answer plus every retrieved chunk with its source, section and distance. |
| `POST /can-graduate` | Rule check on credits and GPA. |
| `POST /ask`, `/ask/v2`, `/ask/v3` | Three prompt-engineering stages, kept as a record of the progression: system prompt → few-shot → enforced JSON. |
| `GET /health` | Liveness probe used by the container health check. |
| `GET /` | Landing page with live demos of the agent and the RAG endpoint. |

Interactive API docs at `/docs`.

---

## Run it

```bash
# Docker — the index is built during the image build, so a fresh clone
# produces an image that behaves identically anywhere.
docker build -t ustudent-ai .
docker run -p 8000:8000 --env-file .env ustudent-ai

# or locally
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env          # add your LLM_API_KEY
python scripts/build_index.py
pytest                        # 143 passed
uvicorn app.main:app --reload
```

Any OpenAI-compatible provider works — Groq, DeepSeek, OpenAI, OpenRouter —
by changing three lines in `.env`. No code change.

```bash
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat
LLM_API_KEY=...
```

---

## Engineering decisions worth reading

Every number below is measured, and the runs are reproducible with
`python scripts/eval_rag.py both` and
`python lessons/lesson-10-eval-safety/starter/eval.py golden`.

### Distance is not quality

Seven chunking configurations, evaluated against the same question set:

| chunk / overlap | chunks | answer rate | top-1 distance | context chars |
|---|---|---|---|---|
| 100 / 16 | 233 | **50%** | **0.226** | 299 |
| 300 / 50 | 79 | 100% | 0.274 | **898** |
| 600 / 100 | 40 | 100% | 0.369 | 1800 |
| 3000 / 500 | 9 | 100% | 0.463 | 8051 |

The configuration with the *best* similarity score answers *half* the
questions. Small chunks contain little noise, so the vector matches cleanly —
and they cut answers in half. Retrieval located the right section; what came
back was a fragment.

**Distance measures "did I find the right place". Answer rate measures "is what
I found enough". Selecting on distance alone picks the worst configuration in
the table.**

### A threshold tuned on the test set fails on real users

The first distance threshold, 0.40, was derived from the evaluation set's own
distribution and scored perfectly on it. Then:

| query | distance | retrieved | verdict at 0.40 |
|---|---|---|---|
| `How many credits do I need to graduate?` | 0.283 | § Graduation | answered |
| `how can i graduate` | 0.445 | **§ Graduation** | refused — wrong |
| `graduate` | 0.566 | **§ Graduation** | refused — wrong |
| `How do I appeal a final grade?` | 0.520 | § Grading scale | refused — correct |

Retrieval was right in every case; the gate was wrong. Loosening to 0.50 would
rescue the first and admit the last — they are 0.075 apart. No single threshold
separates them.

Replaced with two layers: a loose threshold (0.75) whose only job is to skip
generation on hopeless queries, and an abstention instruction in the prompt that
does the actual judging. Across ten cases, every unanswerable question was
caught by the second layer, not the first.

### Typos break the embedding, not the chunking

`gradute` retrieves the wrong section entirely. Four controlled experiments:

- `gradute` vs `graduate` — cosine similarity **0.549**. An unrelated word,
  `advising`, scores **0.491**. A single missing letter costs almost all of the
  word's meaning.
- Three chunking strategies (heading-aware, heading-aware without the title
  prefix, fixed 600-character window) — all three fail on the typo, all three
  succeed without it. Not a chunking problem.
- The same typo inside a long sentence scores 0.900 similarity. Impact scales
  with how much of the query the broken token represents.
- After the typo, the top ten results sit within 0.054 of each other and ranks
  one and two differ by 0.013. **The failure mode is not "picked the wrong
  chunk" — ranking has become noise.**

Fixed before retrieval: the query is normalised into one well-formed English
sentence. Chinese `毕业需要多少学分` went from 0.684 (wrong section) to 0.270
(correct). The rewrite is used **only for retrieval** — the answer prompt always
receives the user's original wording, so a bad rewrite cannot change which
question gets answered.

---

## Evaluation

Two sets, kept apart:

| set | size | purpose | refusal accuracy | retrieval | grounded |
|---|---|---|---|---|---|
| `dev-set.json` | 22 | tuning — used freely | 95% | 100% | 94% |
| `rag-eval.json` | 8 | held out — run once | 100% | 100% | 100% |

An LLM-as-judge harness grades answers semantically
(`lessons/lesson-10-eval-safety/starter/eval.py`). Both the judge **and the
system under test** run at `temperature=0`: a deterministic judge grading a
non-deterministic answer still produces a score that drifts between runs.
Deterministic calls are cached to disk, so a re-run is free and identical.

Three of the failures found during this work were defects in the *measurement*,
not the system — including one where a weak reference answer caused the judge to
mark a genuine retrieval failure as a pass. Fixing the metric moved the dev score
from an inflated 21/22 down to a truthful 19/22.

---

## Safety

**PII redaction** — email, student ID and Australian phone numbers are stripped
before anything is logged or returned. The audit line records counts, never
values. Phone patterns are matched before ID patterns; otherwise the eight-digit
run inside a phone number is consumed by the ID rule and the remainder is logged
in the clear.

**Prompt-injection screening** — retrieved chunks are checked before they are
quoted to the model. A hit stops the request and sets `blocked=true`; the model
is never called. Screening the corpus matters more than screening the question:
a user typing "ignore previous instructions" is attacking a prompt they cannot
see, while a poisoned document is already inside the trusted context and is
quoted as authoritative. The material block is fenced with untrusted-data
markers, and any marker already present in the text is stripped first — otherwise
an attacker closes the fence early and the remainder reads as instructions again.

Both are heuristics and are documented as such. Regex PII detection misses
unusual formats; pattern matching is bypassed by paraphrase. They raise the cost
of the easy attacks and give the audit log something to record. They are one
layer, not the defence.

---

## Failure states are distinct fields

```
used_fallback=true, blocked=false, degraded=false   handbook has no answer
used_fallback=true, blocked=false, degraded=true    answer exists, generation failed
used_fallback=true, blocked=true,  degraded=false   refused to process — attack signal
```

Collapsing these into one flag buries an attack inside ordinary traffic, and
makes an outage indistinguishable from a normal refusal.

When generation fails, retrieval has already succeeded — returning 500 would
discard completed work. The service returns the retrieved handbook sections
instead. Verified by pointing the service at an invalid API key: the request
still returns useful material.

---

## Known limitations

| | |
|---|---|
| **Exact-match retrieval** | `CS101` and `CS201` differ by one character; dense vectors cannot separate them. "CS101 一共有多少学分" retrieves CS201 material even though the corpus states `Credits: 3`. Needs hybrid retrieval (dense + BM25). |
| **No reranking** | Standard in production RAG. Not added because top-3 answer rate is already 100% on 67 chunks — the evaluation set lacks the resolution to justify the decision. |
| **Evaluation resolution** | Six answerable questions means the smallest measurable step is 16.7%. Five of seven chunk configurations tied at 100%; the choice was actually made on cost and threshold separation. |
| **Judge quality** | An LLM grading an LLM. Reasons are printed with `--show` so a disputed verdict can be checked by hand. |
| **Query rewriting can drift** | `deadline` was rewritten as "assignment submission deadline" when the handbook means the enrolment deadline. Mitigated by three constraints, tested on fifteen samples. |
| **Cross-chunk answers** | "The difference between dropping and withdrawing" spans three sections and relies on top-3 assembling it. Parent-document retrieval would fix this. |

---

## Tests

```
143 passed
```

Covering: chunk boundaries and step arithmetic, ID uniqueness across files,
index idempotency, query normalisation and its fallbacks, threshold gating,
structured abstention, unusable model output, graceful degradation, PII
patterns and match ordering, six injection patterns, boundary-marker forgery,
and the endpoint contracts.

---

## Repository

```
app/
  main.py                FastAPI entry point
  config.py              settings — provider-agnostic by design
  llm.py                 OpenAI-compatible client, 429 backoff, disk cache
  safety.py              PII redaction, injection detection, data fencing
  rag/
    index.py             heading-aware chunking, indexing, search
    query.py             query normalisation before retrieval
    pipeline.py          retrieve → gate → grounded prompt → answer
    parse.py             tolerant JSON extraction
  agent/agent.py         LangGraph ReAct agent, three tools, memory
  routes/                one module per endpoint
scripts/
  build_index.py         index builder — also run during docker build
  eval_rag.py            string-match evaluation, dev vs held-out
data/golden/             evaluation sets
lessons/                 course exercises, design documents, reports
docs/portfolio/          design rationale and experiment records (Chinese)
```

The design document, the seven-configuration experiment and the typo
root-cause analysis are under [`docs/portfolio/`](docs/portfolio/).

---

## Course context

Twelve-week bootcamp. This repository holds lessons 1–10; the front end and
enrolment backend are separate repositories provided by the course. Exercise
briefs and my submissions are under `lessons/`.
