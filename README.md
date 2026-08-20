# UStudent AI — Production-Oriented RAG & Agent System

**English** · [简体中文](README.zh-CN.md)

> An AI engineering project that turns a conventional university course-enrolment system into a grounded, tool-using assistant. It can answer policy questions from a handbook, query live course data, preserve multi-turn context, and execute enrolment actions through controlled backend tools.

**Live system:** [http://49.235.155.82/](http://49.235.155.82/)

**Interactive API docs:** [http://49.235.155.82:8000/docs](http://49.235.155.82:8000/docs)

**GitHub:** [github.com/Fakercoke/ustudent-ai](https://github.com/Fakercoke/ustudent-ai)

> The React UI and Spring Boot enrolment backend are course-provided integration services. This repository contains my independently implemented AI layer, evaluation work, safety controls, experiments, tests, and deployment configuration.

## Results at a glance

| Engineering result | Measured outcome |
|---|---:|
| Automated tests | **160 passed** |
| Held-out RAG evaluation | **8/8 (100%)** |
| Development evaluation | **20/22 (91%)** |
| Chinese retrieval distance | **0.684 → 0.270** |
| Typo query `gradute` | **0.639 → 0.445** |
| Bilingual regression cases | **9/9 correct retrieval** |

## Resume-ready summary

### Australia / English resume

**UStudent AI — RAG & Agent Course Assistant | Personal Project**

- Engineered a production-oriented RAG and LangGraph Agent service using FastAPI, ChromaDB and an OpenAI-compatible LLM client, enabling grounded policy Q&A, live course lookup, multi-turn memory and controlled enrolment actions.
- Diagnosed multilingual and typo-related embedding failures through controlled chunking and vector-similarity experiments; introduced constrained query normalisation and a two-layer abstention strategy, improving Chinese retrieval distance from **0.684 to 0.270** and achieving **8/8 held-out evaluation accuracy**.
- Built deterministic RAG evaluation, PII redaction, prompt-injection screening and a password-protected operations dashboard with **160 automated tests**, then integrated the service with a course-provided React/Spring Boot/PostgreSQL system and deployed the four-service stack to Tencent Cloud using Docker Compose.

## System architecture

```text
Student Browser
      │
      ▼
React Frontend (course-provided)
      ├── /api/* ──► Spring Boot Backend ──► PostgreSQL
      │                 ▲
      └── /ai/*  ──► UStudent AI Service
                        ├── RAG ──► Chroma + handbook
                        ├── Agent ─► get_course / enrol_course / handbook_qa
                        └── LLM ───► DeepSeek (OpenAI-compatible client)
```

The four services are deployed with Docker Compose on Tencent Cloud. Health checks, persistent database storage, reverse-proxy routing, memory limits, and automatic container restart are configured for a reproducible public demo.

---

## What I built

UStudent AI is the intelligence layer of a university course-enrolment system. A student can ask a question in English or Chinese; the service decides whether to retrieve handbook policy, query live course data, or invoke an enrolment tool. Answers are grounded in retrieved sources, and the system abstains when the handbook does not contain enough evidence.

Core capabilities:

- **Grounded RAG:** heading-aware chunking, Chroma retrieval, source citations, distance diagnostics, threshold gating, structured abstention, and graceful degradation.
- **Query normalisation:** corrects short, colloquial, multilingual, and misspelled queries before retrieval while preserving the original user message for answer generation.
- **Tool-using Agent:** LangGraph ReAct agent with `get_course`, `enrol_course`, and `handbook_qa` tools plus thread-level multi-turn memory.
- **Provider-independent LLM client:** works with DeepSeek, Groq, OpenAI, or another OpenAI-compatible provider through environment configuration only.
- **Evaluation:** separate development and held-out sets, deterministic LLM-as-judge scoring, disk caching, retrieval checks, groundedness checks, and regression tests.
- **Safety:** PII redaction, prompt-injection screening of retrieved documents, untrusted-data boundaries, and separate fallback/degraded/blocked observability states.
- **Operations:** request-level tracing, redacted query previews, RAG distance and failure-layer diagnosis, Agent tool traces, LLM token/cost accounting, offline eval history, and printable reports in a password-protected dashboard.
- **Deployment:** multi-container Docker Compose deployment with a React UI, Spring Boot backend, PostgreSQL, reverse proxy, persistent storage, health checks, and restart recovery.

## Public API

| Endpoint | Purpose |
|---|---|
| `POST /agent-chat` | Routes a request to RAG or backend tools and maintains thread memory. |
| `POST /rag-ask` | Answers handbook questions and returns the retrieved source chunks and distances. |
| `POST /can-graduate` | Performs a deterministic credits-and-GPA graduation check. |
| `POST /ask`, `/ask/v2`, `/ask/v3` | Preserves three prompt-engineering stages: system prompt, few-shot prompting, and enforced JSON. |
| `GET /health` | Container and deployment health probe. |
| `GET /docs` | Swagger/OpenAPI documentation. |
| `GET /ops` | Password-protected RAG quality, usage, token-cost and diagnosis dashboard. |

## Operations and RAG quality dashboard

The `/ops` dashboard joins signals from one request under the same request ID:

```text
HTTP request → retrieval distance/sources → fallback or answer
             → Agent tool calls → LLM tokens/errors → diagnosis → SQLite
```

It separates live risk signals from true quality measurement. Live traffic can show an empty retrieval, distance-gate refusal, model abstention, generation failure, security block, latency and token usage; it cannot prove that a plausible answer is correct. Correctness is reported separately from the latest fixed `dev` and held-out `golden` evaluation runs. A privacy-minimal Nginx JSON log supplies React page-entry counts without IPs, query strings, referrers or user agents. Input previews receive limited regex-based PII redaction and are capped at 200 characters, client addresses are salted and hashed, and the dashboard remains disabled until `OPS_ADMIN_PASSWORD` and a private salt are configured.

See the [Chinese operations and diagnosis guide](docs/operations-dashboard.md) for metric definitions and the incident workflow.

## The retrieval problem I diagnosed

### 1. A better vector distance did not mean a better answer

I evaluated seven chunking configurations instead of choosing a chunk size by intuition:

| chunk / overlap | chunks | answer rate | top-1 distance | context chars |
|---|---:|---:|---:|---:|
| 100 / 16 | 233 | **50%** | **0.226** | 299 |
| 300 / 50 | 79 | 100% | 0.274 | **898** |
| 600 / 100 | 40 | 100% | 0.369 | 1,800 |
| 3000 / 500 | 9 | 100% | 0.463 | 8,051 |

The smallest chunks produced the best similarity score but only half the correct answers because relevant passages were cut into incomplete fragments. This showed that distance measures how closely a chunk matches a query, not whether the returned context is sufficient to answer it.

### 2. A single distance threshold could not separate good and bad cases

The original `0.40` threshold scored well on the tuning set but rejected valid real-user queries:

| query | distance | retrieved result | outcome at 0.40 |
|---|---:|---|---|
| `How many credits do I need to graduate?` | 0.283 | Graduation | answered |
| `how can i graduate` | 0.445 | **Graduation** | incorrectly refused |
| `graduate` | 0.566 | **Graduation** | incorrectly refused |
| `How do I appeal a final grade?` | 0.520 | Grading scale | correctly refused |

The valid `graduate` query had a worse distance than an unanswerable appeal question. No single cutoff could correctly classify both. I replaced the hard decision with two layers:

1. a loose `0.75` threshold that rejects only hopeless retrievals;
2. a grounded abstention instruction that judges whether the retrieved context actually supports an answer.

### 3. The typo failure came from the embedding model, not chunking

The misspelling `gradute` retrieved an unrelated section. I tested three different indexing strategies—heading-aware chunks, chunks without title prefixes, and fixed 600-character windows—and all failed on the typo while all succeeded on `graduate`.

Additional evidence:

- `graduate` ↔ `gradute` cosine similarity: **0.549**
- `graduate` ↔ unrelated `advising`: **0.491**
- top-10 typo results were compressed into a **0.054** distance range
- rank 1 and rank 2 differed by only **0.013**

The ranking had become noise. I added constrained query normalisation before retrieval. The rewrite is used only to search; answer generation still receives the original question. This improved Chinese retrieval from `0.684` to `0.270` and the typo case from `0.639` to `0.445`.

## Evaluation and safety

| evaluation set | size | purpose | result |
|---|---:|---|---:|
| `dev-set.json` | 22 | iterative diagnosis and tuning | **20/22 (91%)** |
| `rag-eval.json` | 8 | held-out regression set | **8/8 (100%)** |

Both the system under test and the LLM judge run with `temperature=0`. Deterministic responses are cached, making repeated evaluation reproducible and inexpensive. During evaluation, I also found defects in the metric itself—including a weak reference answer that incorrectly marked a genuine retrieval failure as a pass.

Safety controls include:

- redaction of email addresses, student IDs, and Australian phone numbers;
- logging only redaction counts, never the detected values;
- prompt-injection screening on retrieved chunks before any LLM call;
- stripping forged untrusted-data boundary markers;
- separate `used_fallback`, `degraded`, and `blocked` states for operational diagnosis.

## Run locally

```bash
git clone https://github.com/Fakercoke/ustudent-ai.git
cd ustudent-ai
cp .env.example .env
# Add an OpenAI-compatible API key and provider configuration to .env

docker build -t ustudent-ai .
docker run --rm -p 8000:8000 --env-file .env ustudent-ai
```

Or run the development environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
python scripts/build_index.py
pytest
uvicorn app.main:app --reload
```

## Repository structure

```text
app/
  main.py                 FastAPI entry point
  llm.py                  provider-independent client, retry and cache
  safety.py               PII and prompt-injection controls
  rag/
    index.py              chunking, indexing and retrieval
    query.py              constrained query normalisation
    pipeline.py           retrieve → gate → generate → validate
  agent/agent.py          LangGraph agent, tools and memory
  routes/                 API endpoints
data/                     handbook, FAQ and course catalogue
data/golden/              evaluation datasets
scripts/                  indexing, evaluation and deployment scripts
tests/                    160 automated tests
docs/portfolio/           experiment records and interview notes
```

## Known limitations

- Dense retrieval still struggles with exact identifiers such as `CS101` versus `CS201`; hybrid dense + BM25 retrieval is the next improvement.
- Query rewriting adds latency and is a semantic black box. A rewrite can drift, especially for ambiguous one-word inputs.
- The evaluation set is intentionally small; the current scores demonstrate regression control, not universal production accuracy.
- The public demo currently uses an HTTP IP address. A domain, ICP filing, and HTTPS are deployment follow-ups.
- React page views count server entries/reloads from the front Nginx. Pure client-side SPA route changes do not produce a server request and are therefore not counted as separate views.
