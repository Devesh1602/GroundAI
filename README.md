# GroundAI

**A multimodal maintenance assistant that knows when to answer and when to escalate.**

GroundAI reads equipment manuals, technical tables, raster engineering drawings and scanned service reports, and answers technician questions with page-level citations and highlighted evidence. Before it writes anything, it scores the evidence. Weak or missing evidence produces an escalation ticket and a logged documentation gap instead of a guess.

| | |
|---|---|
| [Live](https://groundaai.netlify.app/)|
| API + web app | `docker compose up --build` then open http://localhost:8000 (API docs at `/docs`) |
| Tests | `make test` (Python/JS parity on 44 questions x 2 modes, API handlers) |

## What it does

- **Document ingestion** pdfplumber layout parsing (headings, paragraphs, numbered procedures), ruled tables split into one chunk per row with the header attached, contextual chunk headers.
- **OCR for scans** OpenCV deskew + Tesseract with per-line confidence and bounding boxes. OCR tokens are corrected against identifiers found in the born-digital documents (on the demo drawing `RO]` becomes `RO1`; the rule also maps e.g. `FQ001` to `F0001`).
- **Drawing understanding (computer vision)** OpenCV finds component boxes and wires on a raster single-line diagram, OCR reads each box and wire label, and the wiring becomes graph edges (`Q104 MCCB 100 A -> VFD-104 via 3 x 50 mm2 Cu`).
- **Knowledge graph** Assets, models, fault codes, parameters, terminals, parts and components with typed edges (`connected_to`, `has_model`, `variant_of`, `documents`, `references`). A question about pump P-104 is expanded along P-104 -> M-104 -> VFD-104 -> HX-500-055 -> HX-500, which is how the drive manual gets searched for a pump question. Exports to Neo4j.
- **Hybrid retrieval** BM25 (exact codes, part numbers) + dense vectors (LSA by default; sentence-transformers + FAISS optional), reciprocal rank fusion, entity boosts weighted by how much of the question a passage answers, per-document diversity so independent sources can corroborate.
- **Confidence gate** A calibrated score from question coverage (weighted by term specificity), named-equipment match, semantic and keyword strength, number of independent sources, OCR quality and cross-document value conflicts. Hard rules: unknown equipment or codes and requests to bypass protection always escalate.
- **Answers** Extractive answers quoted from the sources with `[n]` citations, procedures kept in order, a safety/lockout line whenever the technician is about to act. Optional LLM wording (Anthropic, any OpenAI-compatible endpoint, or Claude inside the artifact), with every generated line checked against the passage it cites.
- **Escalation and gap log** Tickets routed to a senior technician, grouped into documentation gaps (by missing equipment or topic). SQLite by default, PostgreSQL via `DATABASE_URL`.
- **Field and Research modes** Short step-first answers for the plant floor, or longer multi-document answers with more evidence.

## Results

| Set | Answer accuracy | Escalation recall | Unsafe answers | False escalations | Conflicts found |
|---|---|---|---|---|---|
| Development, 28 questions (used while tuning) | 100% | 100% | 0% | 0% | 1/1 |
| Held-out, 16 questions (written afterwards, run once) | 90% | 83% | 17% (1 of 6) | 10% (1 of 10) | n/a |

Answer accuracy requires the expected fact in the answer **and** a citation to the expected document. The held-out misses are reported as they happened: "How do I replace the HX-500 control board?" got a *Verify on site* answer instead of an escalation, and "Which digital input starts the drive?" was escalated at 0.44. Mean query time is about 2 ms on the 120-chunk corpus. See `data/eval/`.

## Quick start

```bash
# 1. Python pipeline (needs tesseract-ocr installed for scans and drawings)
pip install -r backend/requirements.txt
python scripts/make_corpus.py            # generate the demo documents (fictional equipment)
python scripts/build_snapshot.py --rebuild   # ingest: layout, tables, OCR, drawing CV, index, graph
python scripts/evaluate.py && python scripts/evaluate.py heldout.json

# 2. API + web app
cd backend && uvicorn groundai.api:app --reload --port 8000   # http://localhost:8000

# 3. Static demo (no server): build and open dist/index.html
python scripts/build_web.py --standalone
```

Docker: `docker compose up --build` (API, web app, PostgreSQL). Add `--profile graph` for Neo4j, then `python scripts/export_neo4j.py --load`.

LLM wording (optional): `GROUNDAI_LLM=anthropic ANTHROPIC_API_KEY=... ` or `GROUNDAI_LLM=openai OPENAI_BASE_URL=http://localhost:11434/v1` (Ollama, vLLM). Without an LLM, answers are extractive and still cited.

## Repository layout

```
backend/groundai/
  ingest/pipeline.py   PDF/image -> chunks (layout, tables, OCR routing, drawing handling)
  ingest/ocr.py        deskew, Tesseract lines + confidence, lexicon correction
  ingest/diagram.py    OpenCV box/wire detection -> components and connections
  text.py              tokenizer, stemmer, identifiers, synonyms (mirrored in web/engine.js)
  index.py             BM25, LSA dense index, optional neural encoder + FAISS
  kg.py                knowledge graph build + weighted expansion
  engine.py            plan -> retrieve -> evaluate confidence -> answer or escalate
  llm.py               grounded prompt, provider calls, per-line citation verification
  gaps.py              escalation tickets and documentation-gap grouping (SQLite/PostgreSQL)
  api.py               FastAPI endpoints
web/                   app (vanilla JS, no build step), engine.js = browser port of the engine
scripts/               corpus generator, snapshot/index build, evaluation, web build, Neo4j export
data/corpus/           5 demo documents; data/eval/ questions and results
tests/                 Python/JS parity, API handlers
docs/                  technical documentation, project summary, demo video script
```


Five fictional documents for a raw-water pump station, built to exercise each format: an AC drive service manual (HX-500), a centrifugal pump O&M manual (CP-300), a service bulletin that supersedes a lubrication interval in the pump manual, a raster single-line and control wiring drawing (DWG E-104), and a skewed, noisy scanned field service report (SR-2291). All equipment, manufacturers and people are invented.
