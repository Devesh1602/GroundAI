# GroundAI technical documentation

## 1. Architecture


            ┌───────────────── INGESTION (Python) ─────────────────┐
PDF / image │ born-digital page ─ pdfplumber: words, fonts, tables  │
            │ image-only page  ─ OpenCV deskew ─ Tesseract lines    │──► chunks (text, table, table_row,
            │ drawing page     ─ OpenCV boxes + wires ─ OCR labels  │     ocr_text, figure, connection)
            └───────────────────────────────────────────────────────┘     each with page, section, bbox,
                                                                          OCR confidence, safety flag
chunks ──► BM25 index ─┐
       ──► dense index ├─► hybrid retrieval ─► confidence gate ─┬─► answer (extractive or LLM + verification)
       ──► knowledge   ┘      (RRF + entity                     └─► escalate: ticket + documentation gap
           graph               + graph expansion)


Two runtimes share one algorithm:

| Runtime | Used for | Notes |
|---|---|---|
| Python (`backend/groundai`) | Ingestion, API, evaluation, snapshot export | FastAPI, SQLite/PostgreSQL, FAISS / sentence-transformers / Neo4j |
| Browser (`web/engine.js`) | Deployed, offline use | Loads `snapshot.json` (chunks, LSA model, graph, page images). `tests/test_parity.py` requires identical evidence, decisions, confidence (±0.01) and answer text on all 44 evaluation questions in both modes |

## 2. Ingestion

**Page routing.** A page with more than 40 characters in its text layer is treated as born-digital; otherwise it is rasterised at 200 dpi and sent to OCR. Born-digital documents are ingested first so their identifiers form the OCR correction lexicon.

**Born-digital pages** (`ingest/pipeline.py`)
- Tables: `pdfplumber.find_tables()`. Each table becomes one `table` chunk and one `table_row` chunk per row, written as `Caption - Header: value; Header: value`. The caption is read from the line under the table. Row bounding boxes come from the table geometry.
- Text outside tables: lines are grouped into blocks by vertical gap, bold 11-13 pt lines become section headings, numbered procedure lines are kept together, footers are dropped.
- Every chunk gets a contextual header for indexing: `[about: <document entities>] <document> | <section> | <text>`. Document-level entities (e.g. the asset named in a report header) are attached to every chunk of that document, so a ROOT CAUSE section stays linked to P-104.

**Scanned pages** (`ingest/ocr.py`)
- Grayscale, median denoise, Otsu threshold, skew angle from `minAreaRect` over ink pixels, rotation if 0.2° < |angle| < 8°.
- Tesseract `image_to_data`: words grouped into lines with bounding boxes and mean confidence.
- Domain correction: a token that mixes letters and digits (or contains `]`, `|`, `!`) is rewritten with a confusion map (O/Q/D→0, I/l/|/]→1, S→5, B→8, Z→2) only if the result is an identifier that already exists in the corpus. Plain words are never touched.
- Lines are grouped into sections by ALL-CAPS headings and large gaps.

**Drawings** (`ingest/diagram.py`)
1. Component boxes: closed 4-vertex contours with fill ratio > 0.85 and a plausible size; duplicate inner/outer borders merged; "phantom" rectangles formed by two parallel wires between two boxes are removed.
2. Box text: Tesseract on each box interior.
3. Wires: morphological opening with long horizontal and vertical kernels, box areas and the sheet frame masked out, connected components.
4. A wire component that touches the rings around two or more boxes becomes an edge. Vertical alignment means a power path ("feeds"), otherwise a signal ("is wired to").
5. Labels: the page is OCR'd again with the boxes painted out; the nearest short free-text line to each wire becomes its label.
Output: `figure` chunks per component, `connection` chunks per edge, a drawing overview chunk, and free-text notes.

## 3. Indexing

- Tokeniser: lower-case alphanumeric runs that keep internal `-`, `.`, `:` (so `P-104`, `22.11`, `X1:8` survive), plus their parts; light suffix stemmer; stop-word list. Identifier regex for tags, models, codes and parameters; canonical form removes hyphens (`XR-9` = `XR9`).
- BM25: k1 = 1.2, b = 0.75.
- Dense: TF-IDF (sublinear tf, smooth idf, L2) projected with a 64-dimension truncated SVD (LSA). Chosen because the same model runs in the browser. Setting `GROUNDAI_EMBEDDER=neural` swaps in any sentence-transformers model with a FAISS inner-product index.
- Synonym groups (trip/fault/alarm, overcurrent/overload, grease/lubrication, ...) are used for query expansion at weight 0.35 and count as 0.7 of a match in coverage.

## 4. Knowledge graph (`kg.py`)

Node types: asset, model, fault, parameter, terminal, part, document_ref, component, document. Edges: `connected_to` (from drawings), `has_model` (drawing boxes and `TAG (MODEL` patterns), `variant_of` (`HX-500-055` → `HX-500`), `documents` (document → subjects), `references` (fault ↔ parameter in the same row), `co_occurs`.
Query expansion is a cheapest-path search from the question's entities with edge costs connected_to 0.75, has_model 0.4, variant_of 0.2, documents 0.9 and a budget of 2.2. Each reached entity adds its tokens to the query with weight 0.6 × (1 − cost/2.7). The same paths are shown to the technician ("Equipment chain for P-104 ...") and the drawing connections along the path are cited.

## 5. Retrieval

1. BM25 and dense scores for every chunk; reciprocal rank fusion over the top 40 of each (k = 60).
2. Entity boosts: +0.012 per question entity found in the chunk and +0.006 × graph weight, both multiplied by the chunk's coverage of the question, so a drawing box that only names P-104 does not outrank the manual section that explains overcurrent.
3. Drawing overview ×0.6; root-cause / troubleshooting sections ×1.25 for "why" questions.
4. Greedy selection with a 0.8^n per-document discount, max 3 chunks per page, a specific table row replaces its whole table when it scores at least 60% as high.
5. k = 7 (Field) or 10 (Research), plus graph evidence (drawing connections on the entity path) and a lockout/warning passage from the same manual when the question implies work on equipment.

## 6. Confidence gate (`engine.py: evaluate`)

Signals (all shown in the UI under "Why this confidence"):

| Signal | Definition |
|---|---|
| coverage | 0.7 × best single passage + 0.3 × top-3 union; each question term weighted by its IDF, terms absent from the corpus get the maximum IDF |
| entities | share of question identifiers present in the top-5 evidence |
| semantic | top dense cosine |
| keyword | BM25 top / (BM25 top + 6) |
| sources | documents in the top 5 that each cover ≥ 50% of the question |
| OCR | lowest OCR confidence among the top 3 |

`confidence = sigmoid(−4.0 + 3.4·coverage + 1.6·semantic + 1.2·keyword + 0.8·min(sources−1, 2)/2 + 1.0·entities − 1.5·max(0, 0.9 − OCR))`

| Band | Rule |
|---|---|
| Grounded | ≥ 0.70, or ≥ 0.80 when the question is safety-critical (torque, voltage, isolation, pressure, insulation, capacitors) |
| Verify on site | 0.45 to the grounded bar; also any answer where documents disagree on a value |
| Escalate | < 0.45; any unknown identifier (hard cap 0.15); any request to bypass/jumper/disable protection or interlocks (cap 0.20, the manual's prohibition is cited) |

Conflict detection: for value questions, numbers with units are extracted from top passages of manuals and bulletins that cover ≥ 50% of the question. Different values for the same unit in different documents are a conflict; the document that declares `supersedes` (or else the newest) is named as preferred.

## 7. Answer generation

- **Extractive (default).** Passages are split into sentences, table rows and list items; units must cover ≥ 40% of the question (a unit naming the asked-about entity gets a bonus), ranked by weighted term match and passage rank, near-duplicates removed. Procedures and numbered findings are reproduced in order. A safety line is appended for procedures, "why" and safety-critical questions.
- **LLM (optional).** Prompt in `llm.py` / `engine.js`: evidence only, a citation on every sentence, numbers copied verbatim, `INSUFFICIENT_EVIDENCE` if unsupported. Each generated sentence is verified: it must cite existing passages, ≥ 34% of its content terms must appear in the cited text and every number in it must appear there. If more than 30% of lines fail, the extractive answer is shown instead. The LLM is only called after the gate has decided to answer.

## 8. Escalation and documentation gaps (`gaps.py`)

Every escalation creates a ticket `{id, created_at, question, mode, reason, confidence, topic, routed_to, status, resolution, evidence}`. The topic is the unknown identifier, "Protection bypass request", or the uncovered terms. Gaps are tickets grouped by topic and sorted by open count. SQLite file by default; PostgreSQL when `DATABASE_URL` is set. In the static demo the queue is kept in the viewer's browser.

## 9. API

| Method | Path | |
|---|---|---|
| GET | `/healthz` | documents, chunks, LLM status |
| POST | `/api/query` | `{question, mode: field|research, use_llm}` → band, confidence, signals, reasons, answer blocks with citations, evidence with bounding boxes, graph paths, escalation ticket |
| GET | `/api/documents` | document metadata and ingestion stats |
| POST | `/api/documents` | upload PDF/image, runs the full pipeline and re-indexes |
| GET | `/api/documents/{id}/pages/{n}.jpg` | page image for the evidence viewer |
| GET | `/api/graph` | knowledge graph |
| GET | `/api/escalations` | tickets and grouped gaps |
| POST | `/api/escalations/{id}/resolve` | `{resolution}` |
| GET | `/api/snapshot` | portable index for the browser engine |

## 10. Setup

Requirements: Python 3.10+, Tesseract 5 (`apt install tesseract-ocr`), Node 18+ only for the parity test.

```bash
pip install -r backend/requirements.txt
make all            # corpus -> ingest/index -> eval -> web build -> tests
make serve          # http://localhost:8000
docker compose up --build              # API + web + PostgreSQL
docker compose --profile graph up      # + Neo4j, then: python scripts/export_neo4j.py --load
```

Environment: `GROUNDAI_CORPUS`, `GROUNDAI_CACHE`, `DATABASE_URL`, `GROUNDAI_LLM` (`anthropic|openai`), `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `GROUNDAI_LLM_MODEL`, `GROUNDAI_EMBEDDER=neural`, `GROUNDAI_EMBED_MODEL`.

Adding your own documents: drop PDFs or images into `data/corpus/` and run `python scripts/build_snapshot.py --rebuild`, or upload through the web app / `POST /api/documents`.

Static deployment: `python scripts/build_web.py --standalone` writes `dist/index.html` (about 1.6 MB, everything inlined). Host it on GitHub Pages, Netlify or Vercel as-is.

## 11. Limitations and next steps

- The confidence weights were set by hand on the development set; the held-out set shows the expected drop (one unsafe *Verify* answer, one false escalation). Next: fit the weights on a larger labelled set from real sites and report calibration curves.
- Partial topic overlap is the main failure mode: "HX-500 control board" matched "control panel battery" and "control terminals". Planned fix: a noun-phrase level check that the asked-about component itself appears in the evidence.
- Table detection relies on ruled tables; borderless tables need a layout model (e.g. Table Transformer).
- Drawing parsing handles box-and-wire diagrams; symbol recognition for IEC schematic symbols is future work.
- In the browser demo, uploaded scans are transcribed only when the page is opened inside Claude (vision); the Python pipeline handles them with Tesseract.
