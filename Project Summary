# GroundAI: Project Summary

## Problem

Maintenance knowledge is spread across dense manuals, technical tables, wiring drawings and scanned service reports. Keyword search misses most of it, and general AI assistants answer fluently even when the evidence is weak. On a plant floor a confident wrong answer means a wrong repair, damaged equipment, downtime or an injury.

## Solution

GroundAI is a multimodal assistant that answers only from the documents it has, shows exactly where each statement comes from, and escalates when the documents do not support an answer.

1. **Understands every format** Layout-aware PDF parsing, table rows as individual facts, OCR with deskewing and domain correction for scanned reports, and computer vision that reads component boxes and wiring from raster drawings.
2. **Connects documents** An equipment knowledge graph links assets to models, motors, drives, fault codes and parameters, so "why is pump P-104 tripping?" also searches the drive manual for the drive that feeds that pump.
3. **Finds evidence two ways** Keyword search for exact codes and part numbers plus semantic search for paraphrases, fused and diversified across documents.
4. **Scores the evidence before answering** Coverage of the question, named-equipment match, semantic and keyword strength, independent sources, scan quality and conflicting values produce a calibrated confidence with three outcomes: *Grounded answer*, *Verify on site*, or *Escalated*. Safety-critical questions need more evidence; unknown equipment and requests to bypass protection always escalate.
5. **Answers with citations, or escalates and learns** Answers quote the sources with clickable citations that open the page with the passage highlighted. Escalations create a ticket for a senior technician and a documentation-gap entry for the knowledge team.

## Demonstrated on

A five-document pump-station corpus (fictional equipment): drive manual, pump manual, superseding service bulletin, raster single-line drawing, scanned service report.

- *"Why is pump P-104 tripping on overcurrent?"* → Grounded (0.95). Three documents agree: acceleration ramp too short and impeller blockage; cites the scanned report, the drive manual's troubleshooting section and the pump manual's overload table; shows the P-104 → M-104 → VFD-104 → HX-500 chain read from the drawing; adds the lockout step.
- *"What's the torque spec for the new XR-9 assembly?"* → Escalated (0.15). XR-9 is in no document, so no CP-300 torque values are offered in its place; ticket and gap logged.
- *"How often should the CP-300 bearings be greased?"* → Verify on site. The manual says 4000 h, the newer bulletin says 2000 h for continuous duty; GroundAI shows both and names the bulletin as superseding.

## Results

Development set (28 questions): 100% correct answers with correct source, 100% of undocumented questions escalated, conflict detected. Held-out set written after tuning (16 questions, run once): 90% correct answers, 5 of 6 undocumented questions escalated, 1 false escalation. About 2 ms per query.

## Impact

- Faster troubleshooting: the relevant table row, procedure step or drawing link in one answer instead of cross-referencing PDFs.
- Safer decisions: no unsupported values, higher evidence bar for safety-critical questions, lockout steps attached, protection-bypass requests refused.
- Better documentation over time: every escalation becomes a prioritised documentation gap.

## Technology

Python, FastAPI, pdfplumber, Tesseract OCR, OpenCV, BM25 + dense retrieval (LSA; sentence-transformers + FAISS optional), knowledge graph with Neo4j export, SQLite/PostgreSQL, LLM APIs (OpenAI/Claude) with per-line citation verification, dependency-free web app with a browser port of the engine, Docker Compose.

## Scalability and adoption

Stateless API behind a load balancer; corpus index rebuilt per site or per equipment family; PostgreSQL for tickets; FAISS/Qdrant for large corpora; any on-prem LLM through an OpenAI-compatible endpoint so documents never leave the plant; the browser build runs offline on a technician's tablet.

## Next

Bounding-box citations inside drawings (already stored per component), contradiction detection across manual revisions, larger labelled evaluation from real sites, symbol recognition for IEC schematics, shift hand-over reports generated from resolved tickets.
