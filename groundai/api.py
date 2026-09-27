"""GroundAI REST API (FastAPI).

Run:  uvicorn groundai.api:app --reload --port 8000      (from backend/)
Docs: http://localhost:8000/docs
The web app in ../web is served at / and talks to these endpoints when opened from
the API server; the same app also runs standalone in the browser (deployed demo).
"""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import llm
from .engine import ESCALATION_TARGET
from .gaps import GapLog
from .snapshot import export_snapshot
from .store import CorpusStore

ROOT = Path(__file__).resolve().parents[2]
CORPUS = Path(os.getenv("GROUNDAI_CORPUS", ROOT / "data" / "corpus"))
CACHE = Path(os.getenv("GROUNDAI_CACHE", ROOT / "data" / "cache"))

app = FastAPI(title="GroundAI", version="1.0.0",
              description="Multimodal, citation-backed maintenance assistant with confidence-calibrated escalation.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

store = CorpusStore(CORPUS, CACHE).load()
engine = store.engine()
gaps = GapLog(CACHE / "groundai.db")


class QueryIn(BaseModel):
    question: str = Field(..., min_length=3, max_length=600)
    mode: str = Field("field", pattern="^(field|research)$")
    use_llm: bool = True


class ResolveIn(BaseModel):
    resolution: str = Field(..., min_length=3)


@app.get("/healthz")
def health():
    return {"ok": True, "documents": len(store.docs), "chunks": len(store.chunks), "llm": llm.enabled()}


@app.post("/api/query")
def query(q: QueryIn):
    r = engine.query(q.question, q.mode, generator=llm.generator if q.use_llm else None)
    if r["band"] == "escalate":
        r["escalation"] = gaps.log(r, ESCALATION_TARGET)
    return r


@app.get("/api/documents")
def documents():
    return [d.to_dict() for d in store.docs]


@app.get("/api/documents/{doc_id}/pages/{page}.jpg")
def page_image(doc_id: str, page: int):
    f = CACHE / "pages" / f"{doc_id}-{page}.jpg"
    if not f.exists():
        raise HTTPException(404, "page not found")
    return FileResponse(f, media_type="image/jpeg")


@app.post("/api/documents")
async def upload(file: UploadFile = File(...)):
    global engine
    suffix = Path(file.filename or "upload.pdf").suffix.lower()
    if suffix not in {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
        raise HTTPException(415, "Upload a PDF or an image (PNG, JPG, TIFF).")
    dest = CORPUS / Path(file.filename).name
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
    shutil.move(tmp.name, dest)
    doc = store.add(dest)
    engine = store.engine()           # re-index (BM25 + dense + knowledge graph)
    return doc.to_dict()


@app.get("/api/graph")
def graph():
    return engine.kg.export()


@app.get("/api/escalations")
def escalations():
    return {"tickets": gaps.list(), "gaps": gaps.gaps()}


@app.post("/api/escalations/{tid}/resolve")
def resolve(tid: str, body: ResolveIn):
    gaps.resolve(tid, body.resolution)
    return {"ok": True}


@app.get("/api/snapshot")
def snapshot():
    """Portable index (chunks, LSA model, graph, page images) used by the browser engine."""
    return export_snapshot(store, engine, CACHE, include_images=False)


WEB = ROOT / "web"
if WEB.exists():
    app.mount("/", StaticFiles(directory=WEB, html=True), name="web")
