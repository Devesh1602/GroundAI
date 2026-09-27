"""Export the ingested corpus + models into one JSON the browser engine can load."""
from __future__ import annotations

import base64
import json
from pathlib import Path

from . import text as T
from .engine import THRESHOLDS, WEIGHTS


def export_snapshot(store, engine, cache: Path, include_images: bool = True, eval_results: Path | None = None) -> dict:
    docs = [d.to_dict() for d in store.docs]
    for d in docs:
        d["stats"] = {k: v for k, v in d["stats"].items()}
        if not include_images:
            d["images"] = [f"/api/documents/{d['id']}/pages/{i}.jpg" for i in range(1, d["pages"] + 1)]
        else:
            d["images"] = []
            for i in range(1, d["pages"] + 1):
                f = cache / "pages" / f"{d['id']}-{i}.jpg"
                d["images"].append("data:image/jpeg;base64," + base64.b64encode(f.read_bytes()).decode() if f.exists() else "")
    chunks = []
    for c in engine.chunks:
        cd = c.to_dict()
        chunks.append(cd)
    snap = {
        "version": 1,
        "docs": docs,
        "chunks": chunks,
        "lsa": engine.index.lsa.export(),
        "graph": engine.kg.export(),
        "synonyms": T.SYNONYM_GROUPS,
        "stopwords": sorted(T.STOPWORDS),
        "question_noise": sorted(T.QUESTION_NOISE),
        "policy": {"thresholds": THRESHOLDS, "weights": WEIGHTS},
    }
    if eval_results and Path(eval_results).exists():
        snap["eval"] = json.loads(Path(eval_results).read_text())
    return snap
