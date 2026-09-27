"""Corpus store: ingests a folder of documents once and caches the result as JSON."""
from __future__ import annotations

import json
from pathlib import Path

from .engine import Engine
from .ingest.pipeline import Ingestor
from .models import Chunk, Document


class CorpusStore:
    def __init__(self, corpus_dir: Path, cache_dir: Path):
        self.corpus_dir, self.cache_dir = Path(corpus_dir), Path(cache_dir)
        (self.cache_dir / "pages").mkdir(parents=True, exist_ok=True)
        self.docs: list[Document] = []
        self.chunks: list[Chunk] = []
        self.ingestor = Ingestor()

    # born-digital first so their identifiers form the OCR correction lexicon
    @staticmethod
    def _order(p: Path):
        return (p.suffix.lower() != ".pdf", p.name)

    def load(self, rebuild: bool = False) -> "CorpusStore":
        cache = self.cache_dir / "corpus.json"
        files = sorted([p for p in self.corpus_dir.iterdir() if p.suffix.lower() in {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}], key=self._order)
        if cache.exists() and not rebuild:
            data = json.loads(cache.read_text())
            if data.get("files") == [f.name for f in files]:
                self.docs = [Document(**d) for d in data["docs"]]
                self.chunks = [Chunk(**c) for c in data["chunks"]]
                return self
        self.docs, self.chunks = [], []
        # two passes: digital text first builds the lexicon used to correct OCR
        digital, raster = [], []
        import pdfplumber
        for f in files:
            is_raster = f.suffix.lower() != ".pdf"
            if not is_raster:
                with pdfplumber.open(str(f)) as pdf:
                    is_raster = sum(len(pg.chars) for pg in pdf.pages) < 40
            (raster if is_raster else digital).append(f)
        for f in digital + raster:
            self.add(f, persist=False)
        self.save([f.name for f in files])
        return self

    def add(self, path: Path, persist: bool = True) -> Document:
        doc, chunks, images = self.ingestor.ingest(path)
        self.docs = [d for d in self.docs if d.id != doc.id] + [doc]
        self.chunks = [c for c in self.chunks if c.doc_id != doc.id] + chunks
        for i, img in enumerate(images, start=1):
            (self.cache_dir / "pages" / f"{doc.id}-{i}.jpg").write_bytes(img)
        if persist:
            self.save(sorted({d.filename for d in self.docs}))
        return doc

    def save(self, files):
        (self.cache_dir / "corpus.json").write_text(json.dumps(
            {"files": files, "docs": [d.to_dict() for d in self.docs], "chunks": [c.to_dict() for c in self.chunks]}, indent=1))

    def engine(self) -> Engine:
        return Engine(self.docs, self.chunks)
