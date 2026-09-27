"""Hybrid index: BM25 (keyword precision) + dense vectors (semantic recall).

Dense vectors come from one of two encoders:
  * "lsa"    - TF-IDF -> truncated SVD (latent semantic analysis). Pure numpy, small, and
               exportable, so the exact same model runs in the browser (web/engine.js).
  * "neural" - any sentence-transformers model (e.g. BAAI/bge-small-en-v1.5) when the
               package is installed and GROUNDAI_EMBEDDER=neural. Stored in FAISS if
               faiss is installed, otherwise a numpy matrix.
"""
from __future__ import annotations

import math
import os
from collections import Counter

import numpy as np

from . import text as T
from .models import Chunk

try:  # optional accelerators
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.len = np.array([len(d) for d in docs], dtype=float)
        self.avgdl = float(self.len.mean()) if len(docs) else 1.0
        df = Counter()
        for d in docs:
            df.update(set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, q: dict[str, float]) -> np.ndarray:
        s = np.zeros(len(self.tf))
        for t, w in q.items():
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i, tf in enumerate(self.tf):
                f = tf.get(t)
                if f:
                    s[i] += w * idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avgdl))
        return s


class LSA:
    """TF-IDF (sublinear tf, smooth idf, l2) -> SVD. Mirrored in web/engine.js."""

    def __init__(self, docs: list[list[str]], k: int = 64):
        vocab = sorted({t for d in docs for t in d})
        self.vocab = {t: i for i, t in enumerate(vocab)}
        n, V = len(docs), len(vocab)
        df = np.zeros(V)
        for d in docs:
            for t in set(d):
                df[self.vocab[t]] += 1
        self.idf = np.log((n + 1) / (df + 1)) + 1
        X = np.zeros((n, V))
        for i, d in enumerate(docs):
            X[i] = self._tfidf(Counter(d))
        k = max(2, min(k, n - 1, V - 1))
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        self.components = Vt[:k]                  # k x V
        D = X @ self.components.T
        self.doc_vecs = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-9)

    def _tfidf(self, tf: Counter, weights: dict[str, float] | None = None) -> np.ndarray:
        v = np.zeros(len(self.vocab))
        for t, f in tf.items():
            j = self.vocab.get(t)
            if j is not None:
                v[j] = (1 + math.log(f)) * self.idf[j] * (weights.get(t, 1.0) if weights else 1.0)
        nrm = np.linalg.norm(v)
        return v / nrm if nrm else v

    def encode(self, q: dict[str, float]) -> np.ndarray:
        tf = Counter({t: 1 for t in q})
        v = self._tfidf(tf, q) @ self.components.T
        nrm = np.linalg.norm(v)
        return v / nrm if nrm else v

    def export(self) -> dict:
        inv = sorted(self.vocab, key=self.vocab.get)
        return {"vocab": inv, "idf": [round(float(x), 5) for x in self.idf],
                "components": [[round(float(x), 5) for x in row] for row in self.components]}


class NeuralEncoder:  # pragma: no cover - optional, needs sentence-transformers
    def __init__(self, texts: list[str], model: str):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model)
        E = self.model.encode(texts, normalize_embeddings=True)
        self.doc_vecs = np.asarray(E, dtype="float32")
        self.faiss = None
        if faiss is not None:
            self.faiss = faiss.IndexFlatIP(self.doc_vecs.shape[1])
            self.faiss.add(self.doc_vecs)

    def search(self, question: str) -> np.ndarray:
        q = self.model.encode([question], normalize_embeddings=True).astype("float32")
        return (self.doc_vecs @ q[0]).astype(float)


class HybridIndex:
    def __init__(self, chunks: list[Chunk], lsa_dims: int = 64):
        self.chunks = chunks
        self.toks = [T.tokens(c.ctx) for c in chunks]
        self.bm25 = BM25(self.toks)
        self.lsa = LSA(self.toks, lsa_dims)
        self.neural = None
        if os.getenv("GROUNDAI_EMBEDDER") == "neural":  # pragma: no cover
            self.neural = NeuralEncoder([c.ctx for c in chunks], os.getenv("GROUNDAI_EMBED_MODEL", "BAAI/bge-small-en-v1.5"))
        self.token_sets = [set(t) for t in self.toks]
        self.id_sets = [{T.canon_id(i) for i in T.identifiers(c.text + " " + c.section + " " + " ".join(c.meta.get("about", [])))} for c in chunks]

    def dense(self, q: dict[str, float], question: str = "") -> np.ndarray:
        if self.neural is not None and question:  # pragma: no cover
            return self.neural.search(question)
        return self.lsa.doc_vecs @ self.lsa.encode(q)
