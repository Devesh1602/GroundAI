"""GroundAI query engine: hybrid retrieval -> confidence evaluation -> answer or escalate.

The confidence gate runs BEFORE any text is generated. A generator (extractive by
default, an LLM when configured) is only invoked when the evidence justifies it.
"""
from __future__ import annotations

import math
import re
import time
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from . import text as T
from .index import HybridIndex
from .kg import KnowledgeGraph, keep_id
from .models import Chunk, Document

# ----------------------------------------------------------------------------- policy
THRESHOLDS = {"grounded": 0.70, "verify": 0.45, "grounded_safety": 0.80}
WEIGHTS = {"bias": -4.0, "coverage": 3.4, "dense": 1.6, "bm25": 1.2, "support": 0.8, "ids": 1.0, "ocr": 1.5}

SAFETY_BYPASS = re.compile(r"\b(bypass|jumper|defeat|disable|override|short(?:ing)? out|bridge|ignore)\b.{0,40}\b(protection|interlock|earth fault|ground fault|overcurrent|thermal|safety|trip|alarm|guard)"
                           r"|\b(protection|interlock|earth fault|thermal|guard)\b.{0,30}\b(bypass|jumper|defeat|disable|override)", re.I)
SAFETY_CRITICAL = re.compile(r"torque|voltage|live|lockout|loto|isolat|pressure|megger|insulation|capacitor|discharge|current limit|30\.17", re.I)
VALUE_Q = re.compile(r"^(what|which|how (much|many|often|long))\b|torque|interval|rating|rated|spec|default|value|limit|size|range|how often", re.I)
WHY_Q = re.compile(r"^why\b|\bcause|\breason|keeps? (trip|fail)|\btripping\b", re.I)
PROC_Q = re.compile(r"^how (do|to|should|can|would)\b|procedure|steps?\b|replace|reset|install|remove", re.I)
UNIT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(operating hours|hours|h|Nm|kW|A|V|s|bar|mm|MOhm|years?|months?|deg C)\b(?!\s*/)")
CAUSE_RE = re.compile(r"root cause|probable cause|troubleshoot|findings|most often caused", re.I)
ESCALATION_TARGET = "Senior maintenance technician (on-call)"


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


@dataclass
class QueryPlan:
    question: str
    mode: str
    ids: list[str]
    known_ids: list[str]
    unknown_ids: list[str]
    content: list[str]
    weights: dict[str, float]
    expansions: dict[str, float]
    kg: dict[str, dict]
    intents: dict[str, bool]


@dataclass
class Evidence:
    n: int
    chunk: Chunk
    bm25: float
    dense: float
    score: float
    matched: list[str] = field(default_factory=list)
    coverage: float = 0.0
    idx: int = -1


class Engine:
    def __init__(self, docs: list[Document], chunks: list[Chunk]):
        self.docs = {d.id: d for d in docs}
        self.chunks = chunks
        self._contextualise_about(docs, chunks)
        self.by_id = {c.id: c for c in chunks}
        self.index = HybridIndex(chunks)
        self.kg = KnowledgeGraph().build(docs, chunks)
        self.corpus_ids = set().union(*self.index.id_sets) if chunks else set()
        for d in docs:
            self.corpus_ids |= {T.canon_id(s) for s in d.subjects}
        self.max_idf = max(self.index.bm25.idf.values()) if self.index.bm25.idf else 5.0

    @staticmethod
    def _contextualise_about(docs, chunks):
        """Document-level entities ("this report is about P-104 / VFD-104") are attached to
        every chunk of the document, so a section like ROOT CAUSE stays linked to its asset."""
        from .kg import id_type
        for d in docs:
            dch = [c for c in chunks if c.doc_id == d.id]
            about = [s for s in d.subjects if keep_id(s)]
            for c in sorted(dch, key=lambda c: (c.page, c.bbox[1]))[:2]:
                for i in T.identifiers(c.text):
                    if keep_id(i) and id_type(i) in ("asset", "model") and i not in about:
                        about.append(i)
            for c in dch:
                c.meta["about"] = about[:6]
                if about and "[about:" not in c.ctx:
                    c.ctx = f"[about: {' '.join(about[:6])}] " + c.ctx

    # ------------------------------------------------------------------ analysis
    def plan(self, question: str, mode: str = "field") -> QueryPlan:
        ids = [i for i in T.identifiers(question) if keep_id(i)]
        canon = [T.canon_id(i) for i in ids]
        known = [c for c in canon if c in self.corpus_ids]
        unknown = [i for i, c in zip(ids, canon) if c not in self.corpus_ids]
        toks = [t for t in T.tokens(question) if t not in T.QUESTION_NOISE]
        id_toks = set(T.tokens(" ".join(ids)))
        content = []
        for t in toks:
            if t not in id_toks and len(t) > 1 and not t.isdigit() and t not in content:
                content.append(t)
        weights: dict[str, float] = Counter()
        full_ids = {t for t in id_toks if re.search(r"[-.:]", t)}
        for t in toks:
            if t in full_ids or (t in id_toks and not (id_toks - {t}) & set(toks) - {t}):
                weights[t] = 1.5           # the identifier itself (P-104, F0001)
            elif t in id_toks:
                weights[t] = 0.3           # fragments of an identifier ("p", "104")
            else:
                weights[t] = 1.0
        expansions: dict[str, float] = {}
        for s in T.expand(content):
            expansions[s] = 0.35
        kg = self.kg.expand(known)
        for key, info in kg.items():
            for t in T.tokens(info["label"]):
                if re.search(r"[-.:]", t) or t.isalnum() and len(t) > 3:
                    expansions[t] = max(expansions.get(t, 0), round(0.6 * info["weight"], 3))
        for t, w in expansions.items():
            if t not in weights:
                weights[t] = w
        intents = {"safety_bypass": bool(SAFETY_BYPASS.search(question)),
                   "safety_critical": bool(SAFETY_CRITICAL.search(question)),
                   "value": bool(VALUE_Q.search(question)), "why": bool(WHY_Q.search(question)),
                   "procedure": bool(PROC_Q.search(question))}
        return QueryPlan(question, mode, ids, known, unknown, content, dict(weights), expansions, kg, intents)

    # ------------------------------------------------------------------ retrieval
    def retrieve(self, p: QueryPlan, k: int) -> list[Evidence]:
        bm = self.index.bm25.scores(p.weights)
        dn = self.index.dense(p.weights, p.question)
        rrf = np.zeros(len(self.chunks))
        for arr in (bm, dn):
            order = [i for i in np.argsort(-arr, kind="stable")[:40] if arr[i] > 0]
            for r, i in enumerate(order):
                rrf[i] += 1 / (60 + r)
        qids = set(p.known_ids)
        for i, ids in enumerate(self.index.id_sets):
            # entity matches only count in proportion to how much of the actual question
            # the chunk answers - a drawing box that merely names P-104 must not win
            cov = self._coverage(p.content, self.index.token_sets[i]) if p.content else 1.0
            if qids:
                rrf[i] += 0.012 * len(qids & ids) * cov
            kgw = sum(info["weight"] for key, info in p.kg.items() if key in ids)
            rrf[i] += 0.006 * min(kgw, 2.0) * cov
            if self.chunks[i].section == "Drawing overview":
                rrf[i] *= 0.6
            if p.intents["why"] and CAUSE_RE.search(self.chunks[i].section + " " + self.chunks[i].text[:120]):
                rrf[i] *= 1.25   # root-cause / troubleshooting sections answer "why" questions
        # greedy selection with a per-document diversity discount, so one document
        # cannot crowd out independent corroborating sources
        out: list[Evidence] = []
        per_page: Counter = Counter()
        per_doc: Counter = Counter()
        tables_used: set[str] = set()
        chosen: set[int] = set()
        pool = [int(i) for i in np.argsort(-rrf, kind="stable")[:60] if rrf[i] > 0]
        while pool and len(out) < k:
            i = max(pool, key=lambda j: rrf[j] * (0.8 ** per_doc[self.chunks[j].doc_id]))
            pool.remove(i)
            c = self.chunks[i]
            if c.kind == "table":
                rows = [j for j in pool if self.chunks[j].kind == "table_row" and self.chunks[j].id.rsplit("r", 1)[0] == c.id]
                if rows:
                    j = max(rows, key=lambda j: rrf[j])
                    if rrf[j] >= 0.6 * rrf[i]:
                        pool.remove(j)
                        i, c = j, self.chunks[j]   # a specific row is better evidence than the whole table
            tkey = c.id.split("r")[0] if c.kind == "table_row" else c.id
            if c.kind in ("table", "table_row"):
                if tkey in tables_used and c.kind == "table":
                    continue
                if c.kind == "table_row" and c.id.rsplit("r", 1)[0] in {e.chunk.id for e in out}:
                    continue
            if per_page[(c.doc_id, c.page)] >= 3:
                continue
            per_page[(c.doc_id, c.page)] += 1
            per_doc[c.doc_id] += 1
            if c.kind == "table_row":
                tables_used.add(c.id.rsplit("r", 1)[0])
            toks = self.index.token_sets[i]
            matched = [t for t in p.content if self._has(t, toks)] + [x for x in p.known_ids if x in self.index.id_sets[i]]
            out.append(Evidence(len(out) + 1, c, float(bm[i]), float(dn[i]), float(rrf[i]), matched,
                                self._coverage(p.content, toks), int(i)))
        return out

    # ------------------------------------------------------------------ confidence
    def _idf(self, t: str) -> float:
        return self.index.bm25.idf.get(t, self.max_idf)

    def _has(self, t: str, toks: set[str]) -> bool:
        return t in toks or any(s in toks for s in T.synonym_index().get(t, []))

    def _coverage(self, terms: list[str], toks: set[str]) -> float:
        if not terms:
            return 1.0
        tot = sum(self._idf(t) for t in terms)
        got = 0.0
        for t in terms:
            if t in toks:
                got += self._idf(t)
            elif any(s in toks for s in T.synonym_index().get(t, [])):
                got += 0.7 * self._idf(t)          # a synonym is weaker evidence than the term
        return got / tot

    def conflicts(self, p: QueryPlan, ev: list[Evidence]) -> dict | None:
        if not p.intents["value"]:
            return None
        found: dict[str, dict[str, set]] = {}
        for e in ev[:5]:
            d = self.docs[e.chunk.doc_id]
            if d.doc_type not in ("manual", "bulletin") or e.coverage < 0.5:
                continue
            for m in UNIT_RE.finditer(e.chunk.text):
                u = m.group(2).lower()
                unit = "h" if u in ("operating hours", "hours", "h") else "years" if u.startswith("year") else "months" if u.startswith("month") else m.group(2)
                found.setdefault(unit, {}).setdefault(d.id, set()).add(float(m.group(1)))
        for unit, per_doc in found.items():
            if len(per_doc) < 2:
                continue
            docs = list(per_doc)
            vals = [per_doc[x] for x in docs]
            if any(vals[i] - vals[j] for i in range(len(vals)) for j in range(len(vals)) if i != j):
                newest = max(docs, key=lambda x: self.docs[x].date or "")
                supersedes = [x for x in docs if self.docs[x].supersedes]
                winner = supersedes[0] if supersedes else newest
                others = [x for x in docs if x != winner]
                values = [{"doc": self.docs[x].short, "doc_id": x, "date": self.docs[x].date,
                           "values": sorted(per_doc[x]), "n": next(e.n for e in ev if e.chunk.doc_id == x)} for x in docs]
                why = "explicitly supersedes" if supersedes else "is the most recent"
                resolution = (f"{self.docs[winner].short} ({self.docs[winner].date}) {why} "
                              f"{', '.join(self.docs[o].short for o in others)}. Use the newer value and confirm the duty conditions on site.")
                return {"unit": unit, "values": values, "preferred_doc": winner, "resolution": resolution}
        return None

    def evaluate(self, p: QueryPlan, ev: list[Evidence]) -> dict:
        top = ev[:3]
        union = set().union(*[self.index.token_sets[e.idx] for e in top]) if top else set()
        if p.content:
            # the best single passage matters most; terms scattered across unrelated
            # passages ("firmware" nowhere, "update" somewhere) must not add up to an answer
            best_single = max((e.coverage for e in ev[:5]), default=0.0)
            coverage = 0.7 * best_single + 0.3 * self._coverage(p.content, union)
        else:
            coverage = 1.0 if p.known_ids else 0.0
        missing = [t for t in p.content if not self._has(t, union)]
        ev_ids = set().union(*[self.index.id_sets[e.idx] for e in ev[:5]]) if ev else set()
        id_cov = (len([x for x in p.known_ids if x in ev_ids]) / len(p.known_ids)) if p.known_ids else None
        dense_top = max([e.dense for e in top], default=0.0)
        bm_top = max([e.bm25 for e in ev], default=0.0)
        bm_sat = bm_top / (bm_top + 6.0)
        support_docs = sorted({e.chunk.doc_id for e in ev[:5] if e.coverage >= 0.5})
        ocr_vals = [e.chunk.ocr_conf for e in top if e.chunk.ocr_conf is not None]
        ocr_min = min(ocr_vals) if ocr_vals else 1.0
        conflict = self.conflicts(p, ev)
        W = WEIGHTS
        z = (W["bias"] + W["coverage"] * coverage + W["dense"] * max(0.0, dense_top) + W["bm25"] * bm_sat
             + W["support"] * min(len(support_docs) - 1, 2) / 2 + W["ids"] * (id_cov if id_cov is not None else 0.5)
             - W["ocr"] * max(0.0, 0.9 - ocr_min))
        conf = sigmoid(z)
        reasons: list[str] = []
        band = None
        if p.intents["safety_bypass"]:
            band, conf = "escalate", min(conf, 0.2)
            reasons.append("Request asks to bypass or disable a protection function. Only an authorized engineer can approve this.")
        if p.unknown_ids:
            band, conf = "escalate", min(conf, 0.15)
            reasons.append(f"{', '.join(p.unknown_ids)} does not appear in any indexed document.")
        if not ev:
            band, conf = "escalate", 0.0
            reasons.append("No supporting evidence retrieved.")
        if band is None:
            need = THRESHOLDS["grounded_safety"] if p.intents["safety_critical"] else THRESHOLDS["grounded"]
            if conf >= need:
                band = "grounded"
            elif conf >= THRESHOLDS["verify"]:
                band = "verify"
            else:
                band = "escalate"
            if conflict and band == "grounded":
                band = "verify"
            if band == "verify" and p.intents["safety_critical"] and conf < THRESHOLDS["grounded"]:
                reasons.append("Safety-critical question: a higher evidence bar applies.")
        if missing and band != "grounded":
            reasons.append("Not covered by the retrieved evidence: " + ", ".join(f"'{m}'" for m in missing[:5]) + ".")
        if len(support_docs) >= 2 and band != "escalate":
            reasons.append(f"{len(support_docs)} independent documents support the answer.")
        if ocr_min < 0.85 and band != "escalate":
            reasons.append(f"Part of the evidence comes from OCR with {ocr_min:.0%} confidence.")
        if conflict:
            reasons.append("Documents disagree on a value. " + conflict["resolution"])
        if band == "grounded" and not reasons:
            reasons.append("Strong, consistent evidence found.")
        if band == "escalate" and not any("escalat" in r.lower() for r in reasons) and not p.unknown_ids and not p.intents["safety_bypass"]:
            reasons.append("Evidence is too weak to answer without guessing.")
        return {"band": band, "confidence": round(conf, 3), "reasons": reasons, "missing_terms": missing,
                "signals": {"coverage": round(coverage, 3), "id_coverage": None if id_cov is None else round(id_cov, 3),
                            "dense_top": round(dense_top, 3), "bm25_top": round(bm_top, 2), "bm25_sat": round(bm_sat, 3),
                            "support_docs": len(support_docs), "ocr_min": round(ocr_min, 3), "conflict": bool(conflict),
                            "z": round(z, 3)},
                "conflict": conflict}

    # ------------------------------------------------------------------ answer (extractive)
    def _units(self, e: Evidence) -> list[str]:
        c = e.chunk
        if c.kind == "table_row":
            row = c.meta.get("row", []); head = c.meta.get("header", [])
            if row and head:
                first = f"{head[0]} {row[0]}"
                rest = "; ".join(f"{h}: {v}" for h, v in zip(head[1:], row[1:]) if v)
                return [f"{first} - {rest}"]
            return [c.text]
        if c.kind == "table":
            return [f"{c.meta.get('caption','Table')}: " + "; ".join(" | ".join(r) for r in c.meta.get("rows", [])[:6])]
        if c.kind in ("figure", "connection"):
            return [c.text]
        txt = re.sub(r"(?m)^([A-Z][A-Z ]{3,})$\n", "", c.text)          # drop OCR section headers
        txt = re.sub(r"\n(?![-\d])", " ", txt)                          # re-flow OCR line wraps
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9-])|\n|(?=\s\d\.\s[A-Z])", txt)
        return [s.strip(" -") for s in parts if len(s.strip()) > 12]

    def compose(self, p: QueryPlan, ev: list[Evidence], ver: dict) -> dict:
        terms = p.content + [t for t in T.tokens(" ".join(p.ids))] + list(p.expansions)
        wts = {t: (1.0 if t not in p.expansions else 0.4) * self._idf(t) for t in terms}
        tot = sum(wts.values()) or 1.0
        cands = []
        for e in ev:
            for u in self._units(e):
                toks = set(T.tokens(u))
                if e.chunk.section == "Drawing overview" or e.score == 0:
                    continue  # graph / safety additions are cited in their own blocks
                # entity grounding: a line that names the asset asked about is direct evidence
                uids = {T.canon_id(x) for x in T.identifiers(u)}
                qids = set(p.known_ids)
                id_bonus = 0.5 if qids & uids else (0.25 if qids & self.index.id_sets[e.idx] else 0.0)
                cov = self._coverage(p.content, toks | set(T.tokens(e.chunk.section))) if p.content else 0.0
                if p.content and cov + id_bonus < 0.4:
                    continue  # a unit must address the question, not just name the asset
                s = sum(w for t, w in wts.items() if self._has(t, toks)) / tot + id_bonus * 0.5
                s += 0.12 / e.n + (0.05 if e.chunk.kind == "table_row" else 0)
                cands.append((s, e.n, u, e))
        cands.sort(key=lambda x: -x[0])
        limit = 5 if p.mode == "field" else 8
        picked, seen = [], []
        for s, n, u, e in cands:
            if s < 0.18 or len(picked) >= limit:
                continue
            tk = set(T.tokens(u))
            if any(len(tk & o) / max(1, len(tk | o)) > 0.6 for o in seen):
                continue
            seen.append(tk)
            picked.append({"text": u, "cites": [n], "score": round(s, 3), "doc_id": e.chunk.doc_id})
        blocks = []
        support = sorted({e.chunk.doc_id for e in ev[:5] if e.coverage >= 0.5})
        if p.intents["why"]:
            lead = "Likely causes, in order of supporting evidence" + (f" ({len(support)} documents agree):" if len(support) > 1 else ":")
        elif p.intents["procedure"]:
            lead = "Documented procedure:"
        else:
            lead = "From the documentation:"
        blocks.append({"type": "lead", "text": lead, "cites": []})
        steps = self._procedure(ev) if p.intents["procedure"] else self._findings_list(p, ev)
        if steps:
            blocks += steps
            picked = [x for x in picked if not any(x["text"][:40] in s["text"] for s in steps)][:2]
        blocks += [{"type": "bullet", "text": x["text"], "cites": x["cites"]} for x in picked]
        ctx = self._asset_context(p, ev)
        if ctx:
            blocks.append(ctx)
        if ver.get("conflict"):
            blocks.append({"type": "note", "text": "Conflicting values: " + ver["conflict"]["resolution"],
                           "cites": [v["n"] for v in ver["conflict"]["values"]]})
        saf = self._safety(p, ev)
        if saf:
            blocks.append(saf)
        return {"generator": "extractive", "blocks": blocks}

    def _graph_evidence(self, p: QueryPlan, ev: list[Evidence]) -> list[Evidence]:
        """Add the drawing connections that link a named asset to the models whose
        manuals were used, so the multi-document hop itself is cited."""
        if not p.known_ids or not ev:
            return []
        have = {e.chunk.id for e in ev}
        path_nodes = set()
        for v in p.kg.values():
            path_nodes |= set(v["path"])
        out = []
        for i, c in enumerate(self.chunks):
            if c.kind != "connection" or c.id in have:
                continue
            a = [T.canon_id(x) for x in T.identifiers(c.meta.get("from", ""))][:1]
            b = [T.canon_id(x) for x in T.identifiers(c.meta.get("to", ""))][:1]
            if a and b and a[0] in path_nodes and b[0] in path_nodes and (set(p.known_ids) & {a[0], b[0]} or len(out) < 2):
                out.append(Evidence(len(ev) + len(out) + 1, c, 0.0, 0.0, 0.0, [], 0.0, i))
            if len(out) >= 2:
                break
        for e in out:
            e.matched = ["knowledge-graph link"]
        return out

    def _asset_context(self, p: QueryPlan, ev: list[Evidence]) -> dict | None:
        """Explain how a named asset links to the documents used (knowledge-graph path)."""
        for q in p.known_ids:
            n = self.kg.nodes.get(q)
            if not n or n["type"] != "asset":
                continue
            models = [v for k, v in p.kg.items() if self.kg.nodes.get(k, {}).get("type") == "model"]
            if not models:
                continue
            chain = max(models, key=lambda v: len(v["path"]))
            labels = [self.kg.nodes[x]["label"] for x in chain["path"]]
            cites = sorted({e.n for e in ev if e.chunk.kind == "connection"})[:3] or sorted({e.n for e in ev if e.chunk.doc_id in n["docs"]})[:2]
            return {"type": "context", "text": f"Equipment chain for {n['label']}: " + " -> ".join(labels) +
                    ". Manuals for these models were searched as well.", "cites": cites}
        return None

    def _findings_list(self, p: QueryPlan, ev: list[Evidence]) -> list[dict]:
        """When the best passage is itself a numbered list (findings, checks), keep its order."""
        if not ev or ev[0].coverage < 0.5 or p.intents["value"] and not re.search(r"found|finding|list|check", p.question, re.I):
            return []
        items = re.findall(r"(?:^|\n)\s*(\d)\.\s(.+?)(?=\n\s*\d\.\s|$)", ev[0].chunk.text, re.S)
        if len(items) < 3:
            return []
        return [{"type": "step", "text": f"{n}. " + re.sub(r"\s+", " ", t).strip(), "cites": [ev[0].n]} for n, t in items]

    def _procedure(self, ev: list[Evidence]) -> list[dict]:
        for e in ev[:3]:
            c = e.chunk
            if c.kind == "text" and re.search(r"(^|\s)1\.\s", c.text):
                sec = [x for x in self.chunks if x.doc_id == c.doc_id and x.section == c.section and x.kind == "text"]
                steps = []
                for x in sec:
                    for m in re.finditer(r"(?:^|\s)(\d)\.\s(.+?)(?=\s\d\.\s[A-Z]|$)", x.text.replace("\n", " ")):
                        steps.append((int(m.group(1)), m.group(2).strip(), x))
                if len(steps) >= 3:
                    nmap = {ev_.chunk.id: ev_.n for ev_ in ev}
                    return [{"type": "step", "text": f"{i}. {s}", "cites": [nmap.get(x.id, e.n)], "chunk_id": x.id} for i, s, x in steps]
        return []

    LOTO_RE = re.compile(r"warning|lockout|loto", re.I)

    def _safety_evidence(self, p: QueryPlan, ev: list[Evidence]) -> list[Evidence]:
        """Make sure a lockout / warning statement from the documents in play is cited
        whenever the technician is about to act on equipment."""
        if not ev or not (p.intents["procedure"] or p.intents["why"] or p.intents["safety_critical"]):
            return []
        if any(e.chunk.safety and self.LOTO_RE.search(e.chunk.text) for e in ev):
            return []
        docs = [e.chunk.doc_id for e in ev if self.docs[e.chunk.doc_id].doc_type == "manual"]
        for d in dict.fromkeys(docs):
            cands = [(i, c) for i, c in enumerate(self.chunks) if c.doc_id == d and c.safety and self.LOTO_RE.search(c.text)]
            # prefer the warning inside the same section as the evidence, else the first one
            secs = {e.chunk.section for e in ev if e.chunk.doc_id == d}
            cands.sort(key=lambda ic: (ic[1].section not in secs, ic[1].page, ic[1].bbox[1]))
            if cands:
                i, c = cands[0]
                return [Evidence(len(ev) + 1, c, 0.0, 0.0, 0.0, ["safety"], 0.0, i)]
        return []

    def _safety(self, p: QueryPlan, ev: list[Evidence]) -> dict | None:
        if not (p.intents["procedure"] or p.intents["why"] or p.intents["safety_critical"]):
            return None
        for e in ev:
            if e.chunk.safety and self.LOTO_RE.search(e.chunk.text):
                s = next((u for u in self._units(e) if self.LOTO_RE.search(u)), None)
                if s:
                    return {"type": "safety", "text": s, "cites": [e.n]}
        return None

    # ------------------------------------------------------------------ public API
    def query(self, question: str, mode: str = "field", generator=None) -> dict:
        t0 = time.perf_counter()
        p = self.plan(question, mode)
        ev = self.retrieve(p, 7 if mode == "field" else 10)
        ev += self._graph_evidence(p, ev)
        ev += self._safety_evidence(p, ev)
        t1 = time.perf_counter()
        ver = self.evaluate(p, ev)
        answer = None
        if ver["band"] != "escalate":
            answer = self.compose(p, ev, ver)
            if generator is not None:
                try:
                    gen = generator(p, ev, ver, self)
                    if gen:
                        answer = gen
                except Exception as exc:  # generator failure never blocks a grounded answer
                    answer["note"] = f"LLM unavailable, extractive answer shown ({exc.__class__.__name__})."
        elif p.intents["safety_bypass"]:
            saf = next((e for e in ev if e.chunk.safety and re.search(r"never|do not", e.chunk.text, re.I)), None)
            answer = {"generator": "policy", "blocks": [
                {"type": "safety", "text": "Not permitted without authorization. " + (
                    next((u for u in self._units(saf) if re.search(r"never|do not", u, re.I)), "") if saf else ""),
                 "cites": [saf.n] if saf else []}]}
        t2 = time.perf_counter()
        return {
            "question": question, "mode": mode, **ver,
            "unknown_ids": p.unknown_ids, "known_ids": p.known_ids, "intents": p.intents,
            "kg_paths": [{"to": v["label"], "path": [self.kg.nodes[x]["label"] for x in v["path"] if x in self.kg.nodes], "weight": v["weight"]}
                         for k, v in sorted(p.kg.items(), key=lambda kv: -kv[1]["weight"])[:6]],
            "answer": answer,
            "evidence": [self.evidence_dict(e) for e in ev],
            "timings_ms": {"retrieval": round((t1 - t0) * 1000, 1), "total": round((t2 - t0) * 1000, 1)},
        }

    def evidence_dict(self, e: Evidence) -> dict:
        c, d = e.chunk, self.docs[e.chunk.doc_id]
        return {"n": e.n, "chunk_id": c.id, "doc_id": c.doc_id, "doc": d.short, "doc_type": d.doc_type, "page": c.page,
                "section": c.section, "kind": c.kind, "text": c.text, "bbox": c.bbox, "ocr_conf": c.ocr_conf,
                "scores": {"bm25": round(e.bm25, 3), "dense": round(e.dense, 3), "fused": round(e.score, 4)},
                "matched": e.matched, "coverage": round(e.coverage, 3)}
