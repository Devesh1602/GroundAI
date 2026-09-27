"""Equipment knowledge graph built from ingested evidence.

Nodes  : assets (P-104, VFD-104), models (CP-300, HX-500-055), fault codes, parameters,
         terminals, part numbers, components (impeller, bearing ...) and documents.
Edges  : connected_to  - wiring / mechanical links read from drawings (computer vision)
         has_model     - asset -> equipment model (drawing boxes, "P-104 (CP-300 pump)")
         variant_of    - HX-500-055 -> HX-500 family
         documents     - document -> the models / assets it is about
         references    - fault code <-> parameter in the same table row / procedure
         co_occurs     - mentioned together in one evidence chunk
The graph drives multi-document reasoning: a question about "P-104" is expanded to the
pump model, its motor and drive, and therefore to the manuals that cover them.
Optional export to Neo4j: scripts/export_neo4j.py.
"""
from __future__ import annotations

import re
from collections import defaultdict

from . import text as T
from .models import Chunk, Document

COMPONENTS = ["impeller", "bearing", "mechanical seal", "cooling fan", "suction strainer", "coupling", "motor cable",
              "dc link", "capacitor", "heatsink", "air filter", "casing", "shaft", "discharge valve", "pressure transmitter"]


def id_type(i: str) -> str:
    c = T.canon_id(i)
    if re.fullmatch(r"F\d{4}", c): return "fault"
    if re.fullmatch(r"\d{2}\.\d{2}", c): return "parameter"
    if re.fullmatch(r"(SR|SB)\d+", c): return "document_ref"
    if re.fullmatch(r"(AI|AO|DI|DO|RO)\d", c): return "terminal"
    if re.fullmatch(r"KP\d+[A-Z0-9]*", c): return "part"
    if re.fullmatch(r"(P|M|VFD|PT|Q|MCC|PLC|C|G|FT|LT|TT|K|E)\d{1,4}", c): return "asset"
    if re.fullmatch(r"[A-Z]{2}\d{2,3}(\d{3})?", c): return "model"
    return "other"


def keep_id(i: str) -> bool:
    c = T.canon_id(i)
    if len(c) <= 2: return False
    if re.fullmatch(r"M\d{1,2}", c): return False        # thread sizes M8..M20
    if re.fullmatch(r"(I2N|X\d)", c): return False
    return True


class KnowledgeGraph:
    def __init__(self):
        self.nodes: dict[str, dict] = {}
        self.edges: dict[tuple[str, str, str], dict] = {}

    # ---------------------------------------------------------------- build
    def node(self, key: str, label: str, typ: str) -> dict:
        n = self.nodes.get(key)
        if n is None:
            n = self.nodes[key] = {"id": key, "label": label, "type": typ, "mentions": [], "docs": []}
        return n

    def edge(self, a: str, b: str, typ: str, label: str = "", w: float = 1.0):
        if a == b:
            return
        if typ in ("co_occurs", "connected_to", "references") and a > b:
            a, b = b, a
        e = self.edges.get((a, b, typ))
        if e is None:
            self.edges[(a, b, typ)] = {"s": a, "t": b, "type": typ, "label": label, "w": w}
        else:
            e["w"] += w
            if label and not e["label"]:
                e["label"] = label

    def build(self, docs: list[Document], chunks: list[Chunk]) -> "KnowledgeGraph":
        for d in docs:
            dn = self.node("DOC:" + d.id, d.short, "document")
            dn["docs"] = [d.id]
            for s in d.subjects:
                if keep_id(s):
                    self.node(T.canon_id(s), s, id_type(s))
                    self.edge("DOC:" + d.id, T.canon_id(s), "documents")
        for c in chunks:
            ids = [i for i in T.identifiers(c.text) if keep_id(i)]
            low = c.text.lower()
            comps = [k for k in COMPONENTS if k in low]
            keys = []
            for i in ids:
                n = self.node(T.canon_id(i), i, id_type(i))
                keys.append(n["id"])
            for k in comps:
                n = self.node("C:" + k, k, "component")
                keys.append(n["id"])
            for k in keys:
                n = self.nodes[k]
                if c.id not in n["mentions"]:
                    n["mentions"].append(c.id)
                if c.doc_id not in n["docs"]:
                    n["docs"].append(c.doc_id)
            if len(keys) <= 10:
                for a in range(len(keys)):
                    for b in range(a + 1, len(keys)):
                        self.edge(keys[a], keys[b], "co_occurs")
            # typed relations --------------------------------------------------
            if c.kind == "figure" and c.meta.get("box"):
                box_ids = [T.canon_id(i) for i in ids]
                if box_ids and id_type(box_ids[0]) == "asset":
                    for other in box_ids[1:]:
                        if id_type(other) == "model":
                            self.edge(box_ids[0], other, "has_model", "drawing")
            if c.kind == "connection":
                a = [i for i in T.identifiers(c.meta.get("from", "")) if keep_id(i)]
                b = [i for i in T.identifiers(c.meta.get("to", "")) if keep_id(i)]
                if a and b:
                    self.edge(T.canon_id(a[0]), T.canon_id(b[0]), "connected_to", c.meta.get("label", ""))
            for m in re.finditer(r"\b([A-Z]{1,4}-\d{1,4})\s*\(([A-Z]{2}-\d{3}(?:-\d{3})?)", c.text):
                self.edge(T.canon_id(m.group(1)), T.canon_id(m.group(2)), "has_model", "text")
            if c.kind in ("table_row", "text", "ocr_text"):
                faults = [k for k in keys if self.nodes[k]["type"] == "fault"]
                params = [k for k in keys if self.nodes[k]["type"] == "parameter"]
                for f in faults:
                    for p in params:
                        self.edge(f, p, "references")
        # model families: HX-500-055 variant_of HX-500
        for k, n in list(self.nodes.items()):
            if n["type"] == "model":
                m = re.fullmatch(r"([A-Z]{2}\d{3})(\d{3})", k)
                if m and m.group(1) in self.nodes:
                    self.edge(k, m.group(1), "variant_of")
        return self

    # ---------------------------------------------------------------- query
    COST = {"connected_to": 0.75, "has_model": 0.4, "variant_of": 0.2, "documents": 0.9}

    def neighbours(self, key: str):
        for (a, b, typ), e in self.edges.items():
            if typ not in self.COST:
                continue
            if a == key:
                yield b, typ, e
            elif b == key:
                yield a, typ, e

    def expand(self, keys: list[str], max_cost: float = 2.2) -> dict[str, dict]:
        """Weighted neighbourhood of the query entities (Dijkstra over typed edges)."""
        best: dict[str, dict] = {}
        frontier = [(0.0, k, [k]) for k in keys if k in self.nodes]
        seen = {k: 0.0 for _, k, _ in frontier}
        while frontier:
            frontier.sort()
            cost, k, path = frontier.pop(0)
            for nb, typ, e in self.neighbours(k):
                nc = cost + self.COST[typ]
                if nc > max_cost or nb in keys or seen.get(nb, 99) <= nc:
                    continue
                seen[nb] = nc
                if not nb.startswith("DOC:"):
                    best[nb] = {"weight": round(max(0.2, 1 - nc / (max_cost + 0.5)), 3), "path": path + [nb],
                                "label": self.nodes[nb]["label"]}
                frontier.append((nc, nb, path + [nb]))
        return best

    def export(self) -> dict:
        return {"nodes": list(self.nodes.values()), "edges": list(self.edges.values())}
