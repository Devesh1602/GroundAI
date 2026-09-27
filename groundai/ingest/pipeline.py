"""Document ingestion: PDF / image -> structured, citable chunks.

Born-digital pages   : pdfplumber words + fonts -> headings, paragraphs, procedures;
                       ruled tables -> one chunk per table + one chunk per row.
Image-only pages     : OpenCV deskew + Tesseract OCR (line boxes + confidences),
                       domain-lexicon post-correction (FQ001 -> F0001).
Drawings             : OpenCV box/line detection -> components and connections.
Every chunk keeps page number, section and a bounding box so the UI can highlight
the exact evidence on the page image.
"""
from __future__ import annotations

import hashlib
import io
import re
from pathlib import Path

import pdfplumber
import pypdfium2 as pdfium
from PIL import Image

from .. import text as T
from ..models import Chunk, Document
from . import diagram as dg
from . import ocr

OCR_DPI = 200
VIEW_WIDTH = 900  # px width of page images served to the evidence viewer


def _doc_id(path: Path) -> str:
    return hashlib.sha1(path.name.encode()).hexdigest()[:8]


def _frac(b, w, h):
    return [round(max(0, b[0]) / w, 4), round(max(0, b[1]) / h, 4), round(min(w, b[2]) / w, 4), round(min(h, b[3]) / h, 4)]


def _union(bs):
    return [min(b[0] for b in bs), min(b[1] for b in bs), max(b[2] for b in bs), max(b[3] for b in bs)]


SAFETY_RE = re.compile(r"\b(warning|danger|dangerous|lockout|loto|isolate|never bypass|do not)\b", re.I)
HEADING_RE = re.compile(r"^(\d+(\.\d+)*)\s+\S")


class Ingestor:
    def __init__(self, lexicon: set[str] | None = None):
        self.lexicon = set(lexicon or [])

    # ------------------------------------------------------------------ public
    def ingest(self, path: Path) -> tuple[Document, list[Chunk], list[bytes]]:
        path = Path(path)
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
            return self._ingest_image(path)
        return self._ingest_pdf(path)

    # ------------------------------------------------------------------ pdf
    def _ingest_pdf(self, path: Path):
        did = _doc_id(path)
        chunks: list[Chunk] = []
        images: list[bytes] = []
        stats = {"ocr_pages": 0, "tables": 0, "table_rows": 0, "figures": 0, "ocr_corrections": [], "digital_pages": 0}
        pdf = pdfium.PdfDocument(str(path))
        meta_title = (pdf.get_metadata_dict().get("Title") or "").strip()
        page_sizes, first_text, kinds = [], "", set()
        with pdfplumber.open(str(path)) as plumb:
            for pno, page in enumerate(plumb.pages, start=1):
                w, h = float(page.width), float(page.height)
                page_sizes.append([w, h])
                pimg = pdf[pno - 1].render(scale=VIEW_WIDTH / w).to_pil().convert("RGB")
                buf = io.BytesIO(); pimg.save(buf, "JPEG", quality=68, optimize=True); images.append(buf.getvalue())
                if len(page.chars) > 40:
                    stats["digital_pages"] += 1
                    txt = page.extract_text() or ""
                    if pno == 1:
                        first_text = txt
                    chunks += self._digital_page(did, pno, page, stats)
                else:
                    stats["ocr_pages"] += 1
                    big = pdf[pno - 1].render(scale=OCR_DPI / 72).to_pil().convert("RGB")
                    cks, kind, txt = self._raster_page(did, pno, big, stats)
                    kinds.add(kind)
                    if pno == 1:
                        first_text = txt
                    chunks += cks
        title = meta_title or path.stem.replace("_", " ")
        doc = self._make_doc(did, path, title, first_text, len(page_sizes), page_sizes, kinds, stats)
        self._contextualise(doc, chunks)
        stats["chunks"] = len(chunks)
        return doc, chunks, images

    def _ingest_image(self, path: Path):
        did = _doc_id(path)
        img = Image.open(path).convert("RGB")
        stats = {"ocr_pages": 1, "tables": 0, "table_rows": 0, "figures": 0, "ocr_corrections": [], "digital_pages": 0}
        cks, kind, txt = self._raster_page(did, 1, img, stats)
        view = img.copy(); view.thumbnail((VIEW_WIDTH, 4000))
        buf = io.BytesIO(); view.save(buf, "JPEG", quality=68)
        doc = self._make_doc(did, path, path.stem.replace("_", " "), txt, 1, [[img.width, img.height]], {kind}, stats)
        self._contextualise(doc, cks)
        stats["chunks"] = len(cks)
        return doc, cks, [buf.getvalue()]

    # ------------------------------------------------------------------ helpers
    def _make_doc(self, did, path, title, first_text, n, sizes, kinds, stats) -> Document:
        blob = title + "\n" + first_text
        date = ""
        m = re.search(r"(20\d\d-\d\d(?:-\d\d)?)", blob)
        if m:
            date = m.group(1)
        rev = ""
        m = re.search(r"\bRev(?:ision)?\.?\s+([A-Z0-9]{1,3})\b", title) or re.search(r"\bRevision\s+([A-Z0-9]{1,3})\b", first_text)
        if m:
            rev = m.group(1)
        if "drawing" in kinds:
            dtype = "drawing"
        elif "scan" in kinds:
            dtype = "scanned_report"
        elif re.search(r"bulletin", title, re.I):
            dtype = "bulletin"
        else:
            dtype = "manual"
        subjects = [i for i in T.identifiers(title) if not re.fullmatch(r"\d{2}\.\d{2}", i)]
        supersedes = []
        for m in re.finditer(r"supersedes[^.]*?(Section\s+[\d.]+)?[^.]*?([A-Z]{2,4}\d?-[A-Z]{2}-\d{3})", first_text, re.I):
            supersedes.append(m.group(2))
        short = re.split(r"\s+-\s+", title)[0]
        if len(short) > 46:
            short = short[:44].rstrip() + "..."
        return Document(did, title, short, Path(path).name, dtype, rev, date, n, sizes, subjects, supersedes, stats)

    def _contextualise(self, doc: Document, chunks: list[Chunk]):
        """Contextual chunk headers: prepend doc + section so short rows stay findable."""
        for c in chunks:
            c.ctx = f"{doc.short} | {c.section} | {c.ctx or c.text}"
            for i in T.identifiers(c.text):
                self.lexicon.add(i)

    # ------------------------------------------------------------------ digital
    def _digital_page(self, did, pno, page, stats) -> list[Chunk]:
        w, h = float(page.width), float(page.height)
        out: list[Chunk] = []
        tables = page.find_tables()
        tboxes = [t.bbox for t in tables]
        words_all = page.extract_words()
        # captions sit just below a table: "Table 5-1 Fault codes."
        for ti, t in enumerate(tables):
            rows = [[(c or "").replace("\n", " ").strip() for c in r] for r in t.extract()]
            if not rows:
                continue
            x0, top, x1, bottom = t.bbox
            cap_words = [wd for wd in words_all if bottom < wd["top"] < bottom + 22]
            caption = " ".join(wd["text"] for wd in sorted(cap_words, key=lambda z: z["x0"])).strip()
            if not caption.lower().startswith("table"):
                caption = f"Table (page {pno})"
            header = rows[0]
            stats["tables"] += 1
            section = self._section_at(page, top) or caption
            md = caption + "\n" + " | ".join(header) + "\n" + "\n".join(" | ".join(r) for r in rows[1:])
            out.append(Chunk(f"{did}-p{pno}-t{ti}", did, pno, "table", section, md, md, _frac(t.bbox, w, h),
                             meta={"caption": caption, "header": header, "rows": rows[1:]}))
            trows = getattr(t, "rows", [])
            for ri, r in enumerate(rows[1:], start=1):
                cells = "; ".join(f"{header[k]}: {v}" for k, v in enumerate(r) if k < len(header) and v)
                rb = trows[ri].bbox if ri < len(trows) and getattr(trows[ri], "bbox", None) else t.bbox
                txt = f"{caption.rstrip('.')} - {cells}"
                stats["table_rows"] += 1
                out.append(Chunk(f"{did}-p{pno}-t{ti}r{ri}", did, pno, "table_row", section, txt, txt, _frac(rb, w, h),
                                 safety=bool(SAFETY_RE.search(txt)), meta={"caption": caption, "header": header, "row": r}))
        # running text outside tables
        def outside(obj):
            cx, cy = (obj["x0"] + obj["x1"]) / 2, (obj["top"] + obj["bottom"]) / 2
            return not any(b[0] - 22 <= cx <= b[2] + 2 and b[1] - 2 <= cy <= b[3] + 24 for b in tboxes)
        fp = page.filter(lambda o: o.get("object_type") != "char" or outside(o))
        lines = [L for L in fp.extract_text_lines(layout=False, strip=True) if L["text"].strip()]
        lines = [L for L in lines if L["top"] < h - 40]  # drop running footer
        blocks: list[dict] = []
        cur = None
        section = ""
        for L in lines:
            ch = L["chars"][0] if L.get("chars") else {}
            is_head = "Bold" in ch.get("fontname", "") and ch.get("size", 0) >= 11
            if is_head:
                section = L["text"].strip()
                if cur:
                    blocks.append(cur)
                cur = None
                if ch.get("size", 0) >= 14:
                    section = ""  # document title
                    blocks.append({"lines": [L], "section": "Title", "head": True})
                continue
            gap = (L["top"] - cur["lines"][-1]["bottom"]) if cur else 99
            size = sum(len(x["text"]) for x in cur["lines"]) if cur else 0
            steps = bool(cur) and re.match(r"^\d+\.\s", L["text"]) and re.match(r"^\d+\.\s", cur["lines"][-1]["text"])
            if cur is None or size > 650 or (gap > 4.5 and not steps):
                if cur:
                    blocks.append(cur)
                cur = {"lines": [L], "section": section}
            else:
                cur["lines"].append(L)
        if cur:
            blocks.append(cur)
        # merge tiny blocks into the next one within the same section
        merged: list[dict] = []
        for b in blocks:
            txt = " ".join(x["text"] for x in b["lines"])
            if merged and merged[-1]["section"] == b["section"] and len(merged[-1]["txt"]) < 160 and not b.get("head") and not merged[-1].get("head"):
                merged[-1]["lines"] += b["lines"]; merged[-1]["txt"] += "\n" + txt
            else:
                merged.append({**b, "txt": txt})
        for bi, b in enumerate(merged):
            bb = _union([(x["x0"], x["top"], x["x1"], x["bottom"]) for x in b["lines"]])
            txt = b["txt"].strip()
            out.append(Chunk(f"{did}-p{pno}-b{bi}", did, pno, "text", b["section"] or "Introduction", txt, txt,
                             _frac(bb, w, h), safety=bool(SAFETY_RE.search(txt))))
        return out

    @staticmethod
    def _section_at(page, y) -> str:
        best = ""
        for L in page.extract_text_lines(layout=False, strip=True):
            ch = L["chars"][0] if L.get("chars") else {}
            if L["top"] < y and "Bold" in ch.get("fontname", "") and 11 <= ch.get("size", 0) < 14:
                best = L["text"].strip()
        return best

    # ------------------------------------------------------------------ raster
    def _raster_page(self, did, pno, img: Image.Image, stats):
        W, H = img.size
        pre = ocr.preprocess(img)
        lines = ocr.ocr_lines(pre, self.lexicon)
        for L in lines:
            stats["ocr_corrections"] += [f"{a}->{b}" for a, b in L.corrections]
        full = "\n".join(L.text for L in lines)
        diag = dg.parse(pre)
        out: list[Chunk] = []
        if diag.is_diagram():
            stats["figures"] += 1
            masked = pre.copy()
            from PIL import ImageDraw
            dr = ImageDraw.Draw(masked)
            for b in diag.boxes:
                dr.rectangle([b.bbox[0] - 8, b.bbox[1] - 8, b.bbox[2] + 8, b.bbox[3] + 8], fill=255)
            free_lines = ocr.ocr_lines(masked, self.lexicon)
            conns = dg.attach_labels(diag, free_lines)
            full = "\n".join([b.text for b in diag.boxes] + [L.text for L in free_lines])
            comps = []
            for b in diag.boxes:
                comp_txt = b.text.replace("\n", ", ")
                comps.append(comp_txt)
                out.append(Chunk(f"{did}-p{pno}-c{b.id}", did, pno, "figure", "Drawing component",
                                 f"Component {comp_txt}", f"Component {comp_txt}", _frac(b.bbox, W, H),
                                 ocr_conf=round(b.conf, 3), meta={"box": b.title}))
            byt = {b.title: b for b in diag.boxes}
            edges_meta = []
            for k, (a, c, label) in enumerate(conns):
                A, C = byt[a], byt[c]
                vertical = not (A.bbox[2] < C.bbox[0] or C.bbox[2] < A.bbox[0])
                if vertical:
                    up, down = (A, C) if A.bbox[1] < C.bbox[1] else (C, A)
                    sent = f"{up.text.replace(chr(10), ' ')} feeds / is connected to {down.text.replace(chr(10), ' ')}"
                else:
                    up, down = A, C
                    sent = f"{A.text.replace(chr(10), ' ')} is wired to {C.text.replace(chr(10), ' ')}"
                if label:
                    sent += f" via {label}"
                edges_meta.append({"from": up.title, "to": down.title, "label": label})
                bb = _union([A.bbox, C.bbox])
                out.append(Chunk(f"{did}-p{pno}-e{k}", did, pno, "connection", "Drawing connections", sent + ".", sent + ".",
                                 _frac(bb, W, H), ocr_conf=round(min(A.conf, C.conf), 3), meta={"from": up.title, "to": down.title, "label": label}))
            summary = "Drawing components: " + "; ".join(comps) + ". Connections: " + "; ".join(
                f"{e['from']} - {e['to']}" + (f" ({e['label']})" if e["label"] else "") for e in edges_meta) + "."
            out.append(Chunk(f"{did}-p{pno}-fig", did, pno, "figure", "Drawing overview", summary, summary, [0, 0, 1, 1],
                             ocr_conf=round(sum(b.conf for b in diag.boxes) / len(diag.boxes), 3),
                             meta={"components": [b.title for b in diag.boxes], "edges": edges_meta}))
            # free text (notes, title) - OCR'd with the component boxes masked out
            used_labels = {e["label"] for e in edges_meta}
            free = [L for L in free_lines if L.text not in used_labels]
            out += self._group_lines(did, pno, free, W, H, "Drawing notes")
            kind = "drawing"
        else:
            out += self._group_lines(did, pno, lines, W, H, "Scanned page")
            kind = "scan"
        return out, kind, full

    def _group_lines(self, did, pno, lines, W, H, default_section) -> list[Chunk]:
        """Group OCR lines into sections by ALL-CAPS headings / large vertical gaps."""
        out, cur, section = [], [], default_section
        def flush():
            if not cur:
                return
            txt = "\n".join(L.text for L in cur)
            conf = sum(L.conf for L in cur) / len(cur)
            out.append(Chunk(f"{did}-p{pno}-o{len(out)}", did, pno, "ocr_text", section, txt, txt,
                             _frac(_union([L.bbox for L in cur]), W, H), ocr_conf=round(conf, 3),
                             safety=bool(SAFETY_RE.search(txt))))
        prev = None
        for L in lines:
            words = L.text.split()
            heading = L.text.isupper() and len(words) <= 4 and len(L.text) > 3 and not re.search(r"\d", L.text)
            big_gap = prev is not None and (L.bbox[1] - prev.bbox[3]) > 70
            if heading or big_gap or sum(len(x.text) for x in cur) > 600:
                flush(); cur = []
                if heading:
                    section = L.text.title()
            cur.append(L)
            prev = L
        flush()
        return out
