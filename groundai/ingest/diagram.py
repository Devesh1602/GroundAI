"""Engineering-diagram understanding with classical computer vision.

Given a raster page (single-line diagram, block/wiring diagram) we:
  1. detect component boxes  (closed rectangular contours, OpenCV)
  2. OCR the text inside each box                     -> component nodes
  3. detect connection segments outside the boxes    (morphological line extraction)
  4. link boxes whose borders touch a connected path -> edges
  5. read the free-standing label nearest each path  -> edge labels (cable, signal)

The result feeds the knowledge graph (asset -> model, asset -> asset wiring) and a
text rendering of the drawing that becomes a retrievable 'figure' chunk.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from . import ocr

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


@dataclass
class Box:
    id: int
    bbox: tuple[int, int, int, int]
    text: str = ""
    conf: float = 0.0

    @property
    def title(self) -> str:
        return self.text.split("\n")[0].strip() if self.text else f"box{self.id}"


@dataclass
class Diagram:
    boxes: list[Box] = field(default_factory=list)
    edges: list[tuple[int, int, str]] = field(default_factory=list)
    size: tuple[int, int] = (0, 0)

    def is_diagram(self) -> bool:
        return len(self.boxes) >= 3 and len(self.edges) >= 2


def _find_boxes(gray: np.ndarray) -> list[tuple[int, int, int, int]]:
    h, w = gray.shape
    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    contours, _ = cv2.findContours(thr, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        if bw < w * 0.12 or bh < h * 0.03 or bw > w * 0.8 or bh > h * 0.3:
            continue
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(approx) != 4:
            continue
        # rectangle whose contour area ~ bbox area
        if cv2.contourArea(c) < 0.85 * bw * bh:
            continue
        boxes.append((x, y, x + bw, y + bh))
    # de-duplicate inner/outer border contours
    boxes.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    kept: list[tuple[int, int, int, int]] = []
    for b in boxes:
        if any(abs(b[0] - k[0]) < 15 and abs(b[1] - k[1]) < 15 and abs(b[2] - k[2]) < 15 and abs(b[3] - k[3]) < 15 for k in kept):
            continue
        kept.append(b)
    # drop 'phantom' rectangles formed by two parallel wires running between two real
    # boxes: such a rectangle shares its left/right edges with neighbouring boxes.
    def shares_edge(b, others):
        left = right = False
        for o in others:
            if o is b:
                continue
            v_overlap = min(b[3], o[3]) - max(b[1], o[1]) > 0
            left |= v_overlap and abs(b[0] - o[2]) < 12
            right |= v_overlap and abs(b[2] - o[0]) < 12
        return left and right
    return [b for b in kept if not shares_edge(b, kept)]


def _line_mask(gray: np.ndarray, boxes) -> np.ndarray:
    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    h, w = gray.shape
    horiz = cv2.morphologyEx(thr, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(25, w // 40), 1)))
    vert = cv2.morphologyEx(thr, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(25, h // 60))))
    mask = cv2.bitwise_or(horiz, vert)
    # remove the box borders themselves (keep a 6px ring outside so we can detect touching)
    for (x0, y0, x1, y1) in boxes:
        mask[y0 - 4:y1 + 5, x0 - 4:x1 + 5] = 0
    # remove the sheet frame / title block (long lines near the page border)
    mask[:, :60] = 0; mask[:, -60:] = 0; mask[:60, :] = 0; mask[-60:, :] = 0
    return mask


def parse(img: Image.Image) -> Diagram:
    d = Diagram(size=img.size)
    if cv2 is None:
        return d
    gray = np.array(img.convert("L"))
    rects = _find_boxes(gray)
    for i, r in enumerate(rects):
        pad = 6
        text, conf = ocr.ocr_region(img, (r[0] + pad, r[1] + pad, r[2] - pad, r[3] - pad))
        d.boxes.append(Box(i, r, text.strip(), conf))
    d.boxes = [b for b in d.boxes if b.text]
    if len(d.boxes) < 3:
        return d
    mask = _line_mask(gray, [b.bbox for b in d.boxes])
    n, labels = cv2.connectedComponents(mask, connectivity=8)
    # which component touches which box (sample a ring of pixels around each box)
    touches: dict[int, set[int]] = {}
    for b in d.boxes:
        x0, y0, x1, y1 = b.bbox
        ring = np.zeros_like(mask)
        cv2.rectangle(ring, (x0 - 10, y0 - 10), (x1 + 10, y1 + 10), 255, 8)
        comp_ids = set(np.unique(labels[(ring > 0) & (mask > 0)])) - {0}
        for c in comp_ids:
            touches.setdefault(int(c), set()).add(b.id)
    seen = set()
    for comp, bids in touches.items():
        bl = sorted(bids)
        if len(bl) < 2:
            continue
        ys, xs = np.where(labels == comp)
        cx, cy = float(xs.mean()), float(ys.mean())
        for a in range(len(bl)):
            for c in range(a + 1, len(bl)):
                d.edges.append((bl[a], bl[c], f"{cx:.0f},{cy:.0f}"))
    return d


def attach_labels(d: Diagram, free_lines: list[ocr.OcrLine]) -> list[tuple[str, str, str]]:
    """Resolve edges to (from_title, to_title, nearest label text)."""
    by_id = {b.id: b for b in d.boxes}
    inside = lambda L, b: L.bbox[0] >= b.bbox[0] - 5 and L.bbox[2] <= b.bbox[2] + 5 and L.bbox[1] >= b.bbox[1] - 5 and L.bbox[3] <= b.bbox[3] + 5
    free = [L for L in free_lines if not any(inside(L, b) for b in d.boxes)]
    out = []
    used = set()
    for a, c, centre in d.edges:
        cx, cy = map(float, centre.split(","))
        best, bd = "", 1e9
        for i, L in enumerate(free):
            lx, ly = (L.bbox[0] + L.bbox[2]) / 2, (L.bbox[1] + L.bbox[3]) / 2
            dist = ((lx - cx) ** 2 + (ly - cy) ** 2) ** 0.5
            if dist < bd and dist < 260 and len(L.text) < 40:
                best, bd = L.text, dist
        if best:
            used.add(best)
        out.append((by_id[a].title, by_id[c].title, best))
    return out
