"""OCR for scanned pages and raster drawings (Tesseract), with domain post-correction."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageOps

try:
    import pytesseract
except ImportError:  # pragma: no cover
    pytesseract = None

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


@dataclass
class OcrLine:
    text: str
    bbox: tuple[float, float, float, float]  # pixels x0,y0,x1,y1
    conf: float                              # 0..1
    block: int
    corrections: list[tuple[str, str]] = field(default_factory=list)


def available() -> bool:
    if pytesseract is None:
        return False
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def preprocess(img: Image.Image) -> Image.Image:
    """Grayscale, deskew (OpenCV minAreaRect on ink pixels) and light denoise."""
    g = ImageOps.grayscale(img)
    if cv2 is None:
        return g
    a = np.array(g)
    a = cv2.medianBlur(a, 3)
    thr = cv2.threshold(a, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    pts = np.column_stack(np.where(thr > 0))
    if len(pts) > 500:
        angle = cv2.minAreaRect(pts[:, ::-1].astype(np.float32))[-1]
        if angle > 45:
            angle -= 90
        if 0.2 < abs(angle) < 8:
            h, w = a.shape
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            a = cv2.warpAffine(a, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=255)
    return Image.fromarray(a)


# Characters Tesseract commonly confuses inside technical identifiers.
CONFUSIONS = {"!": "1", "O": "0", "Q": "0", "D": "0", "o": "0", "I": "1", "l": "1", "|": "1", "]": "1", "S": "5", "B": "8", "Z": "2"}


def correct_token(tok: str, lexicon: set[str]) -> str | None:
    """Map an OCR token onto a known domain identifier when a confusion swap fixes it.

    Only tokens that mix letters and digits are touched, and only when the result
    is an identifier that already exists in the corpus lexicon (e.g. FQ001 -> F0001).
    """
    core = tok.strip(".,;:()")
    if not core or core.upper() in lexicon:
        return None
    # only touch tokens that look like identifiers, never ordinary words ("All")
    if not re.search(r"\d", core) and not re.search(r"[\]|!]", core) and not core.isupper():
        return None
    # try swapping confusable characters (not the leading letter prefix)
    m = re.match(r"^([A-Za-z]{1,4})(-?)(.*)$", core)
    if not m:
        return None
    letters, dash, rest = m.groups()
    for k in range(len(letters), 0, -1):          # FQ001: try prefix "FQ", then "F"
        prefix, tail = letters[:k], letters[k:] + dash + rest
        fixed = prefix + "".join(CONFUSIONS.get(ch, ch) for ch in tail)
        if fixed != core and fixed.upper() in lexicon:
            return tok.replace(core, fixed)
    return None


def ocr_lines(img: Image.Image, lexicon: set[str] | None = None) -> list[OcrLine]:
    if not available():
        return []
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, config="--psm 3")
    lines: dict[tuple, dict] = {}
    for i, word in enumerate(data["text"]):
        w = (word or "").strip()
        conf = float(data["conf"][i])
        if not w or conf < 0:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        x, y, ww, hh = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        L = lines.setdefault(key, {"words": [], "confs": [], "box": [x, y, x + ww, y + hh]})
        L["words"].append(w)
        L["confs"].append(conf / 100.0)
        b = L["box"]
        b[0], b[1], b[2], b[3] = min(b[0], x), min(b[1], y), max(b[2], x + ww), max(b[3], y + hh)
    out = []
    for key in sorted(lines, key=lambda k: (lines[k]["box"][1], lines[k]["box"][0])):
        L = lines[key]
        words, corr = [], []
        for w in L["words"]:
            fixed = correct_token(w, lexicon) if lexicon else None
            if fixed:
                corr.append((w, fixed))
                words.append(fixed)
            else:
                words.append(w)
        out.append(OcrLine(" ".join(words), tuple(L["box"]), float(np.mean(L["confs"])), key[0], corr))
    return out


def ocr_region(img: Image.Image, box: tuple[int, int, int, int]) -> tuple[str, float]:
    if not available():
        return "", 0.0
    crop = img.crop(box)
    data = pytesseract.image_to_data(crop, output_type=pytesseract.Output.DICT, config="--psm 6")
    words, confs = [], []
    rows: dict[tuple, list] = {}
    for i, w in enumerate(data["text"]):
        w = (w or "").strip()
        c = float(data["conf"][i])
        if w and c >= 0:
            rows.setdefault((data["block_num"][i], data["par_num"][i], data["line_num"][i]), []).append(w)
            confs.append(c / 100)
    text = "\n".join(" ".join(v) for _, v in sorted(rows.items()))
    return text, float(np.mean(confs)) if confs else 0.0
