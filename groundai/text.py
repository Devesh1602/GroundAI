"""Text normalisation shared by every index.

The web client (web/engine.js) implements the *same* rules so that an index
built here can be queried in the browser. Keep the two in sync; the parity test
in tests/test_parity.py checks them against each other.
"""
from __future__ import annotations

import re
from typing import Iterable

TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-.:][a-z0-9]+)*")
# Identifiers: asset tags, model codes, fault codes, parameters, document refs.
IDENT_RE = re.compile(r"(?<![A-Za-z0-9])(?:[A-Za-z]{1,4}-?\d{1,4}[A-Za-z]?(?:-\d{2,4}[A-Za-z]?)?|\d{2}\.\d{2})(?![A-Za-z0-9])")

STOPWORDS = set("""
a an the and or but if then else of to in on at by for with from into onto over under about as is are was were be been
being do does did doing have has had having it its this that these those there here what which who whom whose why how
when where can could should would will shall may might must i me my we our you your he she they them their not no yes
so than too very just also any all each per via up down out off again more most some such only own same both few other
please tell show give find need want know get does s t during
""".split())

# Words that carry no retrieval signal on their own in a maintenance question.
QUESTION_NOISE = set("problem issue issu help explain mean doe keep new".split())

SYNONYM_GROUPS = [
    ["trip", "fault", "alarm", "shutdown", "tripping"],
    ["overcurrent", "overload", "over-current"],
    ["torque", "tighten", "tightening"],
    ["grease", "lubrication", "lubricate", "re-grease", "regrease", "lubricant"],
    ["interval", "frequency", "often", "schedule"],
    ["replace", "replacement", "change", "swap", "renew"],
    ["acceleration", "accel", "ramp"],
    ["temperature", "overtemperature", "hot", "heat", "overheating"],
    ["drive", "vfd", "inverter"],
    ["blocked", "blockage", "clogged", "obstruction", "debris", "jammed", "rags"],
    ["breaker", "mccb", "feeds", "feeder"],
    ["megger", "insulation"],
    ["seal", "leak", "leakage"],
    ["spec", "specification", "rating", "rated", "value"],
    ["terminal", "input", "connection", "wired"],
    ["fan", "cooling"],
    ["found", "findings", "finding"],
]


def _undouble(w: str) -> str:
    """tripp -> trip, runn -> run (but keep 'll', 'ss', 'zz')."""
    if len(w) > 3 and w[-1] == w[-2] and w[-1] not in "lsz" and w[-1].isalpha():
        return w[:-1]
    return w


def stem(w: str) -> str:
    """Tiny deterministic suffix stripper (mirrored in JS)."""
    if not w.isalpha() or len(w) <= 3:
        return w
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("sses"):
        return w[:-2]
    if w.endswith("ing") and len(w) > 5:
        return _undouble(w[:-3])
    if w.endswith("ed") and len(w) > 4:
        return _undouble(w[:-2])
    if w.endswith("s") and not w.endswith("ss") and not w.endswith("us"):
        w = w[:-1]
    if w.endswith("e") and len(w) > 4:
        return w[:-1]            # grease/greased -> greas, replace/replaced -> replac
    return w


def normalize(text: str) -> str:
    text = text.replace("–", "-").replace("—", "-").replace("°", " deg ")
    return text.lower()


def tokens(text: str, keep_stop: bool = False) -> list[str]:
    out: list[str] = []
    for m in TOKEN_RE.finditer(normalize(text)):
        t = m.group(0).strip(".:-")
        if not t:
            continue
        if re.search(r"[-.:]", t):
            out.append(t)
            for p in re.split(r"[-.:]", t):
                if p and (keep_stop or p not in STOPWORDS):
                    out.append(stem(p))
        else:
            if not keep_stop and t in STOPWORDS:
                continue
            out.append(stem(t))
    return out


def canon_id(s: str) -> str:
    """Canonical identifier: upper-case, hyphens removed (XR-9 == xr9)."""
    return s.upper().replace("-", "")


def identifiers(text: str) -> list[str]:
    seen, out = set(), []
    for m in IDENT_RE.finditer(text):
        raw = m.group(0)
        if raw.isdigit():
            continue
        # require at least one letter+digit mix, or a dotted parameter number
        if not ("." in raw or (re.search(r"[A-Za-z]", raw) and re.search(r"\d", raw))):
            continue
        # ignore unit-like tokens: 50mm, 45kW, 400V, 10s, 2000h ...
        if re.fullmatch(r"\d+(?:mm|kw|v|a|s|h|nm|bar|hz|rpm|m|g|mm2|kv|ma)", raw.lower()):
            continue
        c = canon_id(raw)
        if c not in seen:
            seen.add(c)
            out.append(raw.upper())
    return out


_SYN_INDEX: dict[str, list[str]] | None = None


def synonym_index() -> dict[str, list[str]]:
    global _SYN_INDEX
    if _SYN_INDEX is None:
        idx: dict[str, list[str]] = {}
        for g in SYNONYM_GROUPS:
            stems = []
            for w in g:
                tk = tokens(w)
                t = tk[0] if tk else None      # full form only ("over-current", not "current")
                if t and t not in stems:
                    stems.append(t)
            for s in stems:
                idx.setdefault(s, [])
                for o in stems:
                    if o != s and o not in idx[s]:
                        idx[s].append(o)
        _SYN_INDEX = idx
    return _SYN_INDEX


def expand(terms: Iterable[str]) -> list[str]:
    idx = synonym_index()
    base = list(terms)
    out: list[str] = []
    for t in base:
        for s in idx.get(t, []):
            if s not in base and s not in out:
                out.append(s)
    return out


SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(])|\n+")


def sentences(text: str) -> list[str]:
    return [s.strip() for s in SENT_SPLIT.split(text) if len(s.strip()) > 2]
