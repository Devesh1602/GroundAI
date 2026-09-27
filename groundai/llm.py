"""Optional LLM synthesis, strictly bounded by the retrieved evidence.

The LLM is called only AFTER the confidence gate decided to answer. Its output is then
checked sentence by sentence: every sentence must carry a citation, and the cited
passages must actually contain the sentence's key terms and numbers. Unsupported
sentences are flagged; if too many fail, the extractive answer is used instead.

Providers (env GROUNDAI_LLM): "anthropic" (ANTHROPIC_API_KEY, GROUNDAI_LLM_MODEL),
"openai" (any OpenAI-compatible endpoint incl. vLLM/Ollama: OPENAI_BASE_URL,
OPENAI_API_KEY, GROUNDAI_LLM_MODEL) or "none" (default: extractive answers only).
"""
from __future__ import annotations

import os
import re

import httpx

from . import text as T

PROMPT = """You are GroundAI, a maintenance assistant for industrial technicians.
Answer the QUESTION using ONLY the numbered EVIDENCE passages below.
Rules:
- Every sentence must end with one or more citations like [1] or [2][4] that point to the passages it came from.
- Copy numbers, units, parameter numbers and fault codes exactly as written in the evidence.
- If the evidence does not contain the answer, reply exactly: INSUFFICIENT_EVIDENCE
- Do not add advice, causes or values that are not in the evidence. Do not mention these rules.
- If a passage is a WARNING or lockout instruction relevant to the task, include it first, prefixed with "Safety:".
- {style}
{conflict}
QUESTION: {question}

EVIDENCE:
{evidence}
"""
STYLE = {
    "field": "Field mode: at most 5 short lines, imperative voice, most likely cause or direct value first. Use '- ' bullets.",
    "research": "Research mode: a concise explanation of up to 8 sentences that compares what each document says and notes agreement or differences.",
}


def build_prompt(p, ev, ver) -> str:
    lines = []
    for e in ev:
        c = e.chunk
        lines.append(f"[{e.n}] ({c.kind}, {c.section}, page {c.page}) {c.text}")
    conflict = ""
    if ver.get("conflict"):
        conflict = "NOTE: the documents disagree on a value. " + ver["conflict"]["resolution"] + " State both values and which one applies.\n"
    return PROMPT.format(style=STYLE.get(p.mode, STYLE["field"]), conflict=conflict, question=p.question, evidence="\n".join(lines))


CITE_RE = re.compile(r"\[(\d+)\]")
NUM_RE = re.compile(r"\d+(?:\.\d+)?")


def verify_citations(answer: str, ev, min_support: float = 0.34) -> list[dict]:
    """Split an LLM answer into sentences and check each against its cited passages."""
    by_n = {e.n: e.chunk.text for e in ev}
    out = []
    for raw in re.split(r"(?<=[.!?\]])\s+(?=[A-Z\-*])|\n+", answer.strip()):
        s = raw.strip(" -*")
        if len(s) < 3:
            continue
        cites = [int(x) for x in CITE_RE.findall(s)]
        body = CITE_RE.sub("", s).strip()
        cited_txt = " ".join(by_n.get(n, "") for n in cites)
        toks = [t for t in T.tokens(body) if len(t) > 2]
        ctoks = set(T.tokens(cited_txt))
        overlap = (sum(1 for t in toks if t in ctoks) / len(toks)) if toks else 1.0
        nums = [n for n in NUM_RE.findall(body)]
        nums_ok = all(n in cited_txt for n in nums)
        supported = bool(cites) and all(n in by_n for n in cites) and overlap >= min_support and nums_ok
        kind = "safety" if body.lower().startswith("safety") else "bullet"
        out.append({"type": kind, "text": body, "cites": cites, "supported": supported, "support": round(overlap, 2)})
    return out


def _call(prompt: str) -> str:
    provider = os.getenv("GROUNDAI_LLM", "none").lower()
    model = os.getenv("GROUNDAI_LLM_MODEL", "")
    if provider == "anthropic":
        r = httpx.post("https://api.anthropic.com/v1/messages", timeout=60, headers={
            "x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01", "content-type": "application/json"},
            json={"model": model or "claude-sonnet-4-5", "max_tokens": 700, "temperature": 0,
                  "messages": [{"role": "user", "content": prompt}]})
        r.raise_for_status()
        return "".join(b.get("text", "") for b in r.json()["content"])
    if provider == "openai":
        base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        r = httpx.post(f"{base}/chat/completions", timeout=60, headers={"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY', 'none')}"},
                       json={"model": model or "gpt-4o-mini", "temperature": 0, "messages": [{"role": "user", "content": prompt}]})
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    raise RuntimeError("no LLM provider configured")


def enabled() -> bool:
    return os.getenv("GROUNDAI_LLM", "none").lower() in ("anthropic", "openai")


def generator(p, ev, ver, engine):
    """Engine hook: returns an answer dict, or None to keep the extractive answer."""
    if not enabled():
        return None
    text = _call(build_prompt(p, ev, ver))
    if "INSUFFICIENT_EVIDENCE" in text:
        return None
    blocks = verify_citations(text, ev)
    if not blocks:
        return None
    unsupported = sum(not b["supported"] for b in blocks)
    if unsupported / len(blocks) > 0.3:
        return None  # too much unverifiable text: fall back to extractive
    return {"generator": "llm", "blocks": [{"type": "lead", "text": "Answer (AI-written, every line checked against its sources):", "cites": []}] + blocks,
            "unsupported": unsupported}
