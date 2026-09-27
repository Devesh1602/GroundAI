"""Escalation queue and documentation-gap log.

SQLite by default (zero setup). Set DATABASE_URL=postgresql://... to use PostgreSQL
(requires psycopg). Every escalated question becomes a ticket routed to the next-level
handler; tickets are grouped into documentation gaps (by missing entity or topic) so
the knowledge base can be improved where technicians actually hit walls.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS escalations (
  id TEXT PRIMARY KEY,
  created_at TEXT NOT NULL,
  question TEXT NOT NULL,
  mode TEXT,
  reason TEXT,
  confidence REAL,
  topic TEXT,
  routed_to TEXT,
  status TEXT DEFAULT 'open',
  resolution TEXT,
  evidence TEXT
)"""


def topic_of(result: dict) -> str:
    if result.get("unknown_ids"):
        return result["unknown_ids"][0]
    if result.get("intents", {}).get("safety_bypass"):
        return "Protection bypass request"
    miss = result.get("missing_terms") or []
    if miss:
        return " ".join(miss[:2])
    words = re.findall(r"[A-Za-z]{4,}", result["question"])
    return " ".join(words[-2:]) or "general"


class GapLog:
    def __init__(self, path: Path | None = None):
        url = os.getenv("DATABASE_URL", "")
        self.pg = url.startswith("postgres")
        if self.pg:  # pragma: no cover - needs a running PostgreSQL
            import psycopg
            self.conn = psycopg.connect(url, autocommit=True)
            self.ph = "%s"
        else:
            path = Path(path or "data/cache/groundai.db")
            path.parent.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(str(path), check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            self.ph = "?"
        self.conn.execute(SCHEMA)
        if not self.pg:
            self.conn.commit()

    def _q(self, sql: str) -> str:
        return sql.replace("?", self.ph)

    def log(self, result: dict, routed_to: str) -> dict:
        tid = "ESC-" + uuid.uuid4().hex[:6].upper()
        row = {"id": tid, "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "question": result["question"],
               "mode": result.get("mode"), "reason": " ".join(result.get("reasons", [])), "confidence": result.get("confidence"),
               "topic": topic_of(result), "routed_to": routed_to, "status": "open", "resolution": None,
               "evidence": json.dumps([{"doc": e["doc"], "page": e["page"], "section": e["section"]} for e in result.get("evidence", [])[:3]])}
        self.conn.execute(self._q("INSERT INTO escalations VALUES (?,?,?,?,?,?,?,?,?,?,?)"), tuple(row.values()))
        if not self.pg:
            self.conn.commit()
        return row

    def list(self) -> list[dict]:
        cur = self.conn.execute("SELECT * FROM escalations ORDER BY created_at DESC")
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]

    def resolve(self, tid: str, resolution: str) -> None:
        self.conn.execute(self._q("UPDATE escalations SET status='resolved', resolution=? WHERE id=?"), (resolution, tid))
        if not self.pg:
            self.conn.commit()

    def gaps(self) -> list[dict]:
        groups: dict[str, dict] = {}
        for r in self.list():
            g = groups.setdefault(r["topic"], {"topic": r["topic"], "count": 0, "open": 0, "questions": [], "last_seen": r["created_at"]})
            g["count"] += 1
            g["open"] += r["status"] == "open"
            if len(g["questions"]) < 5:
                g["questions"].append(r["question"])
        return sorted(groups.values(), key=lambda g: (-g["open"], -g["count"]))
