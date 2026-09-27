from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class Chunk:
    id: str
    doc_id: str
    page: int                      # 1-based
    kind: str                      # text | table | table_row | figure | connection | ocr_text
    section: str
    text: str                      # what the technician sees
    ctx: str                       # what gets indexed (text + contextual header)
    bbox: list[float]              # [x0, y0, x1, y1] as page fractions (top-left origin)
    ocr_conf: Optional[float] = None
    safety: bool = False
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Document:
    id: str
    title: str
    short: str
    filename: str
    doc_type: str                  # manual | bulletin | drawing | scanned_report | upload
    revision: str = ""
    date: str = ""
    pages: int = 0
    page_sizes: list[list[float]] = field(default_factory=list)
    subjects: list[str] = field(default_factory=list)
    supersedes: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)
