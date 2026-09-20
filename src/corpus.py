
from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
CORPUS_PATH = Path(__file__).resolve().parent.parent / "data" / "corpus.json"

HEADING_RE = re.compile(r"^#\s*([A-Z]+-\d+)\s*[—-]\s*(.+?)\s*$")
DATE_FIELD_RE = re.compile(
    r"^(Last reviewed|Effective date|Updated|Last updated|Reviewed)\s*:\s*(.+)$",
    re.IGNORECASE,
)
STATUS_RE = re.compile(r"^STATUS:\s*(.+)$", re.IGNORECASE)


STALE_MARKERS = [
    "older version", "outdated", "obsolete", "no longer", "not be treated as",
    "has been removed", "is no longer offered", "should not be used",
    "retired", "archived documentation", "previous version", "previous mobile",
    "that wording is outdated", "this is no longer",
]


@dataclass
class Record:
    id: str
    title: str
    source_type: str
    text: str
    reviewed: Optional[str] = None
    status: Optional[str] = None
    has_deprecated_note: bool = False
    tags: Optional[list] = None


def _split_entries(raw: str) -> list[str]:
    """Split a source file into entry blocks on a line that is just '---'."""
    blocks = re.split(r"\n\s*-{3,}\s*\n", raw)
    return [b.strip() for b in blocks if b.strip()]


def _extract_tags(text: str) -> list[str]:
    """Very small keyword tagger used only as a secondary retrieval signal /
    human-readable metadata. Not required for TF-IDF search itself."""
    keywords = {
        "refund": "refund", "chargeback": "chargeback", "dispute": "chargeback",
        "subscription": "subscription", "certificate": "certificate",
        "password": "account-security", "login": "account-security",
        "captions": "accessibility", "accessib": "accessibility",
        "offline": "offline-mobile", "mobile": "offline-mobile",
        "progress": "progress-sync", "payment": "billing", "charge": "billing",
        "duplicate": "billing", "family": "account-sharing",
        "organization": "account-sharing", "email": "account-management",
        "browser": "technical", "video": "technical",
    }
    text_l = text.lower()
    found = sorted({tag for kw, tag in keywords.items() if kw in text_l})
    return found


def _locate_heading(lines: list[str]) -> tuple[Optional[re.Match], int]:
    """Find the first heading line in a block (skips a leading document-level
    title like '# LearnForge Policies' that precedes the first real entry)."""
    for i, line in enumerate(lines):
        m = HEADING_RE.match(line.strip())
        if m:
            return m, i
    return None, -1


def parse_faqs(raw: str) -> list[Record]:
    records = []
    for block in _split_entries(raw):
        lines = block.splitlines()
        m, idx = _locate_heading(lines)
        if not m:
            continue
        rid, title = m.group(1), m.group(2)
        body = "\n".join(lines[idx + 1:]).strip()
        records.append(Record(
            id=rid, title=title, source_type="faq", text=body,
            tags=_extract_tags(title + " " + body),
        ))
    return records


def parse_policies(raw: str) -> list[Record]:
    
    raw = re.sub(r"\n#?\s*SECTION\s+\d+\s*[—-].*$", "", raw, flags=re.IGNORECASE)
    records = []
    for block in _split_entries(raw):
        lines = block.splitlines()
        m, idx = _locate_heading(lines)
        if not m:
            continue
        rid, title = m.group(1), m.group(2)
        reviewed = None
        body_lines = []
        for line in lines[idx + 1:]:
            dm = DATE_FIELD_RE.match(line.strip())
            if dm:
                reviewed = dm.group(2).strip()
            else:
                body_lines.append(line)
        body = "\n".join(body_lines).strip()
        has_stale = any(marker in body.lower() for marker in STALE_MARKERS)
        records.append(Record(
            id=rid, title=title, source_type="policy", text=body,
            reviewed=reviewed, has_deprecated_note=has_stale,
            tags=_extract_tags(title + " " + body),
        ))
    return records


def parse_tickets(raw: str) -> list[Record]:
    records = []
    for block in _split_entries(raw):
        lines = block.splitlines()
        m, idx = _locate_heading(lines)
        if not m:
            continue
        rid, title = m.group(1), m.group(2)
        status = None
        body_lines = []
        for line in lines[idx + 1:]:
            sm = STATUS_RE.match(line.strip())
            if sm:
                status = sm.group(1).strip()
            else:
                body_lines.append(line)
        body = "\n".join(body_lines).strip()
        records.append(Record(
            id=rid, title=title, source_type="ticket", text=body,
            status=status, tags=_extract_tags(title + " " + body),
        ))
    return records


def build_corpus() -> list[Record]:
    faqs = parse_faqs((RAW_DIR / "faqs.md").read_text(encoding="utf-8"))
    policies = parse_policies((RAW_DIR / "policies.md").read_text(encoding="utf-8"))
    tickets = parse_tickets((RAW_DIR / "tickets.md").read_text(encoding="utf-8"))
    return policies + faqs + tickets  # authority order by convention


def save_corpus(records: list[Record], path: Path = CORPUS_PATH) -> None:
    path.write_text(
        json.dumps([asdict(r) for r in records], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_corpus(path: Path = CORPUS_PATH) -> list[Record]:
    if not path.exists():
        records = build_corpus()
        save_corpus(records, path)
        return records
    data = json.loads(path.read_text(encoding="utf-8"))
    return [Record(**d) for d in data]


if __name__ == "__main__":
    recs = build_corpus()
    save_corpus(recs)
    by_type: dict[str, int] = {}
    for r in recs:
        by_type[r.source_type] = by_type.get(r.source_type, 0) + 1
    print(f"Wrote {len(recs)} records to {CORPUS_PATH}")
    print(by_type)
