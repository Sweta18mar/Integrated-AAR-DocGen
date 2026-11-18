"""Extraction, redaction, chunking and template parsing.

Maps to the Extraction Lambda (+ Textract), Redaction Analysis Lambda and
Chunker Lambda in the architecture.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

SUPPORTED = {".pdf", ".docx", ".txt", ".md", ".markdown", ".csv"}


class UnsupportedFile(Exception):
    pass


# ---------------------------------------------------------------- extraction
def extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        pages = [(page.extract_text() or "").strip() for page in reader.pages]
        text = "\n\n".join(f"[Page {i}]\n{t}" for i, t in enumerate(pages, 1) if t)
        if not text:
            raise UnsupportedFile(
                f"{path.name} has no text layer (likely a scan). OCR such as Amazon Textract is needed for it."
            )
        return text
    if suffix == ".docx":
        import docx

        document = docx.Document(str(path))
        parts = []
        for para in document.paragraphs:
            if not para.text.strip():
                continue
            level = para.style.name.removeprefix("Heading ") if para.style.name.startswith("Heading") else ""
            parts.append(f"{'#' * int(level)} {para.text}" if level.isdigit() else para.text)
        for table in document.tables:
            for row in table.rows:
                parts.append(" | ".join(cell.text.strip() for cell in row.cells))
        return "\n\n".join(parts)
    if suffix in SUPPORTED:
        return path.read_text(encoding="utf-8", errors="replace")
    raise UnsupportedFile(f"{path.name}: unsupported file type. Use PDF, DOCX, TXT, MD or CSV.")


# ---------------------------------------------------------------- redaction
def _luhn_ok(number: str) -> bool:
    digits = [int(d) for d in re.sub(r"\D", "", number)][::-1]
    total = sum(d if i % 2 == 0 else (d * 2 - 9 if d * 2 > 9 else d * 2) for i, d in enumerate(digits))
    return 13 <= len(digits) <= 19 and total % 10 == 0


_PATTERNS = [
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("CARD", re.compile(r"\b(?:\d[ -]?){12,18}\d\b")),
    ("PHONE", re.compile(r"(?<!\w)(?:\+\d{1,3}[ .-]?)?(?:\(\d{2,4}\)[ .-]?|\d{2,4}[ .-])\d{3,4}[ .-]\d{3,4}(?!\w)")),
    ("IP", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
]


def redact(text: str) -> tuple[str, dict[str, int]]:
    """Replace personal data with typed placeholders. Returns the text and counts per type."""
    counts: dict[str, int] = {}
    for label, pattern in _PATTERNS:
        def sub(match: re.Match, label: str = label) -> str:
            if label == "CARD" and not _luhn_ok(match.group()):
                return match.group()
            counts[label] = counts.get(label, 0) + 1
            return f"[{label}]"

        text = pattern.sub(sub, text)
    return text, counts


# ---------------------------------------------------------------- chunking
def chunk(text: str, size: int = 1200, overlap: int = 150) -> list[str]:
    """Split on paragraph boundaries into chunks of about `size` characters."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        while len(para) > size:  # hard-split very long paragraphs
            head, para = para[:size], para[size - overlap :]
            if current:
                chunks.append(current)
                current = ""
            chunks.append(head)
        if current and len(current) + len(para) + 2 > size:
            chunks.append(current)
            current = current[-overlap:] + "\n\n" + para if overlap else para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current:
        chunks.append(current)
    return chunks


# ---------------------------------------------------------------- templates
@dataclass
class Section:
    heading: str
    guidance: str = ""


def parse_template(text: str) -> list[Section]:
    """Markdown headings become sections; the text under each heading becomes its guidance.

    A file without headings is read as one section per non-empty line.
    """
    sections: list[Section] = []
    for line in text.splitlines():
        match = re.match(r"^\s*#{1,6}\s+(.+?)\s*#*\s*$", line)
        if match:
            sections.append(Section(match.group(1).strip()))
        elif sections and line.strip():
            sections[-1].guidance = (sections[-1].guidance + "\n" + line.strip()).strip()
    if not sections:
        sections = [Section(line.strip(" -*\t")) for line in text.splitlines() if line.strip(" -*\t")]
    # A single top-level title followed by sub-headings: drop the title.
    if len(sections) > 1 and not sections[0].guidance and text.lstrip().startswith("# ") and "\n## " in text:
        sections = sections[1:]
    return sections
