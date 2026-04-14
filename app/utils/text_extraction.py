"""Compatibility text extraction helpers backed by the new parser."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

from app.services.document_parser import DocumentParser


@dataclass
class PageContent:
    text: str
    page: int | None


def extract_text(file_bytes: bytes, filename: str, content_type: str) -> list[PageContent]:
    suffix = Path(filename).suffix or (".pdf" if content_type == "application/pdf" else ".txt")
    parser = DocumentParser()

    with NamedTemporaryFile(delete=True, suffix=suffix) as temp_file:
        temp_file.write(file_bytes)
        temp_file.flush()
        result = parser.parse(temp_file.name)

    return [PageContent(text=page.text, page=page.page_number) for page in result.pages]


__all__ = ["PageContent", "extract_text"]
