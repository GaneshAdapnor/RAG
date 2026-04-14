from __future__ import annotations

from pathlib import Path

from PyPDF2 import PdfReader

from app.models.domain import ParseResult, ParsedPage
from app.utils.text import normalize_whitespace


class DocumentParser:
    def parse(self, file_path: str | Path) -> ParseResult:
        path = Path(file_path)
        suffix = path.suffix.lower()

        if suffix == ".pdf":
            return self._parse_pdf(path)
        if suffix == ".txt":
            return self._parse_txt(path)

        raise ValueError(f"Unsupported file type: {suffix}")

    def _parse_pdf(self, path: Path) -> ParseResult:
        reader = PdfReader(str(path))
        pages: list[ParsedPage] = []
        warnings: list[str] = []

        for index, page in enumerate(reader.pages, start=1):
            try:
                extracted_text = page.extract_text() or ""
            except Exception as exc:
                warnings.append(f"Page {index} could not be parsed: {exc}")
                continue

            normalized_text = normalize_whitespace(extracted_text)
            if not normalized_text:
                warnings.append(f"Page {index} contained no extractable text.")
                continue

            pages.append(ParsedPage(page_number=index, text=normalized_text))

        if not pages:
            raise ValueError("The PDF contained no extractable text.")

        return ParseResult(pages=pages, warnings=warnings)

    def _parse_txt(self, path: Path) -> ParseResult:
        raw_text = path.read_text(encoding="utf-8", errors="ignore")
        normalized_text = normalize_whitespace(raw_text)
        if not normalized_text:
            raise ValueError("The TXT file was empty after normalization.")

        return ParseResult(pages=[ParsedPage(page_number=1, text=normalized_text)])
