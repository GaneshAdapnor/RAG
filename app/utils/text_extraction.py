"""
Text extraction utilities for PDF and plain-text files.

Design decisions:
- PyPDF2 is used for PDFs because it is pure-Python (no system-level poppler
  dependency), making it portable across environments. The trade-off is that
  scanned/image-only PDFs yield empty text — we detect and warn rather than
  silently produce bad results.
- We extract text page-by-page (not the whole document at once) so that we
  can attach page numbers to chunks, enabling source attribution in answers.
- TXT files are read with explicit UTF-8 decoding and a latin-1 fallback to
  handle the most common encodings without crashing on non-ASCII content.
"""

import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import PyPDF2

logger = logging.getLogger(__name__)


@dataclass
class PageContent:
    """Text content from a single page/section of a document."""

    text: str
    page: Optional[int]  # 1-based; None for TXT files


def extract_text_from_pdf(file_bytes: bytes, filename: str) -> List[PageContent]:
    """
    Extract text from a PDF file, returning one PageContent per page.

    Handles:
    - Encrypted PDFs (attempts empty-password decryption)
    - Pages with no extractable text (image-only pages)
    - Malformed PDFs (logs error and returns whatever was extracted)

    Args:
        file_bytes: Raw PDF bytes from the upload.
        filename: Used only for logging context.

    Returns:
        List of PageContent, one per PDF page. Pages with no text are omitted.
    """
    pages: List[PageContent] = []

    try:
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))

        # Handle password-protected PDFs with empty password (common for read-lock)
        if reader.is_encrypted:
            try:
                reader.decrypt("")
                logger.info("PDF '%s' was encrypted; decrypted with empty password.", filename)
            except Exception:
                logger.error(
                    "PDF '%s' is encrypted and could not be decrypted. Skipping.", filename
                )
                return pages

        total_pages = len(reader.pages)
        logger.info("Extracting text from PDF '%s' (%d pages).", filename, total_pages)

        empty_page_count = 0
        for page_num, page in enumerate(reader.pages, start=1):
            try:
                raw_text = page.extract_text() or ""
                # PyPDF2 may return whitespace-only strings for image pages
                cleaned = raw_text.strip()
                if not cleaned:
                    empty_page_count += 1
                    logger.debug(
                        "PDF '%s' page %d/%d: no extractable text (may be image-only).",
                        filename, page_num, total_pages,
                    )
                    continue
                pages.append(PageContent(text=cleaned, page=page_num))
            except Exception as exc:
                logger.warning(
                    "Failed to extract text from page %d of '%s': %s. Skipping.",
                    page_num, filename, exc,
                )

        if empty_page_count > 0:
            logger.warning(
                "PDF '%s': %d/%d pages had no extractable text.",
                filename, empty_page_count, total_pages,
            )

        if not pages:
            logger.error(
                "PDF '%s': No text could be extracted. "
                "The file may be a scanned document requiring OCR.",
                filename,
            )

    except PyPDF2.errors.PdfReadError as exc:
        logger.error("Malformed PDF '%s': %s", filename, exc)
    except Exception as exc:
        logger.error("Unexpected error reading PDF '%s': %s", filename, exc)

    return pages


def extract_text_from_txt(file_bytes: bytes, filename: str) -> List[PageContent]:
    """
    Extract text from a plain-text file.

    Encoding strategy:
    1. Try UTF-8 (standard)
    2. Fall back to latin-1 (covers ISO-8859-1, common in legacy docs)

    Returns a single PageContent with page=None since TXT files have no pages.
    """
    text: Optional[str] = None

    for encoding in ("utf-8", "latin-1"):
        try:
            text = file_bytes.decode(encoding)
            logger.debug("Decoded '%s' as %s.", filename, encoding)
            break
        except UnicodeDecodeError:
            continue

    if text is None:
        logger.error("Could not decode '%s' with any supported encoding.", filename)
        return []

    cleaned = text.strip()
    if not cleaned:
        logger.warning("TXT file '%s' is empty after stripping whitespace.", filename)
        return []

    logger.info("Extracted %d characters from TXT '%s'.", len(cleaned), filename)
    return [PageContent(text=cleaned, page=None)]


def extract_text(file_bytes: bytes, filename: str, content_type: str) -> List[PageContent]:
    """
    Route extraction to the correct handler based on content type or extension.

    Args:
        file_bytes: Raw file bytes.
        filename: Original filename (used for extension fallback).
        content_type: MIME type from the HTTP upload.

    Returns:
        List of PageContent objects ready for chunking.

    Raises:
        ValueError: If the file type is not supported.
    """
    ext = Path(filename).suffix.lower()

    is_pdf = content_type in ("application/pdf",) or ext == ".pdf"
    is_txt = content_type in ("text/plain",) or ext == ".txt"

    if is_pdf:
        return extract_text_from_pdf(file_bytes, filename)
    elif is_txt:
        return extract_text_from_txt(file_bytes, filename)
    else:
        raise ValueError(
            f"Unsupported file type '{content_type}' (extension: '{ext}'). "
            "Only PDF and TXT files are accepted."
        )
