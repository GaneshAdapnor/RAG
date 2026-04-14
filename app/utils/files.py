from __future__ import annotations

import json
import os
import re
from pathlib import Path

from fastapi import HTTPException, UploadFile, status


def sanitize_filename(filename: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", filename).strip("._")
    return cleaned or "document"


def is_supported_extension(filename: str, supported_extensions: list[str]) -> bool:
    return Path(filename).suffix.lower() in supported_extensions


async def save_upload_file(
    upload_file: UploadFile,
    destination: Path,
    max_upload_bytes: int,
) -> int:
    bytes_written = 0

    try:
        with destination.open("wb") as output_file:
            while True:
                chunk = await upload_file.read(1024 * 1024)
                if not chunk:
                    break
                bytes_written += len(chunk)
                if bytes_written > max_upload_bytes:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"File exceeds the maximum size of {max_upload_bytes} bytes.",
                    )
                output_file.write(chunk)
    except Exception:
        if destination.exists():
            destination.unlink()
        raise
    finally:
        await upload_file.close()

    return bytes_written


def atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temp_path, path)
