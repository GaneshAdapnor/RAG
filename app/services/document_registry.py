from __future__ import annotations

import json
from pathlib import Path
from threading import RLock

from app.models.domain import DocumentRecord, DocumentStatus, utc_now
from app.utils.files import atomic_write_json


class DocumentRegistry:
    def __init__(self, storage_path: Path) -> None:
        self.storage_path = storage_path
        self._lock = RLock()
        self._documents = self._load()

    def _load(self) -> dict[str, DocumentRecord]:
        if not self.storage_path.exists():
            return {}

        raw_payload = self.storage_path.read_text(encoding="utf-8")
        if not raw_payload.strip():
            return {}

        return {
            record.document_id: record
            for record in (DocumentRecord.model_validate(item) for item in json.loads(raw_payload))
        }

    def _persist(self) -> None:
        payload = [
            record.model_dump(mode="json")
            for record in sorted(self._documents.values(), key=lambda item: item.created_at)
        ]
        atomic_write_json(self.storage_path, payload)

    def create_document(
        self,
        document_id: str,
        filename: str,
        file_path: str,
        media_type: str,
    ) -> DocumentRecord:
        with self._lock:
            record = DocumentRecord(
                document_id=document_id,
                filename=filename,
                file_path=file_path,
                media_type=media_type,
                status=DocumentStatus.pending,
            )
            self._documents[document_id] = record
            self._persist()
            return record

    def get_document(self, document_id: str) -> DocumentRecord | None:
        with self._lock:
            record = self._documents.get(document_id)
            return None if record is None else record.model_copy(deep=True)

    def list_documents(self, status: DocumentStatus | None = None) -> list[DocumentRecord]:
        with self._lock:
            records = list(self._documents.values())
            if status is not None:
                records = [record for record in records if record.status == status]
            return [record.model_copy(deep=True) for record in records]

    def update_status(self, document_id: str, status: DocumentStatus) -> None:
        with self._lock:
            record = self._documents[document_id]
            self._documents[document_id] = record.model_copy(
                update={"status": status, "updated_at": utc_now()}
            )
            self._persist()

    def mark_ready(
        self,
        document_id: str,
        pages_extracted: int,
        chunks_created: int,
        warnings: list[str] | None = None,
    ) -> None:
        with self._lock:
            record = self._documents[document_id]
            self._documents[document_id] = record.model_copy(
                update={
                    "status": DocumentStatus.ready,
                    "pages_extracted": pages_extracted,
                    "chunks_created": chunks_created,
                    "warnings": warnings or [],
                    "error": None,
                    "updated_at": utc_now(),
                }
            )
            self._persist()

    def mark_failed(self, document_id: str, error: str) -> None:
        with self._lock:
            record = self._documents[document_id]
            self._documents[document_id] = record.model_copy(
                update={
                    "status": DocumentStatus.failed,
                    "error": error,
                    "updated_at": utc_now(),
                }
            )
            self._persist()

    def counts_by_status(self) -> dict[str, int]:
        counts = {status.value: 0 for status in DocumentStatus}
        with self._lock:
            for record in self._documents.values():
                counts[record.status.value] += 1
        return counts
