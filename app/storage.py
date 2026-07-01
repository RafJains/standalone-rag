from __future__ import annotations

import hashlib
import re
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
DEMO_DOCUMENTS_DIR = DATA_DIR / "demo_documents"


def ensure_data_directories() -> None:
    for directory in (UPLOADS_DIR, DEMO_DOCUMENTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def safe_filename(filename: str | None, default: str) -> str:
    raw_name = Path((filename or "").replace("\\", "/")).name.strip()
    if not raw_name or raw_name in {".", ".."}:
        raw_name = default

    sanitized = re.sub(r"[^A-Za-z0-9._ -]+", "_", raw_name)
    sanitized = sanitized.strip(" .")
    if not sanitized:
        sanitized = default

    suffix = Path(raw_name).suffix
    if suffix and not Path(sanitized).suffix:
        sanitized = f"{sanitized}{suffix}"
    return sanitized


def save_upload_bytes(source_id: str, filename: str, content: bytes) -> Path:
    ensure_data_directories()
    target_dir = UPLOADS_DIR / source_id
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / filename
    target_path.write_bytes(content)
    return target_path


def save_text(source_id: str, filename: str, text: str) -> Path:
    return save_upload_bytes(source_id, filename, text.encode("utf-8"))


def relative_storage_path(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def resolve_stored_path(stored_file_path: str | None) -> Path | None:
    if not stored_file_path:
        return None

    candidate = (PROJECT_ROOT / stored_file_path).resolve()
    uploads_root = UPLOADS_DIR.resolve()
    try:
        candidate.relative_to(uploads_root)
    except ValueError:
        return None
    return candidate


def delete_stored_file(stored_file_path: str | None) -> str | None:
    path = resolve_stored_path(stored_file_path)
    if path is None or not path.exists() or not path.is_file():
        return None

    deleted_relative_path = relative_storage_path(path)
    path.unlink()
    _remove_empty_source_dir(path.parent)
    return deleted_relative_path


def delete_source_folder_if_empty(source_id: str) -> None:
    _remove_empty_source_dir(UPLOADS_DIR / source_id)


def document_hash_from_bytes(content: bytes, doc_type: str) -> str:
    digest = hashlib.sha256()
    digest.update(doc_type.strip().lower().encode("utf-8"))
    digest.update(b"\0")
    digest.update(content)
    return digest.hexdigest()


def document_hash_from_text(text: str, filename: str, doc_type: str) -> str:
    digest = hashlib.sha256()
    digest.update(normalize_text(text).encode("utf-8"))
    digest.update(b"\0")
    digest.update(filename.strip().lower().encode("utf-8"))
    digest.update(b"\0")
    digest.update(doc_type.strip().lower().encode("utf-8"))
    return digest.hexdigest()


def content_hash_from_text(text: str, filename: str, doc_type: str, chunk_index: int) -> str:
    digest = hashlib.sha256()
    digest.update(normalize_text(text).encode("utf-8"))
    digest.update(b"\0")
    digest.update(filename.strip().lower().encode("utf-8"))
    digest.update(b"\0")
    digest.update(doc_type.strip().lower().encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(chunk_index).encode("utf-8"))
    return digest.hexdigest()


def normalize_text(text: str) -> str:
    return " ".join(text.split())


def copy_demo_document(source: Path, target_name: str | None = None) -> Path:
    ensure_data_directories()
    filename = safe_filename(target_name or source.name, "demo-document")
    target = DEMO_DOCUMENTS_DIR / filename
    shutil.copy2(source, target)
    return target


def _remove_empty_source_dir(path: Path) -> None:
    try:
        path.rmdir()
    except OSError:
        return
