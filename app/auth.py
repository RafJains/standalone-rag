import hmac
import re

from fastapi import Header, HTTPException

from app.config import get_rag_settings


def require_api_key(x_rag_api_key: str | None = Header(default=None)) -> None:
    settings = get_rag_settings()
    if not settings.rag_require_api_key:
        return

    if not x_rag_api_key:
        raise HTTPException(status_code=401, detail="API key required")

    if not hmac.compare_digest(x_rag_api_key, settings.rag_api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")


def resolve_project_id(project_id: str | None) -> str:
    settings = get_rag_settings()
    resolved = (project_id or settings.rag_default_project_id).strip()
    if not resolved:
        raise HTTPException(status_code=400, detail="project_id cannot be empty")
    if not re.fullmatch(settings.rag_project_id_pattern, resolved):
        raise HTTPException(
            status_code=400,
            detail="project_id may contain only letters, numbers, dash, and underscore",
        )
    return resolved
