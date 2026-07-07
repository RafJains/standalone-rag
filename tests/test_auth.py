from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app import auth


def settings(require_key: bool, api_key: str = "test-key") -> SimpleNamespace:
    return SimpleNamespace(
        rag_require_api_key=require_key,
        rag_api_key=api_key,
        rag_default_project_id="default",
        rag_project_id_pattern=r"^[A-Za-z0-9_-]+$",
    )


def test_api_key_dependency_allows_missing_key_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(auth, "get_rag_settings", lambda: settings(False))

    assert auth.require_api_key(None) is None


@pytest.mark.parametrize("provided_key", [None, "wrong-key"])
def test_api_key_dependency_rejects_missing_or_wrong_key(
    monkeypatch: pytest.MonkeyPatch,
    provided_key: str | None,
) -> None:
    monkeypatch.setattr(auth, "get_rag_settings", lambda: settings(True))

    with pytest.raises(HTTPException) as exc_info:
        auth.require_api_key(provided_key)

    assert exc_info.value.status_code == 401


def test_api_key_dependency_accepts_correct_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(auth, "get_rag_settings", lambda: settings(True, "expected-key"))

    assert auth.require_api_key("expected-key") is None
