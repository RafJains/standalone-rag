from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app import auth


def patch_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        auth,
        "get_rag_settings",
        lambda: SimpleNamespace(
            rag_require_api_key=False,
            rag_api_key="test-key",
            rag_default_project_id="default",
            rag_project_id_pattern=r"^[A-Za-z0-9_-]+$",
        ),
    )


@pytest.mark.parametrize(
    ("raw_project_id", "expected"),
    [
        (None, "default"),
        ("default", "default"),
        ("alpha", "alpha"),
        ("client_1", "client_1"),
        ("client-1", "client-1"),
    ],
)
def test_project_id_valid_values(
    monkeypatch: pytest.MonkeyPatch,
    raw_project_id: str | None,
    expected: str,
) -> None:
    patch_settings(monkeypatch)

    assert auth.resolve_project_id(raw_project_id) == expected


@pytest.mark.parametrize("raw_project_id", ["", "two words", "client/1", "..", "client!"])
def test_project_id_rejects_unsafe_values(
    monkeypatch: pytest.MonkeyPatch,
    raw_project_id: str,
) -> None:
    patch_settings(monkeypatch)

    with pytest.raises(HTTPException) as exc_info:
        auth.resolve_project_id(raw_project_id)

    assert exc_info.value.status_code == 400
