from pathlib import Path

import pytest

from app.document_loader import load_file_as_documents, load_text_as_documents


def test_load_text_as_documents_creates_plain_text_document() -> None:
    documents = load_text_as_documents(
        text="A short generic document.",
        source_id="source-1",
        filename="note.txt",
        doc_type="general",
    )

    assert len(documents) == 1
    assert documents[0].page_content == "A short generic document."
    assert documents[0].metadata["parser_used"] == "plain_text"
    assert "project_id" not in documents[0].metadata


def test_markdown_heading_sets_section_title(tmp_path: Path) -> None:
    path = tmp_path / "guide.md"
    path.write_text("# Overview\n\nBody text.\n\n## Details\n\nMore text.", encoding="utf-8")

    result = load_file_as_documents(str(path), "source-1", path.name, "guide")

    assert result.parser_used == "markdown_text"
    assert result.detected_extension == ".md"
    assert any(document.metadata.get("section_title") == "Overview" for document in result.documents)


def test_csv_empty_row_reports_warning(tmp_path: Path) -> None:
    path = tmp_path / "rows.csv"
    path.write_text("name,value\nalpha,1\n\nbeta,2\n", encoding="utf-8")

    result = load_file_as_documents(str(path), "source-1", path.name, "table")

    assert result.parser_used == "csv"
    assert result.warnings == ["CSV contained empty rows that were skipped."]
    assert any(document.metadata.get("row_number") == 2 for document in result.documents)
    assert all(document.metadata.get("section_title") == "CSV rows" for document in result.documents)


def test_unsupported_extension_raises_value_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "sample.bin"
    path.write_bytes(b"not a supported document")
    monkeypatch.setattr("app.document_loader._load_with_docling", lambda *args: [])

    with pytest.raises(ValueError, match="Unsupported file type"):
        load_file_as_documents(str(path), "source-1", path.name, "general")
