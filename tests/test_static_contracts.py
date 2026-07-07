from pathlib import Path

from app.config import RagSettings
from app.schemas import RagStatusResponse
from scripts.validate_stack import compose_service_names


ROOT = Path(__file__).resolve().parents[1]


def test_compose_services_remain_limited() -> None:
    assert compose_service_names(ROOT / "docker-compose.yml") == {"weaviate", "rag-api"}


def test_status_schema_and_settings_expose_workflow() -> None:
    settings = RagSettings()
    status = RagStatusResponse(
        service="rohan-rag-api",
        rag_enabled=settings.rag_enabled,
        workflow=settings.rag_workflow,
        vector_db=settings.vector_db,
        embedding_model=settings.embedding_model,
        weaviate_url=settings.weaviate_url,
        weaviate_collection=settings.weaviate_collection,
        weaviate_reachable=True,
        message="ok",
    )

    assert settings.rag_workflow == "langgraph"
    assert status.workflow == "langgraph"


def test_requirements_keep_expected_workflow_and_avoid_retired_package() -> None:
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
    retired = "lang" + "chain"

    assert "langgraph" in requirements
    assert retired not in requirements


def test_dockerfile_copies_runtime_sources() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "COPY app/ ./app/" in dockerfile
    assert "COPY scripts/ ./scripts/" in dockerfile


def test_frontend_files_exist() -> None:
    assert (ROOT / "app/static/index.html").is_file()
    assert (ROOT / "app/static/styles.css").is_file()
    assert (ROOT / "app/static/app.js").is_file()
