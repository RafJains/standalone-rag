from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = (
    "lang" + "chain",
    "live" + "kit",
    "ho" + "tel",
    "grand" + "view",
    "gu" + "est",
    "ro" + "om",
    "book" + "ing",
    "check" + "out",
    "check" + "-in",
    "gem" + "ini",
    "deep" + "gram",
    "token" + "-server",
    "agent" + "_server",
    "ho" + "tel" + "_policy",
    "ho" + "tel" + "knowledge",
    "qd" + "rant",
    "mil" + "vus",
    "qw" + "en",
    "ll" + "ama" + "index",
)
ALLOWED_MODEL_TERM_PATTERNS = (
    re.compile(r"\b" + "ll" + "ama" + r"\b.*\bforbidden\b", re.IGNORECASE),
    re.compile(r"\bforbidden\b.*\b" + "ll" + "ama" + r"\b", re.IGNORECASE),
)
SKIP_DIRS = {"__pycache__", ".pytest_cache", ".venv", "venv", ".git"}
SKIP_PATHS = {
    ("data", "uploads"),
    ("data", "demo_documents"),
}


def iter_files() -> list[Path]:
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        rel_parts = path.relative_to(ROOT).parts
        if any(rel_parts[: len(skip_path)] == skip_path for skip_path in SKIP_PATHS):
            continue
        if path.is_file():
            files.append(path)
    return files


def line_has_allowed_model_term(line: str) -> bool:
    return any(pattern.search(line) for pattern in ALLOWED_MODEL_TERM_PATTERNS)


def main() -> int:
    failures: list[str] = []
    for path in iter_files():
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue
        rel_path = path.relative_to(ROOT)
        for line_number, line in enumerate(lines, start=1):
            lowered = line.lower()
            for term in FORBIDDEN:
                if term in lowered:
                    failures.append(f"{rel_path}:{line_number}: forbidden term '{term}'")
            model_term = "ll" + "ama"
            if model_term in lowered and not line_has_allowed_model_term(line):
                failures.append(f"{rel_path}:{line_number}: forbidden term '{model_term}'")

    if failures:
        print("RAG stack validation failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("RAG stack validation passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
