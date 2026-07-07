from __future__ import annotations

import glob
import subprocess
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class Check:
    name: str
    command: list[str]
    expected_stdout: list[str] | None = None


def run_check(check: Check) -> bool:
    print(f"RUN: {check.name}")
    result = subprocess.run(check.command, text=True, capture_output=True)
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    if result.returncode != 0:
        print(f"FAIL: {check.name}")
        return False
    if check.expected_stdout is not None:
        actual = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if actual != check.expected_stdout:
            print(f"FAIL: {check.name} expected {check.expected_stdout}, got {actual}")
            return False
    print(f"PASS: {check.name}")
    return True


def py_compile_command() -> list[str]:
    files = sorted(
        glob.glob("app/*.py")
        + glob.glob("scripts/*.py")
        + glob.glob("tests/*.py")
    )
    return [sys.executable, "-m", "py_compile", *files]


def main() -> int:
    checks = [
        Check("stack validation", [sys.executable, "scripts/validate_stack.py"]),
        Check("python compilation", py_compile_command()),
        Check("pytest", [sys.executable, "-m", "pytest", "-q"]),
        Check(
            "compose services",
            ["docker", "compose", "config", "--services"],
            expected_stdout=["weaviate", "rag-api"],
        ),
    ]

    passed = True
    for check in checks:
        passed = run_check(check) and passed

    if passed:
        print("quality checks passed")
        return 0
    print("quality checks failed", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
