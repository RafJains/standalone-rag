from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RagDocument:
    page_content: str
    metadata: dict[str, Any] = field(default_factory=dict)
