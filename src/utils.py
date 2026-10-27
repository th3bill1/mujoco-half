"""Small shared utilities for checkpoint commands."""

import json
import platform
import sys
from pathlib import Path
from typing import Any


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def runtime_info() -> dict[str, str]:
    return {
        "python": sys.version,
        "os": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }
