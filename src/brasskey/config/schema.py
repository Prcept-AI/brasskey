# SPDX-License-Identifier: Apache-2.0
"""JSON Schemas for brasskey.yaml and brasskey.lock (exported to schemas/, drift-tested)."""

import json
from pathlib import Path
from typing import Any

from brasskey.config.intent import Intent
from brasskey.config.lock import Lock

SCHEMAS: dict[str, Any] = {
    "brasskey.yaml.schema.json": Intent,
    "brasskey.lock.schema.json": Lock,
}


def render_schemas() -> dict[str, str]:
    """File name -> JSON text, deterministic."""
    rendered: dict[str, str] = {}
    for name, model in SCHEMAS.items():
        schema = model.model_json_schema()
        schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
        rendered[name] = json.dumps(schema, indent=2, sort_keys=True) + "\n"
    return rendered


def export_json_schemas(out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, text in render_schemas().items():
        path = out_dir / name
        path.write_text(text, encoding="utf-8")
        written.append(path)
    return written
