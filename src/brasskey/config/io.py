# SPDX-License-Identifier: Apache-2.0
"""Read and write brasskey.yaml / brasskey.lock with plain-language errors."""

import hashlib
import logging
from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel, ValidationError

from brasskey.config.intent import INTENT_FILE, Intent
from brasskey.config.lock import LOCK_FILE, Lock
from brasskey.errors import BrasskeyError

logger = logging.getLogger(__name__)

SCHEMA_BASE = "https://github.com/Prcept-AI/brasskey/blob/main/schemas"
INTENT_HEADER = f"""\
# Brasskey intent: what you want to run. `brasskey run` in this folder uses it.
# Reference: {SCHEMA_BASE}/brasskey.yaml.schema.json
"""
LOCK_HEADER = f"""\
# Brasskey lock: exactly what ran (pinned files, engine, settings, measurements).
# Written by Brasskey; commit it to run the same thing elsewhere. Do not edit by hand.
# Reference: {SCHEMA_BASE}/brasskey.lock.schema.json
"""

ModelT = TypeVar("ModelT", bound=BaseModel)


class ConfigFileError(BrasskeyError):
    default_code = "BK-E-CONFIG"


def _describe(error: ValidationError, limit: int = 3) -> str:
    parts: list[str] = []
    for item in error.errors()[:limit]:
        location = ".".join(str(part) for part in item["loc"]) or "(top level)"
        parts.append(f"{location}: {item['msg']}")
    more = len(error.errors()) - limit
    return "; ".join(parts) + (f"; and {more} more" if more > 0 else "")


def _load(path: Path, model: type[ModelT], what: str) -> ModelT:
    try:
        data: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ConfigFileError(
            f"Couldn't read {path.name}: {error}.",
            fix=f"Fix the {what} file, or delete it to start fresh.",
        ) from error
    try:
        return model.model_validate(data or {})
    except ValidationError as error:
        raise ConfigFileError(
            f"{path.name} is not valid: {_describe(error)}.",
            fix=f"Fix the {what} file, or delete it to start fresh.",
        ) from error


def _dump(path: Path, header: str, model: BaseModel) -> Path:
    document = model.model_dump(mode="json", exclude_none=False)
    text = header + yaml.safe_dump(document, sort_keys=False, width=100, allow_unicode=True)
    path.write_text(text, encoding="utf-8")
    logger.debug("Wrote %s", path)
    return path


def read_intent(folder: Path) -> Intent | None:
    path = folder / INTENT_FILE
    return _load(path, Intent, "intent") if path.is_file() else None


def write_intent(folder: Path, intent: Intent) -> Path:
    return _dump(folder / INTENT_FILE, INTENT_HEADER, intent)


def read_lock(folder: Path) -> Lock | None:
    path = folder / LOCK_FILE
    return _load(path, Lock, "lock") if path.is_file() else None


def write_lock(folder: Path, lock: Lock) -> Path:
    return _dump(folder / LOCK_FILE, LOCK_HEADER, lock)


def intent_sha256(intent: Intent) -> str:
    return hashlib.sha256(intent.model_dump_json().encode()).hexdigest()
