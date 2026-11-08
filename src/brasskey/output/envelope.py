# SPDX-License-Identifier: Apache-2.0
"""The ``--json`` output envelope.

With ``--json`` a command writes exactly one JSON document to stdout (logs and progress go to
stderr), so scripts can rely on this shape::

    {"schema": "brasskey.output/v1", "command": "run", "ok": true,
     "data": {...}, "error": null, "next": {"command": "...", "hint": "..."}}

``ok`` means the command itself completed. A check that ran and FAILED is still ``ok: true``; its
verdict lives in ``data`` and in the exit code.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from brasskey.output.next_step import NextStep

OUTPUT_SCHEMA = "brasskey.output/v1"


class ErrorInfo(BaseModel):
    """A plain-language error with a stable code and a concrete fix."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    code: str
    message: str
    fix: str | None = None


class Envelope(BaseModel):
    """One machine-readable result or error."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Named schema_id because BaseModel reserves `schema`; serialized as "schema".
    schema_id: Literal["brasskey.output/v1"] = Field(default=OUTPUT_SCHEMA, alias="schema")
    command: str
    ok: bool
    data: dict[str, JsonValue] | None = None
    error: ErrorInfo | None = None
    next: NextStep | None = None

    def to_json(self) -> str:
        """Serialize with the public field names."""
        return self.model_dump_json(by_alias=True)
