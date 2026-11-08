# SPDX-License-Identifier: Apache-2.0
"""The one-line "Next:" suggestion that ends every command (CONTRIBUTING.md, CLI contract).

This is plain data with no Rich dependency so core logic (and a future API or dashboard) can
return a next step without knowing how it will be displayed.
"""

from pydantic import BaseModel, ConfigDict, Field


class NextStep(BaseModel):
    """A suggested follow-up command, rendered as ``Next: <command> (<hint>)``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    command: str = Field(min_length=1, description="Command to run next, ready to copy.")
    hint: str | None = Field(default=None, description="Why to run it, in a few plain words.")

    def as_text(self) -> str:
        """Render as a single line without the ``Next:`` label."""
        if self.hint:
            return f"{self.command} ({self.hint})"
        return self.command


# Shared suggestions, kept in one place so they are easy to update as commands become available.
SEE_COMMANDS = NextStep(command="brasskey --help", hint="see the available commands")
