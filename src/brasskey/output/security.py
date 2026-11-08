# SPDX-License-Identifier: Apache-2.0
"""Human view of ``brasskey doctor --security`` results."""

from collections import Counter

from rich.console import Group, RenderableType
from rich.text import Text

from brasskey.output.hardware import LABEL_STYLE, NOTE_STYLE
from brasskey.security.doctor import CheckResult, CheckStatus

STATUS_STYLE = {
    CheckStatus.PASS: "bold green",
    CheckStatus.WARN: "bold yellow",
    CheckStatus.FAIL: "bold red",
    CheckStatus.SKIP: "dim",
}


def summary_counts(results: list[CheckResult]) -> Counter[CheckStatus]:
    return Counter(result.status for result in results)


def security_view(results: list[CheckResult], *, target: str | None = None) -> RenderableType:
    heading = f"Security checks for {target}" if target else "Security checks for this machine"
    lines: list[RenderableType] = [Text(heading, style=LABEL_STYLE)]
    for result in results:
        status = result.status.value.upper()
        line = Text.assemble(
            (f"{status:<5} ", STATUS_STYLE[result.status]),
            (f"{result.id}  ", NOTE_STYLE),
            (f"{result.title}: ", LABEL_STYLE),
            result.detail,
        )
        lines.append(line)
        if result.fix and result.status in (CheckStatus.WARN, CheckStatus.FAIL):
            lines.append(Text.assemble("      ", ("Fix: ", LABEL_STYLE), result.fix))
    counts = summary_counts(results)
    lines.append(Text(""))
    lines.append(
        Text(
            f"{counts[CheckStatus.PASS]} passed, {counts[CheckStatus.WARN]} warning(s), "
            f"{counts[CheckStatus.FAIL]} failed, {counts[CheckStatus.SKIP]} skipped."
        )
    )
    return Group(*lines)
