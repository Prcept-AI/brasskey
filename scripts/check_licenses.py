# SPDX-License-Identifier: Apache-2.0
"""Fail if any installed distribution has a license Brasskey cannot depend on.

Policy (CONTRIBUTING.md, Dependencies): dependencies must be Apache-2.0, MIT or BSD compatible; no
GPL or AGPL. Run it in the synced environment: uv run python scripts/check_licenses.py

How a distribution is classified, in order of preference:
1. ``License-Expression`` (PEP 639 SPDX expression), parsed with AND / OR / WITH and parentheses;
2. ``License ::`` trove classifiers;
3. a short free-form ``License`` field.

Statuses: ``allowed`` (permissive), ``denied`` (GPL family, AGPL, SSPL, BUSL...) or ``unknown``
(no usable metadata, or permissive and copyleft mixed in a way only a human can judge). Unknown
distributions fail the check until reviewed and listed in ``REVIEWED`` with a reason.

MPL-2.0 is allowed (file-level, weak copyleft; fine to depend on unmodified) and listed in the
report so it stays visible in license audits.
"""

import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from importlib import metadata
from typing import Literal

Status = Literal["allowed", "denied", "unknown"]

PERMISSIVE = re.compile(
    r"MIT|BSD|Apache|ISC|PSF|Python Software Foundation|Unlicense|MPL-?2\.0|"
    r"Mozilla Public License 2\.0|0BSD|Zlib|CC0|HPND|Public Domain",
    re.IGNORECASE,
)
COPYLEFT = re.compile(
    r"GPL|General Public License|SSPL|Server Side Public License|BUSL|Business Source|"
    r"Commons Clause",
    re.IGNORECASE,
)
WEAK_COPYLEFT = re.compile(r"MPL|Mozilla Public License", re.IGNORECASE)

# Distributions whose metadata is missing or ambiguous, reviewed by hand: name -> reason.
REVIEWED: dict[str, str] = {}

_TOKEN = re.compile(r"\(|\)|[^\s()]+")


def classify_name(name: str) -> Status:
    """Classify one license identifier or free-form license name."""
    copyleft = bool(COPYLEFT.search(name))
    permissive = bool(PERMISSIVE.search(name))
    if copyleft and permissive:
        return "unknown"  # e.g. "BSD or GPL" in free text: needs a human
    if copyleft:
        return "denied"
    if permissive:
        return "allowed"
    return "unknown"


def _either(statuses: Iterable[Status]) -> Status:
    """OR: we may pick the best alternative."""
    found = set(statuses)
    if "allowed" in found:
        return "allowed"
    if "unknown" in found:
        return "unknown"
    return "denied"


def _all_of(statuses: Iterable[Status]) -> Status:
    """AND: every part applies, so the worst one decides."""
    found = set(statuses)
    if "denied" in found:
        return "denied"
    if "unknown" in found:
        return "unknown"
    return "allowed"


class _ExpressionParser:
    """Recursive-descent parser for SPDX license expressions.

    Grammar: expr := term (OR term)* ; term := factor (AND factor)* ;
    factor := "(" expr ")" | ID [WITH ID]
    """

    def __init__(self, text: str) -> None:
        self.symbols = _TOKEN.findall(text)
        self.position = 0

    def parse(self) -> Status:
        status = self._expr()
        if self.position != len(self.symbols):
            raise ValueError("trailing tokens")
        return status

    def _peek(self) -> str | None:
        return self.symbols[self.position] if self.position < len(self.symbols) else None

    def _take(self) -> str:
        symbol = self._peek()
        if symbol is None:
            raise ValueError("unexpected end of expression")
        self.position += 1
        return symbol

    def _expr(self) -> Status:
        parts: list[Status] = [self._term()]
        while (symbol := self._peek()) is not None and symbol.upper() == "OR":
            self._take()
            parts.append(self._term())
        return _either(parts)

    def _term(self) -> Status:
        parts: list[Status] = [self._factor()]
        while (symbol := self._peek()) is not None and symbol.upper() == "AND":
            self._take()
            parts.append(self._factor())
        return _all_of(parts)

    def _factor(self) -> Status:
        symbol = self._take()
        if symbol == "(":
            status = self._expr()
            if self._take() != ")":
                raise ValueError("missing closing parenthesis")
            return status
        if symbol == ")" or symbol.upper() in {"AND", "OR", "WITH"}:
            raise ValueError(f"unexpected {symbol!r}")
        status = classify_name(symbol)
        if (following := self._peek()) is not None and following.upper() == "WITH":
            self._take()
            self._take()  # the exception id (e.g. LLVM-exception) never removes copyleft terms
        return status


def classify_expression(expression: str) -> Status:
    """Classify an SPDX expression, falling back to free-form matching if it doesn't parse."""
    try:
        return _ExpressionParser(expression).parse()
    except ValueError:
        return classify_name(expression)


def classify_classifiers(classifiers: Iterable[str]) -> Status | None:
    """Classify ``License ::`` trove classifiers; None when there are none."""
    licenses = [c for c in classifiers if c.startswith("License ::")]
    if not licenses:
        return None
    statuses: set[Status] = {classify_name(c) for c in licenses}
    if "allowed" in statuses and "denied" in statuses:
        return "unknown"  # dual-licensed per classifiers; and/or is not expressible there
    return _all_of(statuses)


@dataclass(frozen=True, slots=True)
class Finding:
    name: str
    version: str
    status: Status
    source: str
    license_text: str


def check_distribution(
    name: str,
    version: str,
    *,
    expression: str | None,
    classifiers: list[str],
    license_field: str | None,
) -> Finding:
    """Classify one distribution from its metadata."""
    if expression:
        return Finding(name, version, classify_expression(expression), "expression", expression)
    by_classifier = classify_classifiers(classifiers)
    if by_classifier is not None:
        text = "; ".join(c.removeprefix("License :: ") for c in classifiers if "License ::" in c)
        return Finding(name, version, by_classifier, "classifiers", text)
    if license_field and license_field.strip().upper() != "UNKNOWN":
        first_line = license_field.strip().splitlines()[0][:120]
        return Finding(name, version, classify_name(first_line), "license field", first_line)
    return Finding(name, version, "unknown", "none", "")


def _first(values: list[str] | None) -> str | None:
    return values[0] if values else None


def scan_environment() -> list[Finding]:
    """Classify every distribution installed in the running environment."""
    findings: list[Finding] = []
    for dist in metadata.distributions():
        meta = dist.metadata
        findings.append(
            check_distribution(
                dist.name,
                dist.version,
                expression=_first(meta.get_all("License-Expression")),
                classifiers=[str(c) for c in meta.get_all("Classifier") or []],
                license_field=_first(meta.get_all("License")),
            )
        )
    return sorted(findings, key=lambda f: f.name.lower())


def main() -> int:
    findings = scan_environment()
    failures: list[Finding] = []
    for finding in findings:
        reviewed = REVIEWED.get(finding.name.lower())
        status = finding.status
        if status != "allowed" and reviewed:
            status = "allowed"
        elif status != "allowed":
            failures.append(finding)
        note = " (weak copyleft: MPL)" if WEAK_COPYLEFT.search(finding.license_text) else ""
        note += f" (reviewed: {reviewed})" if reviewed else ""
        print(
            f"{status:8} {finding.name}=={finding.version}  "
            f"[{finding.source}] {finding.license_text}{note}"
        )

    print(f"\n{len(findings)} distributions checked, {len(failures)} problem(s).")
    for finding in failures:
        print(
            f"FAIL: {finding.name}=={finding.version} is {finding.status} "
            f"({finding.source}: {finding.license_text or 'no license metadata'}). "
            "Fix: remove it, or review its license and add it to REVIEWED with a reason."
        )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
