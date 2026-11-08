# SPDX-License-Identifier: Apache-2.0
"""Facts about a Python engine installed in its own environment, read without importing it.

Engines run in their own environments (they pin PyTorch and CUDA), so Brasskey asks the engine's
own interpreter, found from the shebang of its console script, for package versions. The probe
reads metadata and torch's ``version.py`` as text: importing torch or the engine would take
seconds and load CUDA.
"""

import json
import logging
from pathlib import Path
from typing import cast

from brasskey.hardware.commands import CommandRunner

logger = logging.getLogger(__name__)

PROBE_TIMEOUT_S = 30.0

_PROBE_TEMPLATE = """
import importlib.metadata as m, importlib.util as u, json, pathlib, re
out = {"version": m.version(PACKAGE)}
spec = u.find_spec("torch")
if spec is not None and spec.origin:
    text = (pathlib.Path(spec.origin).parent / "version.py").read_text()
    found = re.search(r"^cuda\\b[^=\\n]*=\\s*['\\"]([0-9.]+)", text, re.M)
    if found:
        out["cuda"] = found.group(1)
print(json.dumps(out))
"""


def script_interpreter(script: str) -> str | None:
    """The Python that runs a console script (its absolute shebang), or None."""
    try:
        first = Path(script).read_text(encoding="utf-8", errors="replace").splitlines()[0]
    except (OSError, IndexError):
        return None
    if first.startswith("#!") and first[2:].strip().startswith("/"):
        candidate = first[2:].strip().split()[0]
        if Path(candidate).is_file() and "env" not in Path(candidate).name:
            return candidate
    return None


def probe_source(package: str) -> str:
    """Python source printing {"version": ..., "cuda": ...} for ``package`` (no imports of it)."""
    return f"PACKAGE = {package!r}\n" + _PROBE_TEMPLATE


def parse_probe(text: str) -> tuple[str, str | None] | None:
    """(package version, torch CUDA version or None) from the probe's JSON line."""
    for line in reversed(text.strip().splitlines()):
        try:
            data: object = json.loads(line)
        except ValueError:
            continue
        if not isinstance(data, dict):
            continue
        fields = cast(dict[str, object], data)
        version = fields.get("version")
        if isinstance(version, str):
            cuda = fields.get("cuda")
            return version, cuda if isinstance(cuda, str) else None
    return None


def probe_package(run: CommandRunner, python: str, package: str) -> tuple[str, str | None] | None:
    out = run([python, "-c", probe_source(package)], timeout_s=PROBE_TIMEOUT_S)
    if not out.ok:
        logger.debug("Probe of %s with %s failed (exit %d)", package, python, out.returncode)
        return None
    return parse_probe(out.stdout)
