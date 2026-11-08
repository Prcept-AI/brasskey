# SPDX-License-Identifier: Apache-2.0
"""Hatchling metadata hook: the README on PyPI gets absolute links and images.

GitHub resolves the README's relative paths (``brand/logo/...``, ``SECURITY.md``); PyPI can't.
So the package description points them at this version's tag on GitHub, while README.md itself
keeps relative paths and renders on GitHub whether the repository is private or public.
Deterministic (no clock, no network), so builds stay reproducible.
"""

import re
from pathlib import Path
from typing import Any

from hatchling.metadata.plugin.interface import MetadataHookInterface

OWNER_REPO = "Prcept-AI/brasskey"
PAGES = f"https://github.com/{OWNER_REPO}"
RAW = f"https://raw.githubusercontent.com/{OWNER_REPO}"
IMAGE_SUFFIXES = (".svg", ".png", ".jpg", ".jpeg", ".gif", ".webp")

_VERSION = re.compile(r'^__version__ = "([^"]+)"', re.MULTILINE)
_LINK = re.compile(r"\]\((?!https?://|#|mailto:)([^)\s]+)\)")  # Markdown links and images
_ATTRIBUTE = re.compile(r'\b(src|srcset)="(?!https?://)([^"]+)"')  # HTML <img> / <source>


def absolute_links(text: str, ref: str) -> str:
    """Rewrite relative README links: images to raw files, folders to trees, files to blobs."""

    def link(match: re.Match[str]) -> str:
        path = match.group(1)
        if path.lower().endswith(IMAGE_SUFFIXES):
            return f"]({RAW}/{ref}/{path})"
        kind = "tree" if path.endswith("/") else "blob"
        return f"]({PAGES}/{kind}/{ref}/{path})"

    def attribute(match: re.Match[str]) -> str:
        return f'{match.group(1)}="{RAW}/{ref}/{match.group(2)}"'

    return _ATTRIBUTE.sub(attribute, _LINK.sub(link, text))


class CustomMetadataHook(MetadataHookInterface):
    """Sets the dynamic ``readme`` from README.md with links pointing at tag ``v<version>``."""

    def update(self, metadata: dict[str, Any]) -> None:
        root = Path(self.root)
        init = (root / "src" / "brasskey" / "__init__.py").read_text(encoding="utf-8")
        found = _VERSION.search(init)
        ref = f"v{found.group(1)}" if found else "main"
        readme = (root / "README.md").read_text(encoding="utf-8")
        metadata["readme"] = {"content-type": "text/markdown", "text": absolute_links(readme, ref)}
