# SPDX-License-Identifier: Apache-2.0
"""Export the JSON Schemas for brasskey.yaml and brasskey.lock into schemas/.

Run:  uv run python scripts/export_schemas.py
A test fails when the committed schemas differ from the models, so run this after changing them.
"""

import sys
from pathlib import Path

from brasskey.config.schema import export_json_schemas

OUT_DIR = Path(__file__).resolve().parents[1] / "schemas"


def main() -> int:
    for path in export_json_schemas(OUT_DIR):
        print(f"Wrote {path.relative_to(OUT_DIR.parent)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
