# SPDX-License-Identifier: Apache-2.0
"""A stand-in for ``guidellm run``: writes a real GuideLLM 0.8.1 report where it is told to.

Behavior comes from environment variables, so tests can check what Brasskey passes:
- ``FAKE_GUIDELLM_MODE``: ok (default), unauthorized, old-format, no-report, all-failed;
- ``FAKE_REQUIRE_KEY``: the API key GuideLLM must receive in GUIDELLM__SPEC__BACKEND__API_KEY.
It also records its argv and working directory next to the report (``<report>.argv``).
"""

import json
import os
import sys
from pathlib import Path

FIXTURE = (
    Path(__file__).resolve().parent.parent / "fixtures" / "bench" / "guidellm-0.8.1-report.json"
)


def main() -> int:
    argv = sys.argv[1:]
    if argv == ["--version"]:
        print("guidellm version: 0.8.1")
        return 0
    mode = os.environ.get("FAKE_GUIDELLM_MODE", "ok")
    output = json.loads(argv[argv.index("--output") + 1])["path"]
    seed = json.loads(argv[argv.index("--seed") + 1])["value"]
    Path(output + ".argv").write_text(json.dumps({"argv": argv, "cwd": str(Path.cwd())}))
    required = os.environ.get("FAKE_REQUIRE_KEY")
    if mode == "unauthorized" or (
        required and os.environ.get("GUIDELLM__SPEC__BACKEND__API_KEY") != required
    ):
        print("httpx.HTTPStatusError: Client error '401 Unauthorized'", file=sys.stderr)
        return 1
    if mode == "no-report":
        return 0
    report = json.loads(FIXTURE.read_text())
    metrics = report["benchmarks"][0]["metrics"]
    # Vary throughput a little with the seed, like real runs do.
    metrics["output_tokens_per_second"]["successful"]["mean"] *= 1 + (seed % 3) * 0.01
    if mode == "old-format":
        report["metadata"]["version"] = 1
    if mode == "all-failed":
        metrics["request_totals"].update({"successful": 0, "errored": 12})
    Path(output).write_text(json.dumps(report))
    print("Benchmarking complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
