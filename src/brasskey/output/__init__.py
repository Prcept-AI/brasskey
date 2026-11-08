# SPDX-License-Identifier: Apache-2.0
"""Output layer: human (Rich) and machine (--json) rendering, the "Next:" footer and prompts.

Kept import-free on purpose: ``brasskey.errors`` imports ``brasskey.output.next_step`` and an
eager import here would create a cycle.
"""
