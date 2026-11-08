# SPDX-License-Identifier: Apache-2.0
"""Brasskey: the key to every engine.

Deploy, tune, verify and serve open models on any hardware, measured on your own traffic.
The core logic lives in importable modules; ``brasskey.cli`` is a thin command-line layer on top.
"""

__all__ = ["__version__"]

# Single source of truth for the version (read by hatchling at build time).
__version__ = "0.1.0.dev0"
