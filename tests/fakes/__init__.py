# SPDX-License-Identifier: Apache-2.0
"""Fakes that stand in for real hardware, system tools and engines in unit tests."""

from pathlib import Path

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
HARDWARE_FIXTURES = FIXTURES / "hardware"
