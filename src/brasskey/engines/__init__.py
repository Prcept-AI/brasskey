# SPDX-License-Identifier: Apache-2.0
"""Inference engines: identities, what each can run where, and (from milestone d) adapters.

Engines are never imported into the Brasskey process; adapters launch them as subprocesses or
containers (CONTRIBUTING.md, engines stay out of process).
"""
