# 0003: Hardware probe and the hardware report

Date: 2026-11-08
Status: accepted

## Context

Milestone (b) adds the read-only hardware probe that `run` (model fit), `doctor` and
`brasskey.lock` build on. Several numbers it reports can't be read exactly without loading GPU
frameworks into the Brasskey process, which the architecture rules forbid, and the report must be
safe to share.

## Options

- **Apple GPU memory.** Report total RAM (wrong: macOS caps what the GPU may wire), read Metal's
  `recommendedMaxWorkingSetSize` (needs Metal/pyobjc in-process), or use `iogpu.wired_limit_mb`
  when set and otherwise macOS's default policy, labeled as an estimate.
- **Memory bandwidth** (for later decode-speed estimates). Measure it (slow, needs a GPU
  workload), hard-code tables for everything, or derive it where the hardware reports enough.
- **NVIDIA access.** Parse `nvidia-smi` output, or use NVML through NVIDIA's bindings.
- **GPU selection.** Report every GPU, or respect `CUDA_VISIBLE_DEVICES` like the engines do.

## Decision

- **Apple GPU memory:** the accelerator's `memory_total_bytes` is the GPU working-set limit:
  - `iogpu.wired_limit_mb` when set;
  - otherwise about 2/3 of RAM below 36 GiB and about 3/4 from 36 GiB up, with
    `memory_is_estimate = true` and a note.
- **Bandwidth:**
  - NVIDIA: derived from NVML as bus width × max memory clock × 2. This matches the published
    figures, e.g. 3352 GB/s for H100 SXM.
  - Apple Silicon: from Apple's published specs, keyed by GPU cores where Apple sells
    variants; unset for unknown chips and VMs.
  - Always described as a theoretical peak, used only for labeled estimates.
- **NVIDIA:** NVML via `nvidia-ml-py` (BSD-3-Clause; loads the driver's library lazily; not an
  engine) behind a small `NvmlApi` protocol so tests use a fake. Each NVML error kind maps to a
  plain-language fix. "No library" is silent unless an NVIDIA device is present on the PCI bus.
- **Selection:** `CUDA_VISIBLE_DEVICES` is applied (indices and UUID prefixes; MIG entries are
  reported but not applied). Mixed GPU models selected by index without
  `CUDA_DEVICE_ORDER=PCI_BUS_ID` trigger a warning.
- **Containers:** cgroup v1/v2 memory and CPU limits and the CPU affinity mask are reported,
  because psutil sees the host.
- **Privacy:** reports never contain GPU UUIDs, serial numbers, MAC addresses or hostnames. The
  raw `CUDA_VISIBLE_DEVICES` value is not echoed, since it can hold UUIDs. A test enforces this.
- **Fingerprint:** a sha256 over the OS family, CPU model, core counts, features, RAM and the
  accelerators, with memory rounded to GiB. It excludes free memory, issues, OS and driver
  versions, which the lock records separately.
- **Robustness:** a probe never raises; a crashing probe becomes a warning and the rest still run.

## Consequences

- Fit estimates on Macs inherit the estimate label until a measurement replaces them (for example
  after a first successful `run`).
- The Apple bandwidth table needs an entry when Apple ships new chips; until then bandwidth is
  simply unknown, never guessed.
- MIG partitions are not planned for in v0.1; revisit with the vLLM/SGLang adapters (milestone g).
- Real NVIDIA coverage comes from `pytest -m gpu` on AWS instances (milestone g).
