# Testing on NVIDIA GPUs (AWS)

`uv run pytest -m gpu` runs the real-hardware tests:

- the NVML probe (`tests/integration/test_hw_real.py`);
- `brasskey run` with vLLM, SGLang and a CUDA build of llama.cpp on Qwen3 0.6B
  (`tests/integration/test_engine_nvidia.py`).

They need Linux and an NVIDIA GPU, so they run on a cloud instance, not on a laptop or in CI.

## Instance

| Purpose | Instance | GPU | Notes |
| --- | --- | --- | --- |
| Default | g6.xlarge | 1 x L4, 24 GB, compute 8.9 (Ada) | bf16 and fp8 paths |
| Older GPUs | g4dn.xlarge | 1 x T4, 16 GB, compute 7.5 (Turing) | bf16 runs as float16; no fp8 |
| Ampere | g5.xlarge | 1 x A10G, 24 GB, compute 8.6 | |
| Multi-GPU | g6.12xlarge | 4 x L4 | tensor parallelism |

- **Image:** "AWS Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04)". It ships the
  NVIDIA driver (595 in the October 2026 release, which supports CUDA 13.2), Docker and the
  NVIDIA Container Toolkit.
- **Disk:** a root volume of at least 150 GB. The images take 9.2 GB (vLLM) and 14.4 GB (SGLang)
  to download, roughly double that unpacked, plus the models.
- **Network:** allow SSH (port 22) from your own IP only. Never open the model port: test through
  an SSH tunnel (`ssh -L 8000:127.0.0.1:8000 …`).
- **Cost:** stop or terminate the instance when you're done.

## Steps

```bash
ssh ubuntu@<instance>
nvidia-smi                      # the driver and GPUs
docker info | grep -i nvidia    # the NVIDIA runtime is configured

curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.local/bin/env
git clone https://github.com/Prcept-AI/brasskey.git && cd brasskey   # or rsync your checkout
uv sync

uv run brasskey doctor                   # GPUs, driver, vLLM / SGLang (Docker, pinned images)
uv run brasskey run --dry-run            # what Brasskey would start here
uv run pytest                            # unit tests (no GPU needed)
```

Then build llama.cpp with CUDA, exactly as Brasskey tells users to:

```bash
sudo apt-get install -y cmake            # if `cmake --version` fails
export PATH=/usr/local/cuda/bin:$PATH    # if cmake can't find nvcc
uv run brasskey run qwen3-0.6b --engine llamacpp --dry-run   # Next: the CUDA build command
# run that command from your home folder; time it
uv run brasskey doctor                   # llama.cpp now lists CUDA
uv run pytest -m gpu -v                  # real NVML + vLLM + SGLang + llama.cpp (CUDA)
```

The first `-m gpu` run pulls both images and downloads Qwen3 0.6B (1.5 GB), so allow
20–30 minutes. Later runs take minutes.

Optional: check the warning for a llama.cpp build without CUDA on this GPU machine.

```bash
cd ~/llama.cpp && cmake -B build-cpu -DBUILD_SHARED_LIBS=OFF \
  && cmake --build build-cpu -j --target llama-server && cd ~/brasskey
PATH=~/llama.cpp/build-cpu/bin:$PATH uv run brasskey run qwen3-0.6b --engine llamacpp --dry-run
# expected: planned for system RAM, with "This llama.cpp build reports no GPU it can use"
```

To try it by hand:

```bash
uv run brasskey run qwen3-0.6b --engine vllm --generate-api-key
# in a second terminal:
curl http://127.0.0.1:8000/v1/models -H "Authorization: Bearer <the key it printed>"
uv run brasskey doctor --security
```

## What to report back

- The output of `uv run pytest -m gpu -v`.
- `uv run brasskey doctor --json`.
- How long the llama.cpp CUDA build took, and any error from it.
- For any failure, the engine log path printed in the error.

The assumptions most likely to need adjusting are listed in
[ADR 0010](decisions/0010-vllm-and-sglang-adapters.md#consequences).
