# SPDX-License-Identifier: Apache-2.0
"""NVIDIA probe with a fake NVML: GPUs, failures, MIG, busy memory, CUDA_VISIBLE_DEVICES."""

from dataclasses import replace

import pytest

from brasskey.hardware.nvidia import (
    NvmlErrorKind,
    NvmlProbe,
    format_cuda_version,
    nvidia_pci_present,
    parse_visible_devices,
    peak_bandwidth_gbs,
)
from brasskey.hardware.types import MIB, AcceleratorKind
from fakes.commands import write_tree
from fakes.nvml import A100_40GB, H100_SXM, L4, RTX_4090, T4, FakeNvml, fleet


def _probe(nvml, *, env=None, pci=False):
    return NvmlProbe(
        nvml, platform_name="linux", env=env or {}, pci_check=lambda _root: pci
    ).probe()


def test_eight_h100s():
    nvml = FakeNvml(fleet(H100_SXM, 8))
    result = _probe(nvml)
    assert len(result.accelerators) == 8
    assert [gpu.index for gpu in result.accelerators] == list(range(8))
    gpu = result.accelerators[0]
    assert gpu.kind == AcceleratorKind.NVIDIA_GPU
    assert gpu.name == "NVIDIA H100 80GB HBM3"
    assert gpu.memory_total_bytes == 81559 * MIB
    assert gpu.memory_free_bytes == (81559 - 500) * MIB
    assert gpu.compute_capability == (9, 0)
    assert gpu.driver_version == "560.35.03"
    assert gpu.cuda_driver_version == "12.6"
    assert gpu.mig_enabled is False
    assert gpu.mem_bandwidth_gbs == pytest.approx(3352, abs=1)
    assert result.issues == ()
    assert nvml.shut_down is True


@pytest.mark.parametrize(
    ("gpu", "published_gbs"),
    [(H100_SXM, 3352), (A100_40GB, 1555), (RTX_4090, 1008), (L4, 300), (T4, 320)],
)
def test_bandwidth_matches_published_specs(gpu, published_gbs):
    (accelerator,) = _probe(FakeNvml([gpu])).accelerators
    assert accelerator.mem_bandwidth_gbs == pytest.approx(published_gbs, rel=0.01)


def test_bandwidth_unsupported_query_is_left_unset():
    (gpu,) = _probe(FakeNvml([replace(T4, bus_width=None)])).accelerators
    assert gpu.mem_bandwidth_gbs is None
    assert gpu.compute_capability == (7, 5)


def test_no_nvidia_hardware_is_silent():
    result = _probe(FakeNvml([], init_error=NvmlErrorKind.LIBRARY_NOT_FOUND), pci=False)
    assert result.accelerators == ()
    assert result.issues == ()


def test_gpu_present_without_driver_explains_the_fix():
    result = _probe(FakeNvml([], init_error=NvmlErrorKind.LIBRARY_NOT_FOUND), pci=True)
    (issue,) = result.issues
    assert issue.severity == "warning"
    assert "driver" in issue.message
    assert issue.fix
    assert "nvidia-smi" in issue.fix


@pytest.mark.parametrize(
    ("kind", "fix_words"),
    [
        (NvmlErrorKind.DRIVER_NOT_LOADED, "modprobe"),
        (NvmlErrorKind.VERSION_MISMATCH, "Reboot"),
        (NvmlErrorKind.NO_PERMISSION, "--gpus all"),
        (NvmlErrorKind.OTHER, "nvidia-smi"),
    ],
)
def test_init_failures_have_plain_fixes(kind, fix_words):
    (issue,) = _probe(FakeNvml([], init_error=kind)).issues
    assert issue.severity == "warning"
    assert issue.fix
    assert fix_words in issue.fix


def test_listing_failure_after_init_is_reported_and_nvml_shut_down():
    nvml = FakeNvml([H100_SXM], count_error=NvmlErrorKind.GPU_LOST)
    result = _probe(nvml)
    assert result.accelerators == ()
    assert "Couldn't list" in result.issues[0].message
    assert nvml.shut_down is True


def test_lost_gpu_is_skipped_others_reported():
    gpus = [*fleet(H100_SXM, 3)]
    gpus[1] = replace(gpus[1], error=NvmlErrorKind.GPU_LOST)
    result = _probe(FakeNvml(gpus))
    assert [gpu.index for gpu in result.accelerators] == [0, 2]
    (issue,) = result.issues
    assert "GPU 1" in issue.message


def test_mig_enabled_is_flagged():
    result = _probe(FakeNvml([replace(A100_40GB, mig=True)]))
    (gpu,) = result.accelerators
    assert gpu.mig_enabled is True
    assert any("MIG" in issue.message for issue in result.issues)
    assert any("-mig 0" in (issue.fix or "") for issue in result.issues)


def test_memory_used_by_other_processes_warns():
    busy = replace(H100_SXM, used_mib=30 * 1024)
    (issue,) = _probe(FakeNvml([busy])).issues
    assert issue.severity == "warning"
    assert "30.0 of 79.6 GiB in use" in issue.message


def test_small_display_usage_does_not_warn():
    (gpu,) = _probe(FakeNvml([replace(RTX_4090, used_mib=700)])).accelerators
    assert gpu.memory_free_bytes == (24564 - 700) * MIB


def test_cuda_visible_devices_by_index():
    result = _probe(FakeNvml(fleet(H100_SXM, 8)), env={"CUDA_VISIBLE_DEVICES": "0,3"})
    assert [gpu.index for gpu in result.accelerators] == [0, 3]
    (issue,) = result.issues
    assert issue.severity == "info"
    assert "2 of 8" in issue.message


def test_cuda_visible_devices_by_uuid_prefix():
    gpus = fleet(H100_SXM, 4)
    prefix = gpus[2].uuid[:12]
    result = _probe(FakeNvml(gpus), env={"CUDA_VISIBLE_DEVICES": prefix})
    assert [gpu.index for gpu in result.accelerators] == [2]


@pytest.mark.parametrize("value", ["", "-1", "-1,0"])
def test_cuda_visible_devices_hiding_everything(value):
    result = _probe(FakeNvml(fleet(H100_SXM, 2)), env={"CUDA_VISIBLE_DEVICES": value})
    assert result.accelerators == ()
    assert "(0 of 2)" in result.issues[0].message


def test_cuda_visible_devices_mig_entries_are_not_applied():
    result = _probe(FakeNvml(fleet(H100_SXM, 2)), env={"CUDA_VISIBLE_DEVICES": "MIG-abc"})
    assert len(result.accelerators) == 2
    assert "MIG" in result.issues[0].message


def test_mixed_gpus_selected_by_index_warn_about_cuda_ordering():
    gpus = [H100_SXM, L4]
    result = _probe(FakeNvml(gpus), env={"CUDA_VISIBLE_DEVICES": "1"})
    assert any("CUDA_DEVICE_ORDER" in (issue.fix or "") for issue in result.issues)
    pci_order = {"CUDA_VISIBLE_DEVICES": "1", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
    result = _probe(FakeNvml(gpus), env=pci_order)
    assert not any("CUDA_DEVICE_ORDER" in (issue.fix or "") for issue in result.issues)


def test_uuids_never_appear_in_the_report():
    gpus = fleet(H100_SXM, 2)
    result = _probe(FakeNvml(gpus), env={"CUDA_VISIBLE_DEVICES": gpus[0].uuid})
    dumped = result.model_dump_json()
    assert "GPU-" not in dumped


@pytest.mark.parametrize(
    ("platform_name", "applies"), [("linux", True), ("win32", True), ("darwin", False)]
)
def test_applicability(platform_name, applies):
    assert NvmlProbe(FakeNvml([]), platform_name=platform_name).is_applicable() is applies


def test_helpers():
    assert format_cuda_version(12080) == "12.8"
    assert format_cuda_version(13000) == "13.0"
    assert peak_bandwidth_gbs(0, 1000) is None
    assert parse_visible_devices(None) is None
    selection = parse_visible_devices("2,GPU-ab,junk,3")
    assert selection is not None
    assert selection.indices == (2,)  # CUDA stops at the first invalid entry
    assert selection.uuid_prefixes == ("GPU-ab",)


def test_pci_scan(tmp_path):
    root = write_tree(
        tmp_path,
        {
            "bus/pci/devices/0000:00:02.0/vendor": "0x8086\n",
            "bus/pci/devices/0000:00:02.0/class": "0x030000\n",
            "bus/pci/devices/0000:65:00.0/vendor": "0x10de\n",
            "bus/pci/devices/0000:65:00.1/vendor": "0x10de\n",
            "bus/pci/devices/0000:65:00.1/class": "0x040300\n",  # the GPU's audio function
        },
    )
    assert nvidia_pci_present(root) is False  # 65:00.0 has no class file
    (root / "bus/pci/devices/0000:65:00.0/class").write_text("0x030200\n")
    assert nvidia_pci_present(root) is True
    assert nvidia_pci_present(tmp_path / "missing") is False
