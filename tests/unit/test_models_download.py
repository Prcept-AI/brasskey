# SPDX-License-Identifier: Apache-2.0
"""Download estimates and the bandwidth probe (httpx2 MockTransport, fake clock)."""

from pathlib import Path

import httpx2
import pytest

from brasskey.hardware.types import MIB
from brasskey.models.catalog import find_model, load_catalog
from brasskey.models.download import estimate_download, measure_bandwidth
from fakes.hub import FakeHub

ARTIFACT = next(a for a in find_model(load_catalog(), "qwen3-4b").artifacts if a.format == "gguf")


class Clock:
    def __init__(self, *times):
        self.times = list(times)

    def __call__(self):
        return self.times.pop(0) if len(self.times) > 1 else self.times[0]


def client(handler):
    return httpx2.Client(transport=httpx2.MockTransport(handler))


def test_bandwidth_is_bytes_over_time_after_headers():
    seen = {}

    def handler(request):
        seen["range"] = request.headers.get("Range")
        return httpx2.Response(206, content=b"x" * (8 * MIB))

    rate = measure_bandwidth(
        "https://hf.example/f", client=client(handler), clock=Clock(10.0, 12.0)
    )
    assert seen["range"] == f"bytes=0-{8 * MIB - 1}"
    assert rate == pytest.approx(8 * MIB / 2.0)


def test_bandwidth_probe_failures_return_none():
    assert (
        measure_bandwidth("https://hf.example/f", client=client(lambda _r: httpx2.Response(404)))
        is None
    )

    def boom(_request):
        raise httpx2.ConnectError("no network")

    assert measure_bandwidth("https://hf.example/f", client=client(boom)) is None
    tiny = client(lambda _r: httpx2.Response(206, content=b"x" * 1000))
    assert measure_bandwidth("https://hf.example/f", client=tiny, clock=Clock(0.0, 1.0)) is None


def estimate(hub, bandwidth, free=500 * 10**9):
    return estimate_download(ARTIFACT, hub=hub, bandwidth=bandwidth, disk_free=lambda _p: free)


def test_estimate_with_measured_speed():
    hub = FakeHub()
    result = estimate(hub, lambda _url: 50e6)
    assert result.total_bytes == ARTIFACT.download_bytes
    assert result.to_download_bytes == ARTIFACT.download_bytes
    assert result.bandwidth_source == "measured"
    assert result.eta_seconds == pytest.approx(ARTIFACT.download_bytes / 50e6, abs=0.1)
    assert result.fits_on_disk is True
    assert hub.urls[0].endswith("Qwen3-4B-Q4_K_M.gguf")  # probes the largest file


def test_partially_cached_download():
    hub = FakeHub(cached={ARTIFACT.repo_id: 1_000_000_000})
    result = estimate(hub, lambda _url: 50e6)
    assert result.to_download_bytes == ARTIFACT.download_bytes - 1_000_000_000


def test_fully_cached_needs_no_speed_test():
    def never(_url):
        pytest.fail("no probe when nothing is left to download")

    result = estimate(FakeHub(cached={ARTIFACT.repo_id: 10**12}), never)
    assert result.to_download_bytes == 0
    assert result.bandwidth_source == "not-needed"
    assert result.eta_seconds == 0.0


def test_offline_and_failed_probe_have_no_time():
    assert estimate(FakeHub(), None).bandwidth_source == "offline"
    failed = estimate(FakeHub(), lambda _url: None)
    assert failed.bandwidth_source == "failed"
    assert failed.eta_seconds is None


def test_not_enough_disk_space():
    result = estimate(FakeHub(), lambda _url: 50e6, free=10**9)
    assert result.fits_on_disk is False


def test_cache_dir_that_does_not_exist_yet_uses_its_parent(tmp_path):
    hub = FakeHub(cache_dir=tmp_path / "not" / "created" / "yet")
    seen: list[Path] = []
    estimate_download(ARTIFACT, hub=hub, bandwidth=None, disk_free=lambda p: seen.append(p) or 1)
    assert seen == [tmp_path]
