# SPDX-License-Identifier: Apache-2.0
"""scripts/install.sh, run for real under `sh` with stub uv / curl / uname (no network)."""

import hashlib
import io
import os
import re
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "install.sh"
SH = shutil.which("sh")

pytestmark = pytest.mark.skipif(SH is None, reason="needs a POSIX sh")

FAKE_UV = """#!/bin/sh
echo "$@" >> "$UV_LOG"
case "$1" in
  --version) echo "uv ${FAKE_UV_VERSION:-0.13.6} (fake)" ;;
  tool) [ "$2" = "dir" ] && echo "$FAKE_BIN_DIR" ;;
esac
exit 0
"""


def write_tool(folder: Path, name: str, body: str) -> None:
    path = folder / name
    path.write_text(body)
    path.chmod(0o755)


def uv_archive(target: str) -> bytes:
    """A .tar.gz shaped like uv's release archive, with a fake uv and uvx inside."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name in ("uv", "uvx"):
            data = FAKE_UV.encode()
            entry = tarfile.TarInfo(f"uv-{target}/{name}")
            entry.size = len(data)
            entry.mode = 0o755
            archive.addfile(entry, io.BytesIO(data))
    return buffer.getvalue()


class Machine:
    """A sandbox: HOME, a PATH with only stubs and the system tools, and logs to inspect."""

    def __init__(self, tmp: Path, *, os_name: str = "Linux", arch: str = "x86_64") -> None:
        self.tmp = tmp
        self.stubs = tmp / "stubs"
        self.stubs.mkdir()
        self.home = tmp / "home"
        self.home.mkdir()
        self.bin_dir = tmp / "tool-bin"
        self.uv_log = tmp / "uv.log"
        self.curl_log = tmp / "curl.log"
        self.archive = tmp / "uv-download.tar.gz"
        self.extra_env: dict[str, str] = {}
        write_tool(
            self.stubs, "uname", f'#!/bin/sh\n[ "$1" = "-s" ] && echo {os_name} || echo {arch}\n'
        )
        write_tool(self.stubs, "ldd", "#!/bin/sh\necho 'ldd (GNU libc) 2.39'\n")
        write_tool(
            self.stubs,
            "curl",
            '#!/bin/sh\necho "$@" >> "$CURL_LOG"\n'
            'while [ $# -gt 0 ]; do [ "$1" = "-o" ] && out="$2"; shift; done\n'
            'cp "$FAKE_ARCHIVE" "$out"\n',
        )

    def with_uv(self, version: str = "0.13.6") -> None:
        write_tool(self.stubs, "uv", FAKE_UV)
        self.extra_env = {"FAKE_UV_VERSION": version}

    def run(self, **env: str) -> subprocess.CompletedProcess[str]:
        system_path = "/usr/bin:/bin:/usr/sbin:/sbin"
        environment = {
            "PATH": f"{self.stubs}:{system_path}",
            "HOME": str(self.home),
            "UV_LOG": str(self.uv_log),
            "CURL_LOG": str(self.curl_log),
            "FAKE_ARCHIVE": str(self.archive),
            "FAKE_BIN_DIR": str(self.bin_dir),
            **self.extra_env,
            **env,
        }
        return subprocess.run(
            [str(SH), str(SCRIPT)], env=environment, capture_output=True, text=True, timeout=60,
            check=False,
        )  # fmt: skip

    def uv_calls(self) -> list[str]:
        return self.uv_log.read_text().splitlines() if self.uv_log.exists() else []


def test_existing_uv_installs_brasskey_and_downloads_nothing(tmp_path):
    machine = Machine(tmp_path)
    machine.with_uv()
    result = machine.run()
    assert result.returncode == 0, result.stderr
    assert "tool install --upgrade brasskey" in machine.uv_calls()
    assert not machine.curl_log.exists()
    assert "isn't on your PATH yet" in result.stderr  # the tool bin dir isn't on the sandbox PATH
    assert "Next:" in result.stderr


def test_a_pinned_version(tmp_path):
    machine = Machine(tmp_path)
    machine.with_uv()
    result = machine.run(BRASSKEY_VERSION="0.1.0a1")
    assert result.returncode == 0, result.stderr
    assert "tool install --upgrade brasskey==0.1.0a1" in machine.uv_calls()


def test_old_uv_gets_a_hint_but_is_left_alone(tmp_path):
    machine = Machine(tmp_path)
    machine.with_uv(version="0.9.2")
    result = machine.run()
    assert result.returncode == 0, result.stderr
    assert "uv self update" in result.stderr
    assert "self" not in " ".join(machine.uv_calls())  # never updates your uv for you


def test_without_uv_it_downloads_checks_and_installs_uv(tmp_path):
    machine = Machine(tmp_path, os_name="Linux", arch="aarch64")
    data = uv_archive("aarch64-unknown-linux-gnu")
    machine.archive.write_bytes(data)
    install_dir = tmp_path / "local-bin"
    result = machine.run(
        BRASSKEY_INSTALL_DIR=str(install_dir),
        BRASSKEY_UV_URL="https://example.invalid/uv.tar.gz",
        BRASSKEY_UV_SHA256=hashlib.sha256(data).hexdigest(),
    )
    assert result.returncode == 0, result.stderr
    assert (install_dir / "uv").is_file()
    assert (install_dir / "uvx").is_file()
    assert os.access(install_dir / "uv", os.X_OK)
    assert "tool install --upgrade brasskey" in machine.uv_calls()  # with the uv it installed
    assert "--proto =https" in machine.curl_log.read_text()


def test_the_pinned_uv_release_url_and_checksum_are_used_by_default(tmp_path):
    machine = Machine(tmp_path, os_name="Darwin", arch="arm64")
    machine.archive.write_bytes(uv_archive("aarch64-apple-darwin"))  # not the real archive
    result = machine.run(BRASSKEY_INSTALL_DIR=str(tmp_path / "local-bin"))
    assert result.returncode == 1
    assert "uv-aarch64-apple-darwin.tar.gz" in machine.curl_log.read_text()
    assert "doesn't match its checksum" in result.stderr  # the fake isn't the real uv: refused
    assert not (tmp_path / "local-bin" / "uv").exists()


def test_a_wrong_checksum_installs_nothing(tmp_path):
    machine = Machine(tmp_path)
    machine.archive.write_bytes(uv_archive("x86_64-unknown-linux-gnu"))
    result = machine.run(
        BRASSKEY_INSTALL_DIR=str(tmp_path / "local-bin"),
        BRASSKEY_UV_URL="https://example.invalid/uv.tar.gz",
        BRASSKEY_UV_SHA256="0" * 64,
    )
    assert result.returncode == 1
    assert "nothing was installed" in result.stderr
    assert not (tmp_path / "local-bin").exists() or not any((tmp_path / "local-bin").iterdir())
    assert machine.uv_calls() == []


@pytest.mark.parametrize(
    ("os_name", "arch", "message"),
    [("FreeBSD", "x86_64", "FreeBSD isn't supported"), ("Linux", "riscv64", "riscv64")],
)
def test_unsupported_machines_are_refused_plainly(tmp_path, os_name, arch, message):
    machine = Machine(tmp_path, os_name=os_name, arch=arch)
    result = machine.run()
    assert result.returncode == 1
    assert message in result.stderr


def test_mirror_settings_need_both_values(tmp_path):
    machine = Machine(tmp_path)
    result = machine.run(BRASSKEY_UV_URL="https://example.invalid/uv.tar.gz")
    assert result.returncode == 1
    assert "set both BRASSKEY_UV_URL and BRASSKEY_UV_SHA256" in result.stderr


def test_the_script_pins_the_same_uv_as_ci():
    script = SCRIPT.read_text()
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    script_version = re.search(r'^UV_VERSION="([^"]+)"', script, re.MULTILINE)
    ci_version = re.search(r'UV_VERSION: "([^"]+)"', ci)
    assert script_version is not None
    assert ci_version is not None
    assert script_version.group(1) == ci_version.group(1)
    for target in ("aarch64_apple_darwin", "x86_64_apple_darwin", "aarch64_unknown_linux_gnu",
                   "x86_64_unknown_linux_gnu", "aarch64_unknown_linux_musl",
                   "x86_64_unknown_linux_musl"):  # fmt: skip
        assert re.search(rf'^UV_SHA256_{target}="[0-9a-f]{{64}}"$', script, re.MULTILINE), target


def test_the_script_runs_only_when_complete():
    """A truncated download must not run half a script: everything is inside main()."""
    lines = [line for line in SCRIPT.read_text().splitlines() if line.strip()]
    assert lines[-1] == 'main "$@"'
