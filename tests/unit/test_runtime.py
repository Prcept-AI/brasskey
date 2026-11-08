# SPDX-License-Identifier: Apache-2.0
"""Supervisor (with a real fake-engine process), ports, state files and the foreground loop."""

import os
import signal
import socket
import stat
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx2
import psutil
import pytest

from brasskey.engines.base import (
    EngineNotInstalledError,
    EngineStartError,
    LocalModel,
    ServeOptions,
)
from brasskey.errors import BrasskeyError
from brasskey.hardware.commands import CommandOutput
from brasskey.paths import resolve_paths
from brasskey.runtime import ports
from brasskey.runtime.foreground import serve_until_stopped
from brasskey.runtime.ports import PortInUseError, choose_port, port_is_free
from brasskey.runtime.state import ServerRecord, list_records, remove_record, write_record
from brasskey.runtime.supervisor import ProcessSupervisor
from fakes import machines
from fakes.engine import FAKE_INSTALL, TINY_ARTIFACT, FakeAdapter

HW = machines.m3_pro_18gb()


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def process_gone(pid: int) -> bool:
    """Dead, or a zombie waiting to be reaped by its (already dead) parent's adopter."""
    try:
        return psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return True


def launch(adapter, *, api_key_env=None):
    local = LocalModel(artifact=TINY_ARTIFACT, snapshot_dir=Path(), files=())
    options = ServeOptions(
        port=free_port(), served_model_name="tiny", context_length=512, api_key_env=api_key_env
    )
    return adapter.build_launch(FAKE_INSTALL, local, options, HW)


@pytest.fixture
def supervisor():
    return ProcessSupervisor(resolve_paths())


@pytest.fixture
def client():
    with httpx2.Client(timeout=2.0, trust_env=False) as http:
        yield http


def test_start_ready_and_stop(supervisor, client):
    adapter = FakeAdapter()
    spec = launch(adapter)
    handle = supervisor.start(spec, server_id="t1", engine_name="Fake")
    try:
        seconds = supervisor.wait_until_ready(handle, adapter, client=client)
        assert seconds >= 0.1  # it reported "loading" first
        assert client.get(f"{spec.base_url}/models").json()["data"][0]["id"] == "tiny"
    finally:
        code = supervisor.stop(handle)
    assert handle.process.poll() is not None
    assert code is not None


def test_log_file_is_private(supervisor, client):
    adapter = FakeAdapter()
    handle = supervisor.start(launch(adapter), server_id="t2", engine_name="Fake")
    try:
        supervisor.wait_until_ready(handle, adapter, client=client)
    finally:
        supervisor.stop(handle)
    assert stat.S_IMODE(handle.log_path.stat().st_mode) == 0o600
    assert "fake engine pid=" in handle.log_path.read_text()


def test_crash_is_explained_in_plain_language(supervisor, client):
    adapter = FakeAdapter("crash")
    handle = supervisor.start(launch(adapter), server_id="t3", engine_name="Fake")
    with pytest.raises(EngineStartError) as caught:
        supervisor.wait_until_ready(handle, adapter, client=client)
    assert "ran out of memory" in caught.value.message
    assert "--context" in (caught.value.fix or "")
    assert str(handle.log_path) in (caught.value.fix or "")


def test_hang_times_out_and_is_stopped(supervisor, client):
    adapter = FakeAdapter("hang", startup_timeout_s=0.8)
    handle = supervisor.start(launch(adapter), server_id="t4", engine_name="Fake")
    with pytest.raises(EngineStartError) as caught:
        supervisor.wait_until_ready(handle, adapter, client=client)
    assert "wasn't ready after" in caught.value.message
    assert handle.process.poll() is not None


def test_stop_escalates_to_sigkill_and_leaves_no_orphans(supervisor, client, tmp_path):
    pid_file = tmp_path / "child.pid"
    adapter = FakeAdapter("ignore-term", child_pid_file=pid_file)
    handle = supervisor.start(launch(adapter), server_id="t5", engine_name="Fake")
    supervisor.wait_until_ready(handle, adapter, client=client)
    child = int(pid_file.read_text())
    assert psutil.pid_exists(child)
    started = time.monotonic()
    code = supervisor.stop(handle, grace_s=0.5)
    assert code == -signal.SIGKILL
    assert time.monotonic() - started < 5
    deadline = time.monotonic() + 3
    while not process_gone(child) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert process_gone(child), "the engine's child process outlived it"


def test_secret_comes_from_the_environment_never_argv(client):
    supervisor = ProcessSupervisor(resolve_paths(), env={**os.environ, "MY_KEY": "s3cret"})
    adapter = FakeAdapter()
    spec = launch(adapter, api_key_env="MY_KEY")
    assert "s3cret" not in " ".join(spec.argv)
    handle = supervisor.start(spec, server_id="t6", engine_name="Fake")
    try:
        supervisor.wait_until_ready(handle, adapter, client=client)
    finally:
        supervisor.stop(handle)
    log = handle.log_path.read_text()
    assert "api key set: True" in log
    assert "s3cret" not in log


def test_missing_secret_variable_is_a_plain_error():
    supervisor = ProcessSupervisor(resolve_paths(), env={"PATH": os.environ["PATH"]})
    with pytest.raises(BrasskeyError) as caught:
        supervisor.start(
            launch(FakeAdapter(), api_key_env="NOPE"), server_id="t7", engine_name="Fake"
        )
    assert "NOPE" in caught.value.message


def test_missing_executable():
    spec = launch(FakeAdapter()).model_copy(update={"argv": ("/nonexistent/engine",)})
    with pytest.raises(EngineNotInstalledError):
        ProcessSupervisor(resolve_paths()).start(spec, server_id="t8", engine_name="Fake")


# --- foreground loop ------------------------------------------------------------------------------


def test_ctrl_c_stops_the_engine(supervisor, client):
    adapter = FakeAdapter()
    handle = supervisor.start(launch(adapter), server_id="t9", engine_name="Fake")
    supervisor.wait_until_ready(handle, adapter, client=client)

    def interrupted() -> int:
        raise KeyboardInterrupt

    previous = signal.getsignal(signal.SIGTERM)
    assert serve_until_stopped(handle, supervisor, wait=interrupted) is None
    assert handle.process.poll() is not None
    assert signal.getsignal(signal.SIGTERM) == previous  # handler restored


def test_engine_exit_is_reported(supervisor, client):
    adapter = FakeAdapter("exit-later")
    handle = supervisor.start(launch(adapter), server_id="t10", engine_name="Fake")
    supervisor.wait_until_ready(handle, adapter, client=client)
    assert serve_until_stopped(handle, supervisor) == 3


# --- ports ----------------------------------------------------------------------------------------


def test_busy_port_is_detected_and_a_requested_one_is_never_changed():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        busy = listener.getsockname()[1]
        assert not port_is_free("127.0.0.1", busy)
        with pytest.raises(PortInUseError):
            choose_port("127.0.0.1", busy)


def test_default_port_moves_on_when_8000_is_busy(monkeypatch):
    monkeypatch.setattr(ports, "port_is_free", lambda _host, port: port != 8000)
    assert choose_port("127.0.0.1", None) == (8001, "Port 8000 is busy; using 8001.")
    monkeypatch.setattr(ports, "port_is_free", lambda _host, _port: True)
    assert choose_port("127.0.0.1", None) == (8000, None)


# --- state files ----------------------------------------------------------------------------------


def record(pid: int, started: float) -> ServerRecord:
    return ServerRecord(
        id="srv",
        pid=pid,
        process_started_at=started,
        engine="llamacpp",
        engine_version="1",
        model_id="tiny",
        host="127.0.0.1",
        port=8000,
        base_url="http://127.0.0.1:8000/v1",
        auth="none",
        started_at=datetime.now(UTC),
        log_path="server.log",
        brasskey_version="0",
    )


def test_state_records_are_private_and_listed_while_live():
    paths = resolve_paths()
    me = psutil.Process()
    path = write_record(paths, record(me.pid, me.create_time()))
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert [r.id for r in list_records(paths)] == ["srv"]
    remove_record(paths, "srv")
    assert list_records(paths) == []


@pytest.mark.parametrize("start_time", [None, 12345.0])
def test_stale_records_are_removed(start_time):
    paths = resolve_paths()
    write_record(paths, record(os.getpid(), 1.0))
    assert list_records(paths, start_time=lambda _pid: start_time) == []
    assert not any(paths.servers_dir.glob("*.json"))


def test_unreadable_records_are_removed():
    paths = resolve_paths()
    paths.servers_dir.mkdir(parents=True)
    (paths.servers_dir / "junk.json").write_text("{not json")
    assert list_records(paths) == []
    assert not (paths.servers_dir / "junk.json").exists()


# --- the engine's private working folder --------------------------------------------------------


def test_cwd_links_are_created_for_the_engine_and_removed_on_stop(supervisor, client, tmp_path):
    target = tmp_path / "snapshot"
    target.mkdir()
    adapter = FakeAdapter()
    spec = launch(adapter).model_copy(update={"cwd_links": {"qwen3-0.6b": target}})
    handle = supervisor.start(spec, server_id="links", engine_name="Fake")
    try:
        supervisor.wait_until_ready(handle, adapter, client=client)
        assert handle.run_dir is not None
        link = handle.run_dir / "qwen3-0.6b"
        assert link.is_symlink()
        assert link.resolve() == target.resolve()
        assert psutil.Process(handle.pid).cwd() == str(handle.run_dir.resolve())
    finally:
        supervisor.stop(handle)
    assert not handle.run_dir.exists()
    assert target.is_dir()  # only the link went away


@pytest.mark.parametrize("name", ["../escape", "/etc/passwd"])
def test_unsafe_link_names_are_refused(supervisor, name, tmp_path):
    spec = launch(FakeAdapter()).model_copy(update={"cwd_links": {name: tmp_path}})
    with pytest.raises(BrasskeyError, match="Unsafe link name"):
        supervisor.start(spec, server_id="bad", engine_name="Fake")


def test_nested_link_names_for_hf_models(supervisor, client, tmp_path):
    adapter = FakeAdapter()
    spec = launch(adapter).model_copy(update={"cwd_links": {"hf:Qwen/Qwen3-4B-MLX": tmp_path}})
    handle = supervisor.start(spec, server_id="nested", engine_name="Fake")
    try:
        assert handle.run_dir is not None
        assert (handle.run_dir / "hf:Qwen" / "Qwen3-4B-MLX").is_symlink()
    finally:
        supervisor.stop(handle)


def test_containers_are_stopped_with_the_adapters_stop_command(client):
    """Signals reach only the Docker client; `docker stop` is what ends the container."""
    adapter = FakeAdapter()
    spec = launch(adapter).model_copy(update={"stop_argv": ("docker", "stop", "brasskey-x")})
    calls: list[list[str]] = []
    holder: dict[str, int] = {}

    def run(argv, *, timeout_s=5.0):
        calls.append(list(argv))
        os.kill(holder["pid"], signal.SIGTERM)  # what `docker stop` does to the container
        return CommandOutput(0, "brasskey-x\n")

    supervisor = ProcessSupervisor(resolve_paths(), run=run)
    handle = supervisor.start(spec, server_id="t-docker", engine_name="Fake")
    holder["pid"] = handle.pid
    supervisor.wait_until_ready(handle, adapter, client=client)
    supervisor.stop(handle)
    assert calls == [["docker", "stop", "brasskey-x"]]
    assert handle.process.poll() is not None


def test_a_stop_command_that_fails_falls_back_to_signals(client, monkeypatch):
    monkeypatch.setattr("brasskey.runtime.supervisor.STOP_COMMAND_EXIT_WAIT_S", 0.2)
    adapter = FakeAdapter()
    spec = launch(adapter).model_copy(update={"stop_argv": ("docker", "stop", "gone")})
    supervisor = ProcessSupervisor(
        resolve_paths(),
        run=lambda argv, *, timeout_s=5.0: CommandOutput(1, "", "No such container"),
    )
    handle = supervisor.start(spec, server_id="t-docker-2", engine_name="Fake")
    supervisor.wait_until_ready(handle, adapter, client=client)
    supervisor.stop(handle)
    assert handle.process.poll() is not None
