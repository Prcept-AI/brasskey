# SPDX-License-Identifier: Apache-2.0
"""Replaying brasskey.lock, and planning with the engines this build can start and you have."""

import pytest

from brasskey.config.intent import Intent
from brasskey.engines.types import EngineId
from brasskey.models.catalog import load_catalog
from brasskey.planner import NoFitError, RunRequest, UnsupportedEngineError, plan_run
from brasskey.planner.replay import hardware_summary, lock_from_plan, plan_from_lock
from fakes import machines

CATALOG = load_catalog()
MAC = machines.m3_pro_18gb()


def locked(model="qwen3-4b", engine=EngineId.LLAMACPP, hw=MAC, context=8192):
    plan = plan_run(RunRequest(model=model, engine=engine, context_length=context), hw, CATALOG)
    return plan, lock_from_plan(plan, hw, intent=Intent(model=model), host="127.0.0.1", port=None)


def test_replay_on_the_same_machine_gives_the_same_plan():
    plan, lock = locked()
    replayed = plan_from_lock(lock, MAC, runnable=[EngineId.LLAMACPP])
    assert replayed.engine == plan.engine
    assert replayed.fit.artifact == plan.fit.artifact
    assert replayed.fit.required_bytes == plan.fit.required_bytes
    assert replayed.reasons[0].startswith("Replaying brasskey.lock")
    assert len(replayed.reasons) == 1


def test_replay_on_other_hardware_says_so():
    _plan, lock = locked()
    replayed = plan_from_lock(lock, machines.mac(36, "M3 Max", 400))
    assert "differs from the one in the lock" in replayed.reasons[1]


def test_replay_that_no_longer_fits():
    _plan, lock = locked(model="qwen3-8b")
    with pytest.raises(NoFitError) as caught:
        plan_from_lock(lock, machines.cpu_only(4))
    assert "brasskey.lock" in caught.value.message


def test_replay_with_an_engine_that_cannot_run_here():
    _plan, lock = locked()
    lock = lock.model_copy(update={"engine": lock.engine.model_copy(update={"id": EngineId.MLX})})
    with pytest.raises(UnsupportedEngineError):
        plan_from_lock(lock, machines.h100())


def test_planning_only_uses_engines_this_build_can_start():
    plan = plan_run(RunRequest(), MAC, CATALOG, runnable=[EngineId.LLAMACPP])
    assert plan.engine == EngineId.LLAMACPP
    with pytest.raises(UnsupportedEngineError, match="can't start MLX yet"):
        plan_run(RunRequest(engine=EngineId.MLX), MAC, CATALOG, runnable=[EngineId.LLAMACPP])


def test_installed_engines_come_first():
    both = [EngineId.MLX, EngineId.LLAMACPP]
    assert plan_run(RunRequest(), MAC, CATALOG, runnable=both).engine == EngineId.MLX
    assert (
        plan_run(RunRequest(), MAC, CATALOG, runnable=both, installed=[EngineId.LLAMACPP]).engine
        == EngineId.LLAMACPP
    )


def test_hardware_summary():
    assert hardware_summary(MAC) == "Apple M3 Pro, 18 GiB RAM, Apple M3 Pro GPU"
    assert hardware_summary(machines.h100(8)).endswith("8 x NVIDIA H100 80GB HBM3")
