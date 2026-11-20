import json

import numpy as np
import pytest

from drift_monitoring.phase3.contracts import (
    CheckpointedRun, ContractError, atomic_write_json, load_experiment_config,
)


CONFIG = "configs/phase3_strengthening_v3.json"


def test_locked_config_validates_dimensions_and_seed_bank_separation():
    config = load_experiment_config(CONFIG)
    assert config.feature_dimensions == (8, 32, 128)
    assert config.actual_pca_components(1, 100) == 1
    assert config.actual_pca_components(2, 100) == 2
    assert config.actual_pca_components(128, 4000) == 3
    with pytest.raises(ContractError, match="seed collision"):
        load_experiment_config(CONFIG, reserved_seeds=[73116230])


def test_partition_contract_checks_stable_ids_and_shapes():
    config = load_experiment_config(CONFIG)
    x = np.ones((4, 2)); y = np.array([0, 1, 0, 1]); ids = np.array([50, 51, 52, 53])
    config.validate_partition(x, y, ids, name="calibration", feature_count=2,
                              allowed_labels=(0, 1))
    with pytest.raises(ContractError, match="IDs must be unique"):
        config.validate_partition(x, y, np.array([50, 50, 52, 53]),
                                  name="calibration", feature_count=2)
    with pytest.raises(ContractError, match="shape"):
        config.validate_partition(x, y, ids, name="calibration", feature_count=3)


def test_checkpoint_resume_verifies_config_source_and_committed_artifacts(tmp_path):
    out = tmp_path / "run"
    run = CheckpointedRun(out)
    run.initialize(config_digest="config-A", source_hashes={"source": "sha-A"},
                   expected_cells=["task-seed-1", "task-seed-2"])
    artifact = out / "cells" / "task-seed-1.json"
    atomic_write_json(artifact, {"result": 1})
    run.commit_cell("task-seed-1", artifact)
    run.fail(RuntimeError("synthetic interruption after a complete seed"))
    assert json.loads((out / "protocol_lock.json").read_text())["state"] == "failed_with_checkpoint"

    resumed = CheckpointedRun(out)
    resumed.initialize(config_digest="config-A", source_hashes={"source": "sha-A"},
                       expected_cells=["task-seed-1", "task-seed-2"], resume=True)
    assert set(resumed.lock["completed_cells"]) == {"task-seed-1"}
    assert resumed.lock["failure"] is None
    assert len(resumed.lock["failure_history"]) == 1
    artifact2 = out / "cells" / "task-seed-2.json"
    atomic_write_json(artifact2, {"result": 2})
    resumed.commit_cell("task-seed-2", artifact2)
    resumed.complete()
    assert json.loads((out / "protocol_lock.json").read_text())["state"] == "complete"
    with pytest.raises(ContractError, match="only interrupted/failed runs"):
        CheckpointedRun(out).initialize(config_digest="config-A", source_hashes={"source": "sha-A"},
                                        expected_cells=["task-seed-1", "task-seed-2"], resume=True)


def test_resume_rejects_changed_protocol_and_corrupt_checkpoint(tmp_path):
    out = tmp_path / "run"
    run = CheckpointedRun(out)
    run.initialize(config_digest="config-A", source_hashes={"source": "sha-A"},
                   expected_cells=["cell"])
    artifact = out / "cell.json"
    atomic_write_json(artifact, {"ok": True})
    run.commit_cell("cell", artifact)
    run.fail(RuntimeError("stop"))
    with pytest.raises(ContractError, match="config or source hashes differ"):
        CheckpointedRun(out).initialize(config_digest="config-B", source_hashes={"source": "sha-A"},
                                        expected_cells=["cell"], resume=True)
    artifact.write_text("corrupt", encoding="utf-8")
    with pytest.raises(ContractError, match="checksum"):
        CheckpointedRun(out).initialize(config_digest="config-A", source_hashes={"source": "sha-A"},
                                        expected_cells=["cell"], resume=True)
