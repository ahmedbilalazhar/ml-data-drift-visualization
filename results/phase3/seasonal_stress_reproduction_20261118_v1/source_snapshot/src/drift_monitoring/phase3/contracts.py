"""Validated experiment contracts, atomic artifacts, and restart-safe run state."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Mapping

import numpy as np


ALLOWED_STATES = {"created", "validated", "running", "complete", "failed_with_checkpoint"}
_TRANSITIONS = {
    "created": {"validated", "failed_with_checkpoint"},
    "validated": {"running", "failed_with_checkpoint"},
    "running": {"complete", "failed_with_checkpoint"},
    "failed_with_checkpoint": {"running", "failed_with_checkpoint"},
    "complete": set(),
}


class ContractError(ValueError):
    pass


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_hash(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_write_bytes(path: str | Path, payload: bytes) -> None:
    """Write within the destination directory, fsync, then atomically replace."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(path: str | Path, value) -> None:
    atomic_write_bytes(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def load_experiment_config(path: str | Path, *, reserved_seeds=()) -> "ExperimentConfig":
    source = Path(path)
    config = json.loads(source.read_text(encoding="utf-8"))
    return ExperimentConfig(config=config, source_path=source, reserved_seeds=set(map(int, reserved_seeds)))


@dataclass(frozen=True)
class ExperimentConfig:
    config: Mapping
    source_path: Path
    reserved_seeds: set[int]

    def __post_init__(self):
        c = self.config
        required = {"protocol_id", "phase2_basis", "tasks", "mechanisms", "feature_dimensions",
                    "history", "stream", "representations", "delays_batches", "seed_banks"}
        missing = required - set(c)
        if missing:
            raise ContractError(f"missing configuration keys: {sorted(missing)}")
        phase2 = c["phase2_basis"]
        digest = phase2.get("sha256", "")
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest.lower()):
            raise ContractError("Phase 2 basis must have a valid SHA-256")
        dims = [int(c["feature_dimensions"]["primary"]),
                *map(int, c["feature_dimensions"].get("targeted_sensitivity", []))]
        if any(d < 1 for d in dims) or len(dims) != len(set(dims)):
            raise ContractError("feature dimensions must be positive and unique")
        if not c["tasks"] or len(set(c["tasks"])) != len(c["tasks"]):
            raise ContractError("task list must be non-empty and unique")
        if not c["mechanisms"] or len(set(c["mechanisms"])) != len(c["mechanisms"]):
            raise ContractError("mechanism list must be non-empty and unique")
        history = c["history"]
        if any(not isinstance(history.get(k), int) or history[k] <= 0
               for k in ("train_rows", "calibration_rows", "reference_rows")):
            raise ContractError("historical partition sizes must be positive integers")
        stream = c["stream"]
        batches, size = stream.get("batches"), stream.get("batch_size")
        if not isinstance(batches, int) or batches < 1 or not isinstance(size, int) or size < 1:
            raise ContractError("batch count and size must be positive integers")
        if stream.get("intervention_start_batch", 1) <= stream.get("stationary_batches", 0):
            raise ContractError("intervention must start after the stationary prefix")
        if any(not isinstance(delay, int) or delay < 0 for delay in c["delays_batches"]):
            raise ContractError("delays must be non-negative integer batch counts")
        banks = c["seed_banks"]
        if set(banks) != {"development", "validation", "evaluation"}:
            raise ContractError("seed banks must have development, validation, and evaluation entries")
        normalized = {key: [int(seed) for seed in values] for key, values in banks.items()}
        if any(not values or len(values) != len(set(values)) for values in normalized.values()):
            raise ContractError("each seed bank must be non-empty and internally unique")
        names = list(normalized)
        for i, left in enumerate(names):
            for right in names[i + 1:]:
                if set(normalized[left]) & set(normalized[right]):
                    raise ContractError(f"seed banks overlap: {left} and {right}")
        overlap = set().union(*map(set, normalized.values())) & self.reserved_seeds
        if overlap:
            raise ContractError(f"seed collision with existing study manifests: {sorted(overlap)}")

    @property
    def digest(self) -> str:
        return canonical_hash(self.config)

    @property
    def feature_dimensions(self) -> tuple[int, ...]:
        f = self.config["feature_dimensions"]
        return (int(f["primary"]), *map(int, f.get("targeted_sensitivity", [])))

    def actual_pca_components(self, feature_count: int, training_rows: int) -> int:
        if feature_count < 1 or training_rows < 2:
            raise ContractError("PCA requires at least one feature and two training rows")
        requested = int(self.config["representations"]["pca_requested_components"])
        return min(requested, feature_count, training_rows - 1)

    def validate_partition(self, x, y, ids, *, name: str, feature_count: int,
                           allowed_labels=None) -> None:
        x, y, ids = np.asarray(x), np.asarray(y), np.asarray(ids)
        if x.ndim != 2 or x.shape[1] != feature_count or not len(x):
            raise ContractError(f"{name}: feature matrix must have shape (n,{feature_count})")
        if y.ndim != 1 or ids.ndim != 1 or len(y) != len(x) or len(ids) != len(x):
            raise ContractError(f"{name}: labels and stable row IDs must align with features")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ContractError(f"{name}: non-finite feature or label values")
        if len(np.unique(ids)) != len(ids):
            raise ContractError(f"{name}: row IDs must be unique")
        if allowed_labels is not None and not np.isin(y, allowed_labels).all():
            raise ContractError(f"{name}: labels are outside the declared class set")


class CheckpointedRun:
    """Small state machine for immutable per-cell artifacts and safe resume."""

    def __init__(self, output: str | Path):
        self.output = Path(output).resolve()
        self.lock_path = self.output / "protocol_lock.json"
        self.lock = None

    def initialize(self, *, config_digest: str, source_hashes: Mapping[str, str],
                   expected_cells, resume: bool = False):
        expected = sorted(map(str, expected_cells))
        if self.output.exists():
            if not resume:
                raise FileExistsError(f"run directory exists; set resume=True only for verified checkpoints: {self.output}")
            if not self.lock_path.is_file():
                raise ContractError("existing run has no protocol lock")
            lock = json.loads(self.lock_path.read_text(encoding="utf-8"))
            if lock.get("state") not in {"failed_with_checkpoint", "running"}:
                raise ContractError("only interrupted/failed runs may be resumed")
            if lock.get("config_sha256") != config_digest or lock.get("source_sha256") != dict(source_hashes):
                raise ContractError("resume refused: config or source hashes differ")
            if lock.get("expected_cells") != expected:
                raise ContractError("resume refused: expected experiment cell list differs")
            for cell, item in lock.get("completed_cells", {}).items():
                path = self.output / item["artifact"]
                if not path.is_file() or sha256_file(path) != item["sha256"]:
                    raise ContractError(f"resume refused: completed artifact failed checksum for {cell}")
            self.lock = lock
            # Preserve earlier failure details as history; do not leave a stale
            # active failure attached to a run that is being resumed.
            if self.lock.get("failure") is not None:
                self.lock.setdefault("failure_history", []).append(self.lock["failure"])
                self.lock["failure"] = None
            self.transition("running")
            return
        self.output.mkdir(parents=True)
        self.lock = {"state": "created", "config_sha256": config_digest,
                     "source_sha256": dict(source_hashes), "expected_cells": expected,
                     "completed_cells": {}, "failure": None}
        atomic_write_json(self.lock_path, self.lock)
        self.transition("validated")
        self.transition("running")

    def transition(self, state: str):
        if state not in ALLOWED_STATES or self.lock is None:
            raise ContractError("invalid run-state transition request")
        if state not in _TRANSITIONS[self.lock["state"]]:
            raise ContractError(f"illegal run-state transition {self.lock['state']} -> {state}")
        self.lock["state"] = state
        atomic_write_json(self.lock_path, self.lock)

    def commit_cell(self, cell: str, artifact: str | Path):
        if self.lock is None or self.lock["state"] != "running":
            raise ContractError("cell commits require a running study")
        cell = str(cell)
        if cell not in self.lock["expected_cells"]:
            raise ContractError(f"unexpected experiment cell: {cell}")
        if cell in self.lock["completed_cells"]:
            raise ContractError(f"experiment cell already committed: {cell}")
        path = Path(artifact).resolve()
        if not path.is_file() or not path.is_relative_to(self.output):
            raise ContractError("cell artifact must be a file inside this run directory")
        self.lock["completed_cells"][cell] = {
            "artifact": path.relative_to(self.output).as_posix(), "sha256": sha256_file(path)}
        atomic_write_json(self.lock_path, self.lock)

    def fail(self, error: BaseException):
        if self.lock is None:
            return
        if self.lock["state"] in {"running", "validated", "created", "failed_with_checkpoint"}:
            self.lock["failure"] = {"type": type(error).__name__, "message": str(error)}
            self.transition("failed_with_checkpoint")

    def complete(self):
        if self.lock is None or self.lock["state"] != "running":
            raise ContractError("only a running study can complete")
        missing = set(self.lock["expected_cells"]) - set(self.lock["completed_cells"])
        if missing:
            raise ContractError(f"cannot complete: cells are pending: {sorted(missing)}")
        if self.lock.get("failure") is not None:
            raise ContractError("cannot complete a run with an active failure")
        self.transition("complete")
