"""Add the validated delayed-audit policy matrix to the Phase 3 audit registry."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "docs/phase3/PHASE3_IMPLEMENTATION_SPEC_AUDIT_20261117.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    requirement = next(r for r in registry["requirements"] if r["requirement_id"] == "AUDIT-01")
    requirement.update({
        "promised_behavior": "Strict cap and exact quota; full declared policy family, budget, and delay matrix; force-fills and unused budgets disclosed.",
        "implemented_behavior": "A locked outcome-informed replay evaluates six policies across strict-cap and exact-quota modes, budgets 2/4/8, delays 0/5/20/100, seven conditions, and 30 evaluation seeds. Decisions use current cached predictors and only previously selected labels satisfying source_batch + delay < decision_clock. Seed-level paired uncertainty averages matched conditions and budgets before resampling.",
        "evidence_paths": [
            "scripts/run_audit_policy_matrix.py",
            "results/phase3/audit_policy_matrix_20261117_v1/",
            "reports/phase3/audit_policy_matrix_20261117_v7/",
            "tests/phase3/test_audit_policy_matrix_contract.py",
        ],
        "status": "evaluated",
        "amendment_reason": "The prior limited policy set is expanded into the declared full Cartesian matrix; the new run is additive and does not overwrite earlier analyses.",
        "allowed_claim": "The selected policies are compared on the preserved 30-seed synthetic evaluation under the explicitly logged budgets and delays. Delay 100 cannot verify labels within the 24-batch horizon. Strict-cap episode-yield differences are not equal-spend rankings because acquisition counts differ.",
        "unresolved_limitation": "This is outcome-informed replay on cached synthetic predictions, not fresh independent confirmation, prospective deployment, acquisition-cost validation, or evidence of analyst benefit. The tested policy family is not every Phase 2 policy concept.",
    })
    baseline = next(r for r in registry["requirements"] if r["requirement_id"] == "BASE-01")
    baseline.update({
        "implemented_behavior": "The source-to-implementation audit still distinguishes DriftLens and Domino from internal controls. Canonical River 0.27.0 ADWIN is now run on row-level errors from selected 500-row cohorts, under the same seed bank, delays, budgets, bootstrap, and cost modes as the locked policy matrix.",
        "evidence_paths": [
            "docs/phase3/BASELINE_FIDELITY_AUDIT_20261117.md",
            "scripts/run_adwin_audit_baseline.py",
            "results/phase3/adwin_audit_baseline_20261117_v1/",
            "reports/phase3/adwin_audit_baseline_20261117_v3/",
        ],
        "status": "substituted",
        "amendment_reason": "Canonical ADWIN has been added as a named method baseline; the named DriftLens and Domino implementations remain unreproduced and are not conflated with internal controls.",
        "allowed_claim": "River ADWIN was evaluated under the logged synthetic cohort-feedback protocol. Under strict caps it made no detections after the single bootstrap audit; exact-quota detections are constrained to the tested synthetic streams and do not establish general detector performance.",
        "unresolved_limitation": "DriftLens and Domino are still not reproduced; the ADWIN evaluation reuses one cached synthetic evaluation, is sparse under strict caps, and does not establish performance on natural streams or equal analyst/acquisition value.",
    })
    registry["completion_summary"]["computational_evaluation_completion"] = (
        "partial; the declared audit-policy Cartesian matrix and several other extensions are evaluated, "
        "but specified domains, policies, natural-stream confirmations, and human-use evidence remain incomplete"
    )
    artifacts = registry.setdefault("artifacts", [])
    files = [
        "scripts/run_audit_policy_matrix.py", "scripts/report_audit_policy_matrix.py",
        "scripts/update_audit_policy_matrix_registry.py",
        "scripts/package_phase3_audit_policy_delta.py",
        "scripts/package_phase3_audit_policy_delta_v5.py",
        "scripts/package_phase3_adwin_delta_v6.py",
        "scripts/update_phase3_addendum_package_status.py",
        "scripts/run_adwin_audit_baseline.py", "scripts/report_adwin_audit_baseline.py",
        "tests/phase3/test_audit_policy_matrix_contract.py",
        "tests/phase3/test_adwin_audit_baseline_contract.py",
        "pyproject.toml",
        "results/phase3/audit_policy_matrix_20261117_v1/policy_per_seed.csv",
        "results/phase3/audit_policy_matrix_20261117_v1/online_audit_decisions.csv",
        "results/phase3/audit_policy_matrix_20261117_v1/policy_summary.csv",
        "results/phase3/audit_policy_matrix_20261117_v1/paired_seed_bootstrap.csv",
        "results/phase3/audit_policy_matrix_20261117_v1/protocol_lock.json",
        "reports/phase3/audit_policy_matrix_20261117_v7/audit_policy_matrix.tex",
        "reports/phase3/audit_policy_matrix_20261117_v7/audit_policy_matrix.pdf",
        "reports/phase3/audit_policy_matrix_20261117_v7/audit_policy_effects.png",
        "reports/phase3/audit_policy_matrix_20261117_v7/audit_policy_effects.svg",
        "reports/phase3/audit_policy_matrix_20261117_v7/paired_overall_effects.csv",
        "reports/phase3/audit_policy_matrix_20261117_v7/policy_acquisition_summary.csv",
        "reports/phase3/audit_policy_matrix_20261117_v7/report_receipt.json",
        "results/phase3/adwin_audit_baseline_20261117_v1/adwin_per_seed.csv",
        "results/phase3/adwin_audit_baseline_20261117_v1/adwin_decisions.csv",
        "results/phase3/adwin_audit_baseline_20261117_v1/adwin_summary.csv",
        "results/phase3/adwin_audit_baseline_20261117_v1/paired_comparisons.csv",
        "results/phase3/adwin_audit_baseline_20261117_v1/protocol_lock.json",
        "reports/phase3/adwin_audit_baseline_20261117_v3/adwin_baseline.tex",
        "reports/phase3/adwin_audit_baseline_20261117_v3/adwin_baseline.pdf",
        "reports/phase3/adwin_audit_baseline_20261117_v3/adwin_effects.png",
        "reports/phase3/adwin_audit_baseline_20261117_v3/adwin_effects.pdf",
        "reports/phase3/adwin_audit_baseline_20261117_v3/adwin_effects.svg",
        "reports/phase3/adwin_audit_baseline_20261117_v3/adwin_paired_aggregate_effects.csv",
        "reports/phase3/adwin_audit_baseline_20261117_v3/adwin_delay_summary.csv",
        "reports/phase3/adwin_audit_baseline_20261117_v3/adwin_detection_summary.csv",
        "reports/phase3/adwin_audit_baseline_20261117_v3/report_receipt.json",
        "docs/phase3/BASELINE_FIDELITY_AUDIT_20261117.md",
        "docs/phase3/PHASE3_IMPLEMENTATION_SPEC_AUDIT_20261117.md",
    ]
    by_path = {a["path"]: a for a in artifacts}
    for rel in files:
        path = ROOT / rel
        if not path.is_file():
            raise FileNotFoundError(path)
        by_path[rel] = {"path": rel, "sha256": sha(path), "bytes": path.stat().st_size,
                        "role": "implementation_or_evaluation_evidence"}
    registry["artifacts"] = sorted(by_path.values(), key=lambda a: a["path"])
    REGISTRY.write_text(json.dumps(registry, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Updated {REGISTRY}; AUDIT-01 has {len(files)} verified evidence artifacts")


if __name__ == "__main__":
    main()
