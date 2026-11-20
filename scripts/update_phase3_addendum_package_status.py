"""Point the workspace package status record to the latest validated delta."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATUS = ROOT / "output/phase3/IDM_Viz_Phase3_Addendum_Package_Status_20261117.json"
RECEIPT = ROOT / "output/phase3/IDM_Viz_Phase3_ADWIN_Baseline_Delta_20261117_v6.receipt.json"


def main() -> None:
    status = json.loads(STATUS.read_text(encoding="utf-8"))
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    status["supplementary_delta"] = receipt["package"]
    status["supplementary_delta_receipt"] = RECEIPT.name
    status["supplementary_delta_sha256"] = receipt["sha256"]
    status["supplementary_delta_history"] = [
        "IDM_Viz_Phase3_Matched_Adaptation_Delta_20261117_v3.zip",
        "IDM_Viz_Phase3_Audit_Policy_Delta_20261117_v4.zip",
        "IDM_Viz_Phase3_Audit_Policy_Delta_20261117_v5.zip", receipt["package"]]
    status["note"] = (
        "Recommended base addendum v20 remains unchanged. Apply the latest ADWIN delta v6; it carries "
        "forward the prior audit-policy paired analysis and adds canonical River ADWIN, its delayed-budgeted "
        "run, tests, editable TeX/PDF report, and updated audit registry. CRC and overlay hashes passed. "
        "The full source suite passes 202 tests. Original Phase 2 and submission archives remain unchanged."
    )
    STATUS.write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Updated {STATUS}")


if __name__ == "__main__":
    main()
