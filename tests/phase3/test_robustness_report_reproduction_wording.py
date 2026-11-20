from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/phase3/large_seed_robustness_final_reprowording_v1"


def test_robustness_report_states_discrete_exact_and_float_tolerance_contract():
    markdown = (REPORT / "23I-2581_23I-2604_Phase3_Robustness_Report.md").read_text(encoding="utf-8")
    assert "Discrete fields matched exactly" in markdown
    assert "passed allclose (rtol=1e-12, atol=1e-14" in markdown
    assert "max absolute difference 1.11e-16" in markdown
    assert "fields across 14 fields exactly" not in markdown
    record = (REPORT / "pdf_render_record.json").read_text(encoding="utf-8")
    assert '"pages": 4' in record
