"""Build a corrected, reviewed report edition without altering earlier PDFs."""
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports/phase3/decision_strengthening_v4"
OUT = ROOT / "reports/phase3/decision_strengthening_v5"


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    shutil.copytree(SOURCE, OUT)
    tex = OUT / "REPORT.tex"
    content = tex.read_text(encoding="utf-8")
    content = content.replace(
        "Visual inspection of this automatically compiled edition remains pending. Scientific validation and sample reproduction are recorded separately.",
        "This corrected edition incorporates the completed robustness replication and has been rendered and visually reviewed page by page. The validation record is included with the report assets.",
    )
    content = content.replace(
        "The audit extension tests equal quotas, prefix decisions unaffected by future scores, strict zero-delay release ordering and absence of unavailable feedback at long delay. All budget rows are checked. No full fresh reproduction of the earlier classifier experiments is claimed by this report. The long 1,500-seed-per-task robustness study is still a separate running job and has no final findings here. A larger number of seeds cannot repair calibration transfer, prove future forecasting, establish causal explanations, or replace a participant study.",
        "The audit extension passed equal-quota, future-score isolation, zero-delay ordering and long-delay availability checks. Its 1,500-seed-per-task robustness replication is complete: 36,000 model streams, 864,000 online rows and 864,000 evaluator rows, with no failed seeds. A fresh five-seed sample per task reproduced 2,880 online and 2,880 evaluator rows; categorical and count fields matched exactly, and floating fields differed by at most 1.11e-16 under the recorded tight tolerance. The initial bitwise comparison failure is retained in the continuation log. This report does not claim a fresh full rerun of the earlier classifier experiments. A larger number of seeds cannot repair calibration transfer, prove future forecasting, establish causal explanations, or replace a participant study.",
    )
    content = content.replace(
        "The evidence view labels unresolved estimates and current-cohort scope. Existing as-of release logic is retained; revised wording does not certify current calibration. Inspect the generated pages and dashboard before final submission. Original source documents and submitted archive remain preserved. New results and manuscript are prepared research improvements, not uploaded submissions.",
        "The evidence view labels unresolved estimates and current-cohort scope. Its release logic passed 75,600 boundary checks; this is not a browser usability study and does not certify current calibration. The supplementary package passed CRC and member-hash checks, five focused tests, clean-extraction reruns of the audit and horizon analyses, and exact comparisons of all ten generated result CSVs. The original Phase 2 DOCX, implementation brief and submitted archive remain preserved. New results and this manuscript are research improvements, not an uploaded replacement submission.",
    )
    content = content.replace(
        r"\section{Verified external references}\cite{garg2022,ovadia2019}\bibliographystyle{plain}\bibliography{REFERENCES}",
        "% References and direct source links are given in Section 6.",
    )
    content = content.replace(
        "The larger run retains the central failures: conditional-only changes produce no input alarms and concern flags miss harmful batches; the C6 dependence event alarms in 563/1,500 seeds; and the logistic model remains near chance on stationary XOR while the fixed tree performs better. These results sharpen the scope limits; they do not repair calibration under arbitrary change or demonstrate external validity.",
        "The larger run retains the central failures: conditional-only changes produce no input alarms and concern flags miss harmful batches; the C6 dependence event alarms in 563/1,500 seeds; and the logistic model remains near chance on stationary XOR while the fixed tree performs better. These results sharpen the scope limits; they do not repair calibration under arbitrary change or demonstrate external validity. The supplementary robustness report contains the detailed estimation-error and interval-coverage figure.",
    )
    content = content.replace(
        "\\begin{figure}[htbp]\\centering\n\\includegraphics[width=.97\\linewidth]{../large_seed_robustness_final/estimation_and_coverage.png}\n\\caption{Large-seed estimation and coverage}\n\\end{figure}\n\n",
        "",
    )
    tex.write_text(content, encoding="utf-8")

    md = OUT / "REPORT.md"
    markdown = md.read_text(encoding="utf-8")
    markdown = markdown.replace(
        "The long 1,500-seed-per-task robustness study is still a separate running job and has no final findings here.",
        "The completed 1,500-seed-per-task robustness replication and five-seed-per-task reproduction check are reported below; they do not repair calibration transfer or establish external validity.",
    )
    markdown = markdown.replace(
        "Inspect the generated pages and dashboard before final submission.",
        "The report PDF has been rendered and visually reviewed; the release-boundary checks do not constitute browser usability research.",
    )
    md.write_text(markdown, encoding="utf-8")
    bib = OUT / "REFERENCES.bib"
    if bib.exists():
        bib.unlink()
    print(OUT)


if __name__ == "__main__":
    main()
