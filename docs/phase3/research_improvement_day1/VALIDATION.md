# Day 1 validation record

Status is limited to checks actually performed in this session.

| Check | Outcome |
|---|---|
| Tectonic compiler | Tectonic 0.17.0 installed in the user Scoop environment and invoked successfully. |
| LaTeX report build | `REPORT.tex` compiled to `REPORT.pdf`; BibTeX completed with the Phase 2 methodology reference. Non-fatal underfull box and Fontconfig warnings remain. |
| PDF pages | All 3 pages rendered to PNG via bundled `pypdfium2` and visually inspected. Coordinated figure and table rendered; no clipping/overlap observed. |
| Phase 2 DOCX SHA-256 | Freshly matched validation receipt: `786d4f23d54d0d09edd5f94740198f43ca3b406eb38626ae754fe04fe26c5ef4`. |
| Submitted Phase 3 archive SHA-256 | Freshly matched validation receipt: `279d9867d1c75720a51df12caae2ac0b4d490641b83752e32642dcb87e1c8877`. |
| Other historical archive/document hashes | Freshly matched receipt values; full inventory is `HASHES.sha256`. |
| Source contract inspection | v1 prediction-before-release and as-of filtering, C6 unit marginal variance formula, adaptation persist-before-release ordering, and seed-level bootstrap inspected. |
| Full numerical rerun and tests | Not rerun in this Day 1 implementation session; prior outputs are identified as historical receipt claims. |
| Clean Day 1 archive extraction | Passed for `IDM_Viz_Phase3_Day1_Research_Improvement.zip`: extraction succeeded in a fresh directory; embedded core archive SHA-256 matched the preserved source; embedded PDF has 3 pages and extracted text contains both the methodology reference and references section. |

The validated archive is versioned as `IDM_Viz_Phase3_Day1_v1.1_Validated.zip`; the initial Day 1 archive is retained unchanged. The detailed machine-readable receipt is `PACKAGE_VALIDATION.json`.

The report log may show non-fatal fontconfig and underfull-box warnings. Earlier overfull hash/path lines were removed from the report text and wrapped paths use `\path`. Recheck after the final compile.
