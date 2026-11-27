# IDM-Viz research paper (Phase 4)

**Template:** `IEEEtran` (journal mode, two-column). The citation style is IEEE (`IEEEtran.bst`).

## Files
| Path | Purpose |
|---|---|
| `main.tex` | Preamble and section includes |
| `sections/01_title.tex` ... `15_appendix.tex` | One file per required section, in the order of the brief |
| `references.bib` | 55 verified entries (see `REFERENCE_AUDIT.md`) |
| `figures/*.tex` | TikZ/pgfplots figures (vector) |
| `data/*.dat` | Plot data produced by the generator |
| `tables/*.tex` | Results tables produced by the generator |
| `scripts/build_tables.mjs` | Regenerates every data-bearing table and plot file from `../results` and `../reports` |
| `main.pdf` | Compiled paper |
| `CRITIQUE.md` | Revision-2 critique against the brief: each weakness and its fix |
| `EVIDENCE_LEDGER.md`, `REFERENCE_AUDIT.md`, `DISCREPANCIES.md`, `GAPS.md`, `SELF_REVIEW.md`, `COMPLIANCE_CHECKLIST.md`, `RECON.md`, `STORY.md` | Audit trail |

## Packages
`amsmath`, `amssymb`, `amsfonts`, `graphicx`, `booktabs`, `multirow`, `array`, `tabularx`, `xcolor`, `tikz` (positioning, arrows.meta, fit, backgrounds, shapes.geometric, calc), `pgfplots` 1.18 (groupplots, fillbetween), `algorithm`, `algpseudocode`, `cite`, `url`, `hyperref`; TikZ libraries also include shapes.misc and shadows, and pgfplots uses the colormaps library. All are available on TeX Live and Overleaf.

## Build
1. Regenerate the tables and plot data (requires Node.js 18 or later; reads only):
   ```powershell
   cd paper; node scripts/build_tables.mjs
   ```
2. Compile with any of the following:
   - Tectonic: `tectonic main.tex` (this is how `main.pdf` was produced).
   - latexmk: `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`.
   - Overleaf: upload the `paper/` folder (with `data/`, `tables/`, `figures/`, `sections/`), set `main.tex` as the main document and compile with pdfLaTeX. The files in `data/` must be uploaded, because the figures read them.

## Before submission (authors)
- Confirm the funding sentence in `sections/13_declarations.tex`.
- Read `GAPS.md` and `DISCREPANCIES.md`.
