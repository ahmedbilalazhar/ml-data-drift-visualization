# IDM-Viz research paper (Phase 4)

**Template:** `IEEEtran` (journal mode, two-column). The citation style is IEEE (`IEEEtran.bst`).

## Files
| Path | Purpose |
|---|---|
| `main.tex` | Preamble and section includes |
| `sections/01_title.tex` ... `15_appendix.tex` | One file per required section, in the order of the brief |
| `references.bib` | Bibliography shipped in the source ZIP |
| `figures/*.tex` | TikZ and pgfplots figures |
| `data/*.dat` | Bundled plot data |
| `tables/*.tex` | Bundled results tables |
| `scripts/build_tables.mjs` | Regenerates the tables and plot data from repository results |
| `data/ledger_generated.json` | Evidence ledger |

The [submitted PDF and original source ZIP](../docs/submissions/phase4/) are the reference copies. See the [evidence map](../EXPERIMENTS.md) for experiment links. The course source ZIP contains the LaTeX inputs and generated assets. Audit notes mentioned in the original source README are absent from that package.

## Packages
`amsmath`, `amssymb`, `amsfonts`, `graphicx`, `booktabs`, `multirow`, `array`, `tabularx`, `xcolor`, `tikz` (positioning, arrows.meta, fit, backgrounds, shapes.geometric, calc), `pgfplots` 1.18 (groupplots, fillbetween), `algorithm`, `algpseudocode`, `cite`, `url`, `hyperref`; TikZ libraries also include shapes.misc and shadows, and pgfplots uses the colormaps library. All are available on TeX Live and Overleaf.

## Build
1. Regenerate the tables and plot data (requires Node.js 18 or later; rewrites files under paper/data and paper/tables):
   ```powershell
   cd paper; node scripts/build_tables.mjs
   ```
2. Compile with any of the following:
   - Tectonic: `tectonic main.tex` (this is how `main.pdf` was produced).
   - latexmk: `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`.
   - Overleaf: upload the `paper/` folder (with `data/`, `tables/`, `figures/`, `sections/`), set `main.tex` as the main document and compile with pdfLaTeX. The files in `data/` must be uploaded, because the figures read them.
