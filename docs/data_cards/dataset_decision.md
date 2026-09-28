# Dataset decision (roadmap Sec. 7) — E0 pilot

| Tier | Choice | Role | Status |
|---|---|---|---|
| A controlled | Transparent generator S0–S7 | Exact event/feature truth for E1/E2 | IMPLEMENTED (E0 pilot rows) |
| B natural temporal | Electricity/Elec2 candidate | Chronological external validity | NOT admitted — needs download/license/chronology audit (data card required) |
| C semi-synthetic | Covertype + documented replay + interventions | Multiclass + realistic features, known interventions | NOT admitted — needs subset manifest + provenance audit |
| D optional | DDE high-dim set | Selective monitoring at 100–200 feats | GATED on core completion |

No natural claim is made from synthetic runs. Weather omitted until source/
timestamps verified. Every future admission needs a data card (source, version,
license, checksum, row-order, split ranges) + leakage check by a second reviewer.
