# Decision — Weather dataset OMITTED (2026-09-28)

- Proposal in Phase 1: "Electricity/Weather" as natural context.
- Investigation: no canonical "Weather" concept-drift benchmark exists in the
  MOA/River/OpenML drift literature (closest name match, WeatherBench, is a
  medium-range forecasting benchmark — wrong task, no drift-monitoring protocol).
  No exact source, variant, or timestamp semantics could be established.
- Rule applied (roadmap Sec. 7.2): "If Weather's variant or chronology cannot be
  established, omit it rather than filling a dataset-count target with ambiguous
  evidence."
- Outcome: OMITTED. Tier B is covered by Elec2 (verified); no replacement with
  ambiguous evidence. Revisit only with an exact source + timestamp spec.
