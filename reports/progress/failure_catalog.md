# Failure catalog (roadmap Sec. 9.6) — sampling policy: ALL pilot/engineering
# failures retained (n small); confirmatory phase will use prespecified case sampling.

## Engineering (fixed, provenance kept)
1. Covertype figshare mirror 403 → UCI direct (docs/incidents/001_*).
2. `fetch_openml` has no `download_if_missing` kwarg → removed (TypeError, immediate).
3. pandas int64 shift overflow on Covertype intervention → float cast before shift.
4. Resume wrote partial evidence (drill finding) → incremental persist v2 + merge test.

## Scientific (reported as results, never tuned away)
5. S7 conditional-only harm invisible to KS/Wasserstein (rate 0.00, dF1 0.19).
6. S7@d5 delay hides harm → proxy says `monitor` (false reassurance).
7. Synthetic threshold saturates on real streams (Elec2 54/54, Covertype 40/40 pre-onset).
8. Reference refresh halves score levels yet alerts persist (per-(stream,policy) calibration required).
9. S0 1/12 pilot false alert (small-sample noise → full calibration outstanding).
10. KS ≡ Wasserstein on large pilot shifts (discriminating severities needed).
11. S6 pilot rotation detected by marginal KS (strong rotation; subtler joint cases pending).
