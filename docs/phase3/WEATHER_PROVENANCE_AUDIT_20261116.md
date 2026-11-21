# Weather data admission decision — 16 November 2026

## Decision

Weather was **not admitted** as a Phase 3 natural stream. No processed weather stream or machine-readable source manifest is present in the workspace. The project dataset contract marks acquisition as pending, explicitly rejects OpenML weather v1 as a 14-row toy unrelated to the intended stream, and says the event ordering, label mapping, and source checksum must be verified before use.

## Checks performed

- Read `configs/datasets/weather.yaml` and `configs/phase3_strengthening_v3.json`.
- Checked the declared processed path `data/processed/weather_ordered.parquet` and manifest path `data/manifests/weather_manifest.json`; neither is materialized in the project.
- Confirmed the acquisition code is guarded to skip the rejected toy source; its presence does not establish that the primary MOA/USP candidate was downloaded or verified.
- Confirmed the project requires an ordered event identifier, binary rain/no-rain label construction, chronological train/calibration/test partitions, and checksum before admission.

## Admission requirements still outstanding

1. Record a stable primary source page and exact downloadable file.
2. Preserve the untouched download and compute its SHA-256.
3. Verify row chronology and any ties against the source documentation.
4. Document every feature conversion and the rain/no-rain label mapping.
5. Create the processed stream and manifest, then pass the project dataset audit.

Until these checks pass, no weather metric or weather-based transfer claim is reported. The controlled synthetic extension is not a replacement for a verified natural stream. This audit records unavailability; it does not claim that the dataset itself is invalid.
