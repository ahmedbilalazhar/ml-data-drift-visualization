# Rejected preprocessing intermediate

This directory is preserved for audit history and must not be used as evidence.

The initial parser mishandled NOAA sentinel tokens and retained only 272 rows. The defect was discovered during row-count and missingness validation. Its 272-row output and manifest are not used by any model, figure, metric, report, or package claim.

Use the corrected v2 dataset at `data/processed/weather_noaa_1949_1999_v2/`. The corrected parser treats the documented numeric missing sentinel as exactly 999.9, keeps valid sea-level pressure values above 999.9, requires the full station-year inventory, and records the output checksum.
