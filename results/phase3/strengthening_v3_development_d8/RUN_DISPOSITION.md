# Disposition of this partial run

This run contains 57 of 120 cells and is intentionally marked `failed_with_checkpoint`. The scoring protocol changed to add historical-only pairwise-correlation alarms before completion. Its cells do not contain that detector and therefore cannot satisfy the current locked protocol. Preserve the checkpoint for audit; do not analyze it as the v3 result or resume it. The complete run uses a new output directory and code/config snapshots.
