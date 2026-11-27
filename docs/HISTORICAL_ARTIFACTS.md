# Historical artifacts and reproduction inputs

The [submission index](../submissions/) lists the retained coursework files. The active [Phase 4 evidence map](../EXPERIMENTS.md) links the paper to its research implementation and results.

The cleanup moved superseded packages, draft variants, and scratch material to a dated DAV archive adjacent to the local checkout. Its MANIFEST.jsonl records original paths, sizes, checksums, and whether each archived file was tracked. For tracked items, the repository history also preserves the previous tree at commit 19e049f47e6234270aaf5a0b01dc30492cb7985a.

Historical Phase 3 packaging scripts may expect the original layout under output/phase3. To rerun a package recipe, use a separate checkout of the historical commit and restore any missing local files from the archive to their original paths. Use the recorded checksums before comparing a regenerated package to its receipt.

The locked Phase 3 v2 runner also hashes the original Phase 2 Word input, the Phase 3 brief, and both Phase 3 v1 ZIPs. Keep these at the paths described in the [reproduction guide](phase3/23I-2581_23I-2604_Phase3_Reproduction.md). The v1 full reproduction ZIP exceeds GitHub's regular Git file limit and is a local input. A fresh checkout needs that exact ZIP from the author to run the locked wrapper. The committed evidence and Phase 4 table generator are available independently of that ZIP.
