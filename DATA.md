# Data setup

The experiment uses **BTG-OBD-A26**, listed in the
[BTG Alpha Lab dataset catalog](https://alphalab.btgpactual.com/datasets).
Obtain the data under the catalog's current access and use terms. The dataset
is not part of this software repository.

## Required subset and layout

The frozen protocol uses five tickers, 18 trading dates per ticker, depth 10,
and ten non-overlapping half-hour windows per session. Place the 90 input files
under the repository root:

```text
BTG-OBD-A26/<TICKER>/<YYYYMMDD>/d10/events.parquet
```

For example:

```text
BTG-OBD-A26/ASAI3/20251013/d10/events.parquet
BTG-OBD-A26/MGLU3/20260129/d10/events.parquet
```

[`configs/paper/ticker-books.yaml`](configs/paper/ticker-books.yaml) is the
authoritative record of the symbols, dates, clock windows, and whole-date
Calibration Book/Test Book split. Do not substitute dates or redistribute the
files through this repository.

You can check the expected file count before preparing empirical books:

```bash
find BTG-OBD-A26 -path '*/d10/events.parquet' -type f | wc -l
# expected: 90
```

## Derived empirical artifacts

The preparation phase reconstructs the books and writes targets and window
moments below the ignored `results/real/` tree:

```bash
python -m workflows.benchmark.reproduce_published_study --phase prepare
```

It then freezes the repository's provenance records. The workflow refuses to
silently overwrite an incompatible frozen record. Keep the same raw files and
clone path when resuming a run because the safety fingerprints include file
identity and resolved-path information.

There is no bundled checksum manifest for the source download. Preserve the
provider's original archive and access record if independent byte-level data
provenance is required.
