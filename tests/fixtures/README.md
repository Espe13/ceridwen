# Test fixtures

This directory holds SSP grids committed via **git LFS** — run
`git lfs install && git lfs pull` after cloning, or the files here are tiny
pointer files and the tests that need them skip cleanly rather than erroring.

- `ssp_data_bpass_agb_dust.h5` — used by the BPASS+AGB regression test
  (`tests/test_losvd_no_lyman_spike.py`).

The suite's **main** test grid is not committed here: it is the published BPASS
grid. Fetch it once and the tests find it in the `fetch_grid` cache
(`$CERIDWEN_GRID_DIR` or `~/.ceridwen/grids`; checksum-verified, never downloaded
by the tests; see `tests/_gridfixture.py`):

```bash
python -c "from ceridwen.ssps import fetch_grid; fetch_grid('mist_bpass_v2')"
pytest
```

`fetch_grid('mist_miles_chab')` and `fetch_grid('amist_c3k_hr_krou_afe')` enable the
tests on those grids the same way. Before the cache, the tests look at
`$CERIDWEN_TEST_SSP` (any grid file; CI sets it), this directory and
`ceridwen/data/test_data/` (untracked developer grids).

All grids loaded by the strict schema-2.x loaders must carry the
`ssp_resolution` dataset — convert older files with
`scripts/convert_grids_schema2.py`. Note that every committed fixture version
permanently consumes GitHub LFS storage quota; do not commit
`*_schema1_backup.h5` files the converter leaves behind.

The surviving-mass tests (`tests/test_stellar_mass.py`, regression category
`stellar_mass`) do not need a schema-3 grid: they attach the FSPS mass table stored
in `tests/reference/ssp_stellar_mass.npz` to the canonical grid in memory, matched
by the grid's content hash, and skip for any other grid.
