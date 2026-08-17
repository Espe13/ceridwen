# Test fixtures

This directory holds SSP grids committed via **git LFS** — run
`git lfs install && git lfs pull` after cloning, or the files here are tiny
pointer files and the tests that need them skip cleanly rather than erroring.

- `ssp_data_bpass_agb_dust.h5` — used by the BPASS+AGB regression test
  (`tests/test_losvd_no_lyman_spike.py`).

The suite's **main** test grid is no longer committed here: the former LFS
fixture `ssp_data_test.h5` was retired (2026-08-17). Tests resolve the local
developer grid `ceridwen/data/test_data/ssp_data_bpass.h5` instead (untracked;
see `tests/_gridfixture.py`). On a machine without it, either build one
(`python scripts/make_test_fixture_grid.py`, needs a BPASS python-fsps), or
download the BPASS release grid from Zenodo
([doi:10.5281/zenodo.21221634](https://doi.org/10.5281/zenodo.21221634)) and
place it at that path (convert with `scripts/convert_grids_schema2.py` if it
predates schema 2.0), or point the suite anywhere:

```bash
export CERIDWEN_TEST_SSP=/path/to/ssp_data.h5
pytest
```

All grids loaded by the strict schema-2.x loaders must carry the
`ssp_resolution` dataset — convert older files with
`scripts/convert_grids_schema2.py`. Note that every committed fixture version
permanently consumes GitHub LFS storage quota; do not commit
`*_schema1_backup.h5` files the converter leaves behind.
