# Test fixtures

This directory holds the SSP grids the test suite loads. Both are committed
via **git LFS** — run `git lfs install && git lfs pull` after cloning, or the
files here are tiny pointer files and the grid-dependent tests skip cleanly
(with a message pointing here) rather than erroring, so `import`/smoke CI
stays green.

- `ssp_data_test.h5` — the main test grid (BPASS build, provenance-tagged).
  Tests resolve it via `tests/_gridfixture.py`.
- `ssp_data_bpass_agb_dust.h5` — used by the BPASS+AGB regression test
  (`tests/test_losvd_no_lyman_spike.py`). Written before provenance tracking,
  so it also exercises the legacy-file path of `SSPData.load`.

You can point the suite at a different grid without touching the fixtures:

```bash
export CERIDWEN_TEST_SSP=/path/to/ssp_data.h5
pytest
```

To rebuild a fixture (needs FSPS): `SSPData.from_fsps(save_to=...)` on a
machine with the appropriate python-fsps build, then `git add` the file —
`.gitattributes` routes `tests/fixtures/*.h5` through LFS automatically.
Note that every committed fixture version permanently consumes GitHub LFS
storage quota, so regenerate only when the on-disk schema actually changes.
