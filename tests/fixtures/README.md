# Test fixtures

This directory holds the small SSP grid the test suite loads.

`ssp_data_test.h5` is **not** in the repository yet — build it once (on a
machine with FSPS) and commit it:

```bash
python scripts/make_test_fixture_grid.py
git add tests/fixtures/ssp_data_test.h5
git commit -m "Add committed test SSP fixture grid"
```

Tests resolve the grid via `tests/_gridfixture.py`. If the file is absent, the
grid-dependent tests skip cleanly (with a message pointing here) rather than
erroring, so `import`/smoke CI stays green.

You can also point the suite at an existing grid without committing one:

```bash
export CERIDWEN_TEST_SSP=/path/to/ssp_data.h5
pytest
```

The BPASS+AGB regression test (`tests/test_losvd_no_lyman_spike.py`) needs a
separate grid named `ssp_data_bpass_agb_dust.h5`; commit that here too if you
want that test to run on fresh clones.
