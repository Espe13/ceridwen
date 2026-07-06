# Installation

!!! warning "Use Python 3.11 or newer"
    Nested sampling depends on the official `blackjax` (its merged NSS), which
    requires Python ≥ 3.11. `pip install` will refuse 3.10.

A fresh conda environment is the easy route:

```bash
conda create -n ceridwen python=3.11 -y
conda activate ceridwen

git lfs install    # needs git-lfs: brew install git-lfs / apt-get install git-lfs
git clone https://github.com/Espe13/ceridwen.git
cd ceridwen
git lfs pull       # fetches the SSP test grid (~120 MB) used by the quickstart
pip install .
```

!!! warning "git-lfs required for the quickstart"
    The bundled SSP grids are stored with [git LFS](https://git-lfs.com). Without
    `git lfs install && git lfs pull` a clone contains tiny *pointer files*
    instead of the data, and `examples/quickstart.py` will stop with an
    explanatory message. If you cannot use LFS, download the grid from Zenodo
    instead (see "Getting the SSP grid" below).

Use a plain `pip install .` (not `-e`) unless you intend to edit the source — an
editable install tracks your working copy, so your version would move as the
repo does. To pin a specific release, install a tag directly:

```bash
pip install "git+https://github.com/Espe13/ceridwen.git@v0.1.0"
```

This pulls everything needed to import CERIDWEN, build the forward model, and run
NUTS / VI / nested sampling **including posterior plotting**: `jax`, `jaxlib`,
`numpy`, `scipy`, `matplotlib`, `h5py`, `astropy`, `sedpy-jax`,
`tensorflow-probability`, `blackjax`, `tqdm`, `optax`, and `anesthetic`. The only
thing not installed automatically is FSPS (see below).

There are no extras to choose — `pip install .` includes VI, nested-sampling
plotting, and the test runner. FSPS is installed separately (see below), because
it compiles Fortran and can't be a normal Python dependency. Building this
documentation site needs `pip install ".[docs]"` (maintainers only).

!!! note "blackjax"
    Nested sampling uses `blackjax.nss`, which is merged into the official
    blackjax but not yet in a tagged PyPI release, so CERIDWEN pins a fixed
    blackjax commit (`f73e12956`) — everyone installs the same validated state.
    This is why CERIDWEN installs from source/GitHub rather than PyPI for now;
    once a blackjax release ships NSS it becomes a normal version pin.

## Getting the SSP grid

Fitting needs a pre-computed SSP grid (an HDF5 file). Any one of these works —
the quickstart resolves them in the order `$SSP_FILE` →
`examples/ssp_data.h5` → the LFS test fixture:

1. **git LFS (default).** `git lfs pull` in the clone fetches
   `tests/fixtures/ssp_data_test.h5`, which the quickstart uses automatically.
   Zero extra setup.
2. **Zenodo download (no LFS needed).** Download the quickstart grid and place
   it at `examples/ssp_data.h5`:

    ```bash
    curl -L -o examples/ssp_data.h5 \
        "https://zenodo.org/records/ZENODO_RECORD_ID/files/ssp_data.h5?download=1"
    ```

    <!-- TODO(release): replace ZENODO_RECORD_ID after minting the Zenodo record -->

3. **Build your own with FSPS** (below) — required anyway for nebular emission
   and dust emission, which read the CLOUDY/Draine & Li data from `$SPS_HOME`.

## Verify your setup

After installing, run the environment doctor before your first fit:

```bash
python -m ceridwen.check
```

It prints an `ok` / `warn` / `FAIL` line per component (dependencies, FSPS,
`$SPS_HOME`, nested-sampling support) with the fix for anything missing.

## Installing FSPS and setting `$SPS_HOME`

CERIDWEN uses [FSPS](https://github.com/cconroy20/fsps) (via the
[`python-fsps`](https://dfm.io/python-fsps) wrapper) to build the SSP cache. The
FSPS **data files** also supply the CLOUDY nebular grids and Draine & Li
dust-emission templates: when `add_neb=True` or `add_dust_emission=True`,
CERIDWEN reads those files directly from `$SPS_HOME` (FSPS itself is not run at
fit time — it just provides the data).

FSPS is not a pure-Python wheel: it needs a Fortran compiler and a clone of the
FSPS data files.

```bash
# 1. A Fortran compiler (pick one for your system):
brew install gcc                       # macOS (Homebrew)
sudo apt-get install gfortran          # Debian/Ubuntu
conda install -c conda-forge gfortran  # any OS, inside your conda env

# 2. Pick where the FSPS data should live (ANY path: $HOME, a data disk, cluster
#    scratch, ...). git clone writes to the absolute $SPS_HOME path, so it does
#    not matter which directory you run it from.
export SPS_HOME="$HOME/fsps"           # <- edit to your chosen location
git clone https://github.com/cconroy20/fsps.git "$SPS_HOME"

# 3. Install the Python wrapper (it compiles against $SPS_HOME):
python -m pip install "fsps>=0.4.4"
```

!!! tip "Make `$SPS_HOME` permanent"
    `python-fsps` needs `$SPS_HOME` in every shell session and fails to import
    without it. Add it to your shell startup file (use the same path as above):

    ```bash
    echo 'export SPS_HOME="$HOME/fsps"' >> ~/.zshrc   # zsh (macOS default)
    echo 'export SPS_HOME="$HOME/fsps"' >> ~/.bashrc  # bash (most Linux)
    ```

    Open a new terminal and check `echo $SPS_HOME` prints the path, then run
    `python -m ceridwen.check`.
