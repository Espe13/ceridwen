# Installation

!!! warning "Requires Python 3.11"
    Nested sampling depends on the official `blackjax` (its merged NSS), which
    requires Python 3.11. Create the environment with exactly that version.

A fresh conda environment is the easy route:

```bash
conda create -n ceridwen python=3.11 -y
conda activate ceridwen

git clone https://github.com/Espe13/ceridwen.git
cd ceridwen
pip install .
```

This pulls everything needed to import CERIDWEN, build the forward model, and run
NUTS / VI / nested sampling **including posterior plotting**: `jax`, `jaxlib`,
`numpy`, `scipy`, `matplotlib`, `h5py`, `astropy`, `sedpy-jax`,
`tensorflow-probability`, `blackjax`, `tqdm`, `fastprogress`, `optax`,
`anesthetic`, and `pytest`. The only
thing not installed automatically is FSPS (see below).

There are no extras to choose. `pip install .` includes VI, nested-sampling
plotting, and the test runner. FSPS is installed separately (see below), because
it compiles Fortran and cannot be a normal Python dependency. Building this
documentation site needs `pip install ".[docs]"` (maintainers only).

!!! note "blackjax"
    Nested sampling uses `blackjax.nss`, which is merged into the official
    blackjax but not yet in a tagged PyPI release, so CERIDWEN pins a fixed
    blackjax commit (`f73e12956`), so everyone installs the same validated state.
    This is why CERIDWEN installs from source/GitHub rather than PyPI for now;
    once a blackjax release ships NSS it becomes a normal version pin.

## Getting the SSP grid

Fitting needs a pre-computed SSP grid (an HDF5 file). The quickstart resolves
it in the order `$SSP_FILE`, then `examples/ssp_data.h5`, then a local
developer grid at `ceridwen/data/test_data/ssp_data_bpass.h5` (not shipped
in the repository).

1. **Build your own with FSPS (recommended for custom choices).** Install
   FSPS (below) and let the quickstart build the grid on first run, or call
   `SSPData.from_fsps(save_to="examples/ssp_data.h5", imf_type=1)` directly.
   You control the isochrones, spectral library, and IMF. FSPS is needed
   anyway for nebular and dust emission, which read the CLOUDY and Draine & Li
   data from `$SPS_HOME`.
2. **Download from Zenodo (no FSPS needed):**
   [doi:10.5281/zenodo.21221634](https://doi.org/10.5281/zenodo.21221634).
   The canonical grids are registered in `ceridwen.ssps.grid_fetch`, so the
   easiest route is by name — downloaded once into `~/.ceridwen/grids`
   (override with `$CERIDWEN_GRID_DIR`) and verified against a pinned
   SHA-256 on every load:

    ```python
    from ceridwen.ssps import fetch_grid, available_grids, SSPData

    print(available_grids())                       # name -> description
    ssp = SSPData.load(fetch_grid("mist_miles_chab_v3.2"))
    ```

    or by hand, e.g. for the quickstart location:

    ```bash
    curl -L -o examples/ssp_data.h5 \
        "https://zenodo.org/records/21221634/files/ssp_data.h5?download=1"
    ```

## α-enhanced grids: download, don't build

The [α/Fe]-aware grids for `CSPBasis_afe` are a special case, in both
directions:

- **Building them yourself is hard** — it requires python-fsps compiled from
  source with `AFE_FLAG=1` against the FSPS v4.0 data tree (aMIST isochrones
  + C3K spectra), an easy source of silent misbuilds.
- **Downloading them is all you need** — `CSPBasis_afe` carries **no nebular
  model** (no α-enhanced CLOUDY tables exist), so nothing is read from
  `$SPS_HOME` at fit time. With the downloaded grid, fitting [α/Fe] requires
  **no FSPS install at all**: skip the whole FSPS section below.

```python
from ceridwen.ssps import fetch_grid, SSPDataAfe
from ceridwen.csp import CSPBasis_afe
import jax.numpy as jnp

path = fetch_grid("amist_c3k_lr_chab_afe")     # cached + checksummed
ssp  = SSPDataAfe.load(path)                   # (n_afe, n_Z, n_age, n_wave)
csp  = CSPBasis_afe(ssp, lookback_time=jnp.linspace(0.0, 12.0, 9),
                    zh_const=True, verbose=False)
```

`CSPBasis_afe` accepts only α-aware (4-D) grids; passing a solar-scaled 3-D
grid raises a `TypeError` pointing you back to `CSPBasis`. Conversely the
nebular and dust-emission switches of `CSPBasis` still need `$SPS_HOME`, so
solar-scaled fits with emission keep using the FSPS data files as before.

## Installing FSPS and setting `$SPS_HOME`

CERIDWEN uses [FSPS](https://github.com/cconroy20/fsps) (via the
[`python-fsps`](https://dfm.io/python-fsps) wrapper) to build the SSP cache. The
FSPS **data files** also supply the CLOUDY nebular grids and Draine & Li
dust-emission templates: when `add_neb=True` or `add_dust_emission=True`,
CERIDWEN reads those files directly from `$SPS_HOME` (FSPS itself is not run at
fit time; it just provides the data).

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

    Open a new terminal and check `echo $SPS_HOME` prints the path.

## Verify your setup

With FSPS installed and `$SPS_HOME` set, run the environment doctor before your
first fit:

```bash
python -m ceridwen.check
```

It prints an `ok` / `warn` / `FAIL` line per component (dependencies, FSPS,
`$SPS_HOME`, nested-sampling support) with the fix for anything missing. Run it
only after FSPS is set up. Before that it will correctly report python-fsps as
missing.
