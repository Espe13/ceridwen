"""Environment self-check (``python -m ceridwen.check``)."""
from __future__ import annotations

import importlib
import os
import sys

_TTY = sys.stdout.isatty()
_OK = "\033[32mok  \033[0m" if _TTY else "ok  "
_WARN = "\033[33mwarn\033[0m" if _TTY else "warn"
_FAIL = "\033[31mFAIL\033[0m" if _TTY else "FAIL"


def _try_import(modname):
    try:
        return importlib.import_module(modname), None
    except Exception as exc:  # pragma: no cover - message path
        return None, exc


def check_environment(verbose: bool = True) -> bool:
    """Return True if all required components are present; optional ones only warn."""
    required_ok = True
    lines = []

    def record(status, name, detail=""):
        lines.append(f"  [{status}] {name}" + (f"  -- {detail}" if detail else ""))

    py = sys.version_info
    if py >= (3, 11):
        record(_OK, f"Python {py.major}.{py.minor}")
    else:
        record(
            _FAIL,
            f"Python {py.major}.{py.minor}",
            "CERIDWEN requires Python >= 3.11 (blackjax >= 1.6 and jax >= 0.9 "
            "need it). Create a 3.11+ environment and reinstall.",
        )
        required_ok = False

    for mod in ("jax", "jaxlib", "numpy", "scipy", "matplotlib", "h5py",
                "astropy", "tqdm", "optax", "anesthetic"):
        m, err = _try_import(mod)
        if m is None:
            required_ok = False
            record(_FAIL, mod, f"not importable ({err}); reinstall ceridwen "
                               "(`pip install ceridwen`, or `pip install .` in a clone)")
        else:
            record(_OK, mod, getattr(m, "__version__", ""))

    jax, _ = _try_import("jax")
    if jax is not None:
        x64 = bool(jax.config.read("jax_enable_x64"))
        if x64:
            record(_OK, "jax float64", "enabled")
        else:
            record(_WARN, "jax float64",
                   "disabled; `import ceridwen` enables it. Evidence/gradients "
                   "need it.")

    # Filters and attenuation curves are part of ceridwen itself since v1.0.0
    # (vendored from sedpy-jax, which is no longer a dependency), so what is
    # worth checking is that the data files came along with the install.
    try:
        from ceridwen.observation.filters import list_available_filters
        from ceridwen.dust.attenuation_laws import ATTENUATION_LAWS
        n_filt, n_law = len(list_available_filters()), len(ATTENUATION_LAWS)
        if n_filt and n_law:
            record(_OK, "filters + attenuation",
                   f"{n_filt} filters, {n_law} attenuation laws")
        else:
            required_ok = False
            record(_FAIL, "filters + attenuation",
                   "package data missing; reinstall ceridwen")
    except Exception as e:                                    # noqa: BLE001
        required_ok = False
        record(_FAIL, "filters + attenuation", f"not importable ({e})")

    tfp, err = _try_import("tensorflow_probability.substrates.jax")
    if tfp is None:
        required_ok = False
        record(_FAIL, "tensorflow-probability",
               f"jax substrate not importable ({err})")
    else:
        record(_OK, "tensorflow-probability", "jax substrate")

    bj, err = _try_import("blackjax")
    if bj is None:
        required_ok = False
        record(_FAIL, "blackjax", f"not importable ({err}); pip install 'blackjax>=1.6'")
    else:
        record(_OK, "blackjax", getattr(bj, "__version__", ""))
        ns, _ = _try_import("blackjax.ns")
        if ns is not None:
            record(_OK, "blackjax.ns", "nested sampling available")
        else:
            record(_WARN, "blackjax.ns",
                   "missing -> nested sampling unavailable (this blackjax is "
                   "older than 1.6). Upgrade: pip install -U 'blackjax>=1.6'")

    fsps, _ = _try_import("fsps")
    if fsps is None:
        record(_WARN, "python-fsps",
               "not importable. Only needed to build your own SSP grid "
               "(SSPData.from_fsps); the published grids need no FSPS "
               "(ceridwen.ssps.fetch_grid).")
    else:
        record(_OK, "python-fsps", "")

    sps_home = os.environ.get("SPS_HOME")
    sps_ok = False
    if not sps_home:
        record(_WARN, "$SPS_HOME",
               "unset: nebular emission (add_neb) and dust emission "
               "(add_dust_emission) read FSPS's data files from it. No "
               "compiling needed: `git clone https://github.com/cconroy20/fsps` "
               "and `export SPS_HOME=/path/to/fsps`")
    elif not os.path.isdir(sps_home):
        record(_WARN, "$SPS_HOME", f"set to {sps_home!r} but that directory does not exist")
    else:
        # the files the forward model opens: CLOUDY grids (nebular/*.lines, *.cont),
        # dust-emission templates (dust/dustem/DL07_MW3.1_*.dat), the FSPS line list
        neb = os.path.join(sps_home, "nebular")
        dustem = os.path.join(sps_home, "dust", "dustem")
        emlines = os.path.join(sps_home, "data", "emlines_info.dat")
        neb_files = (sorted(f for f in os.listdir(neb) if f.endswith((".lines", ".cont")))
                     if os.path.isdir(neb) else [])
        dl07 = (sorted(f for f in os.listdir(dustem) if f.startswith("DL07_MW3.1_"))
                if os.path.isdir(dustem) else [])
        missing = ([] if neb_files else ["nebular/*.lines, *.cont (CLOUDY grids)"]) \
            + ([] if dl07 else ["dust/dustem/DL07_MW3.1_*.dat (dust emission)"]) \
            + ([] if os.path.isfile(emlines) else ["data/emlines_info.dat (line list)"])
        if not missing:
            record(_OK, "$SPS_HOME", f"{sps_home} ({len(neb_files)} CLOUDY files, "
                   f"{len(dl07)} DL07 templates, emlines_info.dat)")
            sps_ok = True
        else:
            record(_WARN, "$SPS_HOME",
                   f"{sps_home} lacks {'; '.join(missing)}: is it a full FSPS clone?")

    if verbose:
        header = "CERIDWEN environment check"
        print(header)
        print("=" * len(header))
        print("\n".join(lines))
        print()
        if required_ok and sps_ok:
            print("All components present: the FSPS data files that nebular and dust emission "
                  "read are in $SPS_HOME.")
        elif required_ok:
            print("Core installation OK: fits of stellar populations with the published "
                  "grids work.\nNOT yet available: nebular emission and dust emission "
                  "(need $SPS_HOME, see above).")
        else:
            print("Some REQUIRED components are missing -- see FAIL lines above.")
    return required_ok


def main() -> int:
    return 0 if check_environment() else 1


if __name__ == "__main__":
    raise SystemExit(main())
