"""
Environment self-check for CERIDWEN.

Run it before your first fit to confirm every dependency and data file is in
place, with actionable messages for anything missing::

    python -m ceridwen.check

or from Python::

    import ceridwen
    ceridwen.check_environment()

The check is deliberately import-light and never raises: each probe is wrapped
so a broken component is reported, not crashed on. Exit code is 0 if all
*required* checks pass, 1 otherwise (handy for CI / setup scripts).
"""
from __future__ import annotations

import importlib
import os
import sys

# ANSI colours (no-op if output is not a TTY).
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
    """Probe the CERIDWEN runtime environment.

    Returns True if all *required* components are present, else False.
    Optional components (FSPS, anesthetic, optax) only emit warnings.
    """
    required_ok = True
    lines = []

    def record(status, name, detail=""):
        lines.append(f"  [{status}] {name}" + (f"  -- {detail}" if detail else ""))

    # ---- Python version --------------------------------------------------
    py = sys.version_info
    if py >= (3, 11):
        record(_OK, f"Python {py.major}.{py.minor}")
    else:
        record(
            _FAIL,
            f"Python {py.major}.{py.minor}",
            "CERIDWEN requires Python >= 3.11 (official blackjax nested sampling "
            "needs it). Create a 3.11+ environment and reinstall.",
        )
        required_ok = False

    # ---- Core required deps (incl. optax + anesthetic, now core) --------
    for mod in ("jax", "jaxlib", "numpy", "scipy", "matplotlib", "h5py",
                "astropy", "tqdm", "optax", "anesthetic"):
        m, err = _try_import(mod)
        if m is None:
            required_ok = False
            record(_FAIL, mod, f"not importable ({err}); run `pip install .`")
        else:
            record(_OK, mod, getattr(m, "__version__", ""))

    # ---- float64 enabled (CERIDWEN sets this on import) -----------------
    jax, _ = _try_import("jax")
    if jax is not None:
        x64 = bool(jax.config.read("jax_enable_x64"))
        if x64:
            record(_OK, "jax float64", "enabled")
        else:
            record(_WARN, "jax float64",
                   "disabled; `import ceridwen` enables it. Evidence/gradients "
                   "need it.")

    # ---- sedpy-jax (+ the inres fix) ------------------------------------
    sj, err = _try_import("sedpy_jax")
    if sj is None:
        required_ok = False
        record(_FAIL, "sedpy_jax",
               f"not importable ({err}); `pip install sedpy-jax>=0.1.1`")
    else:
        try:
            import inspect
            from sedpy_jax import smoothing
            has_inres = "inres" in inspect.signature(
                smoothing.make_lsf_smoother).parameters
            if has_inres:
                record(_OK, "sedpy_jax", "make_lsf_smoother has inres")
            else:
                required_ok = False
                record(_FAIL, "sedpy_jax",
                       "too old (no `inres` in make_lsf_smoother); "
                       "`pip install -U 'sedpy-jax>=0.1.1'`")
        except Exception as exc:  # pragma: no cover
            record(_WARN, "sedpy_jax", f"version probe failed: {exc}")

    # ---- tensorflow-probability (priors) --------------------------------
    tfp, err = _try_import("tensorflow_probability.substrates.jax")
    if tfp is None:
        required_ok = False
        record(_FAIL, "tensorflow-probability",
               f"jax substrate not importable ({err})")
    else:
        record(_OK, "tensorflow-probability", "jax substrate")

    # ---- blackjax (+ nested sampling) -----------------------------------
    bj, err = _try_import("blackjax")
    if bj is None:
        required_ok = False
        record(_FAIL, "blackjax", f"not importable ({err})")
    else:
        record(_OK, "blackjax", getattr(bj, "__version__", ""))
        ns, _ = _try_import("blackjax.ns")
        if ns is not None:
            record(_OK, "blackjax.ns", "nested sampling available")
        else:
            record(_WARN, "blackjax.ns",
                   "missing -> nested sampling unavailable. Install blackjax "
                   "with NSS: pip install "
                   "'git+https://github.com/blackjax-devs/blackjax@main'")

    # ---- FSPS + $SPS_HOME (grid building + nebular/dust-emission) -------
    fsps, _ = _try_import("fsps")
    if fsps is None:
        record(_WARN, "python-fsps",
               "not importable. Needed to build SSP grids and (with nebular / "
               "dust emission) at runtime. See README Installation.")
    else:
        record(_OK, "python-fsps", "")

    sps_home = os.environ.get("SPS_HOME")
    if not sps_home:
        record(_WARN, "$SPS_HOME",
               "unset. Required for nebular / dust-emission models. "
               "`export SPS_HOME=/path/to/fsps`")
    elif not os.path.isdir(sps_home):
        record(_WARN, "$SPS_HOME", f"set to {sps_home!r} but that directory does not exist")
    else:
        neb = os.path.join(sps_home, "nebular")
        if os.path.isdir(neb):
            record(_OK, "$SPS_HOME", sps_home)
        else:
            record(_WARN, "$SPS_HOME",
                   f"{sps_home} exists but has no nebular/ subdir; nebular "
                   "models will not find their CLOUDY grids.")

    if verbose:
        header = "CERIDWEN environment check"
        print(header)
        print("=" * len(header))
        print("\n".join(lines))
        print()
        if required_ok:
            print("All required components present."
                  " (warnings above are optional features / FSPS setup.)")
        else:
            print("Some REQUIRED components are missing -- see FAIL lines above.")
    return required_ok


def main() -> int:
    return 0 if check_environment() else 1


if __name__ == "__main__":
    raise SystemExit(main())
