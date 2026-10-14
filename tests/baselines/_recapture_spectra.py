"""Re-capture the spectrum / line-spectrum / maggies baselines of
tests/csp/test_lookback_flip_invariant.py with the CURRENT forward model.

The W_*.npy files are NOT touched: they were captured under the old
(decreasing-lookback) convention and are what the flip-invariance test is
about.  The spectral products only depend on W and on the broadening, so
they are re-captured whenever the broadening changes (2026-09-10: one
kinematic kernel, no CSP-side LOSVD, lines painted at the pixel floor).

    python tests/baselines/_recapture_spectra.py --only TAG [TAG ...] --reason "..."

Each configuration is built exactly as the test builds it (its ``_EXTRA``: the AGN set-up,
CSP options and theta of the AGN goldens). ``--only`` is required, so a re-capture names the
goldens that moved, and ``--reason`` is recorded per tag in manifest.json. Before 2026-10-14
this script ignored ``_EXTRA`` and would have re-captured the AGN goldens without the AGN.
"""
import argparse
import importlib.util
import json
import pathlib
import sys

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))                       # _gridfixture
spec = importlib.util.spec_from_file_location(
    "flip_test", HERE.parent / "csp" / "test_lookback_flip_invariant.py")
t = importlib.util.module_from_spec(spec); spec.loader.exec_module(t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", required=True, help="the tags to re-capture")
    ap.add_argument("--reason", required=True, help="why they moved (recorded in the manifest)")
    a = ap.parse_args()
    known = [c[3] for c in t.CONFIGS]
    unknown = sorted(set(a.only) - set(known))
    if unknown:
        raise SystemExit(f"unknown tags {unknown}; known: {known}")
    ssp = t.SSPData.load(t.SSP_FILE)
    m = json.load(open(HERE / "manifest.json"))
    for sfh_interp, zh_const, per_bin, tag in t.CONFIGS:
        if tag not in a.only:
            continue
        extra = t._EXTRA.get(tag, {})
        theta = t._build_theta_new(zh_const, per_bin)
        csp = t._build_csp(ssp, theta, sfh_interp=sfh_interp, zh_const=zh_const,
                           agn=extra.get("agn"), **extra.get("csp", {}))
        phot = t._photometry()
        th = {k: jnp.asarray(v) for k, v in csp.theta_init.items()}
        th.setdefault("logmass", jnp.zeros(1)); th.setdefault("zred", jnp.zeros(1))
        th.setdefault("igm_factor", jnp.ones(1)); th.setdefault("eline_scaling", jnp.ones(1))
        th.update({k: jnp.array([v]) for k, v in extra.get("theta", {}).items()})
        W = np.asarray(csp.calculate_ssp_weights(th))
        W_ref = np.load(HERE / f"W_{tag}.npy")
        np.testing.assert_allclose(W, W_ref, rtol=1e-12, atol=0)   # never re-captured
        spec_phot = csp._apply_mass_redshift_igm(*csp._assemble_observer_spectra(th), th)[0]
        np.save(HERE / f"spec_{tag}.npy", np.asarray(csp.get_spectrum(th)))
        np.save(HERE / f"lines_{tag}.npy", np.asarray(csp.get_line_spec(th)))
        np.save(HERE / f"maggies_{tag}.npy", np.asarray(phot.get_maggies(csp.wave, spec_phot)))
        m.setdefault("spectra_recaptured_by_tag", {})[tag] = a.reason
        print(f"{tag:<40} re-captured (W unchanged)")
    json.dump(m, open(HERE / "manifest.json", "w"), indent=2)


if __name__ == "__main__":
    main()
