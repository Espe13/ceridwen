#!/usr/bin/env python3
"""verify_library_resolution.py — gate for the automatic library-resolution
handling in the Spectrum projection (schema 2.0, inres="auto").

Run AFTER applying the library-resolution patches and BEFORE refitting any
real spectrum.  Requires jax + the ceridwen environment (Mac or Tursa):

    python verify_library_resolution.py

Checks
------
1. STRICTNESS: a smoothing-enabled Spectrum with the default
   ``inres="auto"`` and no library curve RAISES at setup (release
   behaviour: the subtraction can no longer be silently skipped).
2. NO-SUBTRACTION EQUIVALENCE: an all-NaN curve (resolution unknown
   everywhere, with a warning) matches explicit ``inres=0.0`` exactly.
3. CONSTANT-CURVE EQUIVALENCE: a constant sigma_v curve through the
   automatic path matches the explicit scalar-inres fast path.
4. WIDTH RECOVERY (the actual fix): input lines carrying the
   wavelength-DEPENDENT library width, smoothed to target sigma_t, come
   out with width ~ sigma_t (library removed in quadrature), NOT
   sqrt(sigma_t^2 + sigma_lib^2) (the pre-fix over-smoothing).
5. FLOOR: where sigma_lib > sigma_t the output stays at library width
   (no deconvolution), with a warning.
6. REDSHIFT INVARIANCE: the same rest-frame curve gives the same
   effective subtraction at z = 0 and z = 1 (velocity units are
   frame-invariant).
"""
import sys
import warnings
import numpy as np

import jax
jax.config.update("jax_enable_x64", True)

CKMS = 2.998e5
FAILURES = []


def check(name, ok, detail=""):
    tag = "ok  " if ok else "FAIL"
    print(f"  [{tag}] {name}  {detail}")
    if not ok:
        FAILURES.append(name)


def gaussian_line(wave, lam0, sigma_v):
    sig_l = lam0 * sigma_v / CKMS
    return np.exp(-0.5 * ((wave - lam0) / sig_l) ** 2)


def measured_sigma_v(wave, spec, lam0):
    w = np.asarray(spec, float)
    w = np.clip(w - np.median(w), 0, None)
    m0 = np.trapezoid(w, wave)
    mu = np.trapezoid(w * wave, wave) / m0
    var = np.trapezoid(w * (wave - mu) ** 2, wave) / m0
    return CKMS * np.sqrt(var) / lam0


def main():
    from ceridwen.observation.observation import Spectrum

    wave_rest = np.exp(np.linspace(np.log(3000.0), np.log(9000.0), 6000))
    wave_obs = wave_rest.copy()
    SIG_T = 180.0
    sig_lib = np.interp(wave_rest, [3000.0, 9000.0], [40.0, 120.0])

    def make_spec(wavelength=wave_obs, **kw):
        return Spectrum(wavelength=wavelength,
                        flux=np.ones_like(wavelength),
                        uncertainty=np.ones_like(wavelength) * 0.01,
                        name="v", resolution=SIG_T, smoothtype="vel", **kw)

    x = gaussian_line(wave_rest, 5000.0, 30.0)

    # ---- 1. strictness ---------------------------------------------------
    print("1. strictness (auto + smoothing + no curve raises)")
    try:
        make_spec().setup_for_model(wave_rest, zred=0.0)
        check("raises without curve", False, "no exception raised")
    except ValueError as e:
        check("raises without curve", "lib_resolution" in str(e))

    # ---- 2. all-NaN curve == explicit inres=0 ---------------------------
    print("2. unknown-resolution equivalence")
    s0 = make_spec(inres=0.0)
    s0.setup_for_model(wave_rest, zred=0.0)
    s_nan = make_spec()
    with warnings.catch_warnings(record=True) as wlog:
        warnings.simplefilter("always")
        s_nan.setup_for_model(wave_rest, zred=0.0,
                              lib_resolution=(wave_rest,
                                              np.full_like(wave_rest, np.nan)))
    a = np.asarray(s0.predict(x, wave_rest))
    b = np.asarray(s_nan.predict(x, wave_rest))
    check("all-NaN curve == inres=0 exactly",
          np.array_equal(a, b), f"max|d|={np.max(np.abs(a - b)):.2e}")
    check("NaN-coverage warning emitted",
          any("unknown" in str(w.message) for w in wlog))

    # ---- 3. constant curve == scalar inres -------------------------------
    print("3. constant-curve equivalence")
    s_sc = make_spec(inres=80.0)
    s_sc.setup_for_model(wave_rest, zred=0.0)
    s_ct = make_spec()
    s_ct.setup_for_model(wave_rest, zred=0.0,
                         lib_resolution=(wave_rest,
                                         np.full_like(wave_rest, 80.0)))
    a = np.asarray(s_sc.predict(x, wave_rest))
    b = np.asarray(s_ct.predict(x, wave_rest))
    check("constant curve == scalar inres",
          np.allclose(a, b, rtol=1e-10, atol=1e-12),
          f"max|d|={np.max(np.abs(a - b)):.2e}")

    # ---- 4. width recovery (wavelength-dependent curve) ------------------
    print("4. width recovery (the actual fix)")
    s_auto = make_spec()
    s_auto.setup_for_model(wave_rest, zred=0.0,
                           lib_resolution=(wave_rest, sig_lib))
    for lam0 in (3600.0, 5000.0, 8000.0):
        slib = float(np.interp(lam0, wave_rest, sig_lib))
        y = gaussian_line(wave_rest, lam0, slib)   # line at LIBRARY width
        out = np.asarray(s_auto.predict(y, wave_rest))
        w = measured_sigma_v(wave_obs, out, lam0)
        w_wrong = np.hypot(SIG_T, slib)
        check(f"lam={lam0:.0f}: width {w:.1f} ~ {SIG_T:.0f} "
              f"(pre-fix: {w_wrong:.1f})", abs(w - SIG_T) / SIG_T < 0.05)

    # ---- 5. floor --------------------------------------------------------
    print("5. floor (library coarser than target)")
    s_fl = make_spec()
    with warnings.catch_warnings(record=True) as wlog:
        warnings.simplefilter("always")
        s_fl.setup_for_model(wave_rest, zred=0.0,
                             lib_resolution=(wave_rest,
                                             np.full_like(wave_rest, 300.0)))
    y = gaussian_line(wave_rest, 5000.0, 300.0)
    out = np.asarray(s_fl.predict(y, wave_rest))
    w = measured_sigma_v(wave_obs, out, 5000.0)
    check(f"output stays ~300 km/s (got {w:.1f})",
          abs(w - 300.0) / 300.0 < 0.05)
    check("floor warning emitted",
          any("LIBRARY resolution" in str(wm.message) for wm in wlog))

    # ---- 6. redshift invariance ------------------------------------------
    print("6. redshift invariance")
    z = 1.0
    s_z = make_spec(wavelength=wave_rest * (1.0 + z))
    s_z.setup_for_model(wave_rest, zred=z,
                        lib_resolution=(wave_rest, sig_lib))
    lam0 = 5000.0
    slib = float(np.interp(lam0, wave_rest, sig_lib))
    y = gaussian_line(wave_rest, lam0, slib)
    out = np.asarray(s_z.predict(y, wave_rest))
    w = measured_sigma_v(wave_rest * (1.0 + z), out, lam0 * (1.0 + z))
    check(f"z=1: width {w:.1f} ~ {SIG_T:.0f}",
          abs(w - SIG_T) / SIG_T < 0.05)

    print()
    if FAILURES:
        print(f"GATE FAILED ({len(FAILURES)}): {FAILURES}")
        return 1
    print("GATE PASSED — automatic library-resolution handling verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
