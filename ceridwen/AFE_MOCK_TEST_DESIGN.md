# [alpha/Fe] mock-test design for ceridwen

Companion to `csp/csp_afe.py` / `ssps/ssp_data_afe.py` and the validation
figure suite `csp/figures/afe_val_0[1-6]_*.pdf` (2026-08-03).

## 0. Status of the machinery (what the figures already prove)

The figure suite runs the REAL `CSPBasis_afe` weight kernel and spectrum
methods on a synthetic alpha grid whose spectral response has the correct
sign structure (Mg b deepens, Fe blanketing lifts with [alpha/Fe] at
fixed total Z):

- `afe_val_01` — grid physics sanity: SSP- and CSP-level responses are
  monotone with the right signs.
- `afe_val_02` — interpolator exactness: at grid nodes the model equals a
  direct plane contraction to ~1e-17; at bin midpoints it equals a
  hand-lerped plane to float32 eps; a dense afe scan is continuous,
  piecewise-linear, and edge-clamped.
- `afe_val_03` — gradient structure: dF/dafe is piecewise-constant with
  jumps at the five nodes and zero in the clamp regions (same regularity
  NUTS already tolerates for Z).
- `afe_val_04` — structural contracts: SSP weights are exactly
  afe-independent; a legacy 3-D grid and a theta without `"afe"` both hit
  the solar plane bit-for-bit; total SFH mass in the weights is
  afe-invariant.
- `afe_val_05` — end-to-end inversion: a noisy S/N=25 quiescent mock at
  (Z, afe) = (-1.90, +0.30) is recovered at (-1.90, +0.29) by a chi2 scan
  through the full forward model.
- `afe_val_06` — scalar-model bias under an evolving alpha history: a
  two-epoch composite (old at +0.4, young at +0.0; light fraction old
  0.44) is best fit by scalar afe = +0.18, i.e. the LIGHT-weighted mean
  (+0.176 predicted), with residuals concentrated in the alpha windows.

Caveats: the synthetic grid is not C3K, the harness runs numpy (not
jit/grad under real JAX), and no FSPS alpha grid was touched.  Those are
exactly what the campaign below covers.

## 1. Phase A — real-grid validation (before any fitting)

1. Build python-fsps@main (>= 2026-08-02) with `AFE_FLAG=1`, FSPS v4.0
   data tree in `$SPS_HOME`.  Smoke checks: `n_afe == 5`;
   `_all_ssp_spec` shapes; `afeindx` plane 2 identical to an
   `AFE_FLAG=0` C3K build (solar cross-check, catches the 1-based
   off-by-one).
2. `SSPDataAfe.from_fsps(save_to=...)` -> the 4-D grid; also build the
   `AFE_FLAG=0` C3K null grid (n_afe = 1).  Archive both (Zenodo, like
   the release grid).
3. Re-run `afe_val_02/03/04` (the notebook versions) against the REAL
   grid under REAL JAX: node exactness, lerp identity, jit/grad checks
   (`jax.grad` w.r.t. afe finite everywhere, correct sign across a
   node, no NaNs at the clamp edges), plus a strict-vs-FSPS composite
   comparison at fixed afe grid points (the `10_strict_vs_fsps` pattern,
   `zcontinuous=0, afeindx=j`).
4. LSF audit: C3K_LR resolution vs the `spectrum.py` smoothing
   assumptions (the MILES-era numbers are stale).  Gate: no fit runs
   until the LOSVD/LSF chain is validated on the C3K wavelength grid.

## 2. Phase B — scalar afe recovery campaign

Mock generation (real grid, `CSPBasis_afe` forward model — self-mocks,
so this tests inference, not template mismatch):

- Truth grid: afe in {0.0, +0.2, +0.4, +0.6} x Z in {-2.2, -1.9, -1.6}
  x three SFH classes: fast-quench (formed < 1 Gyr, quenched), extended
  quiescent, residual star formation (1% recent mass).
- Observations: JADES-like PRISM/grating wavelength coverage and S/N in
  {10, 25, 50} per pixel; photometry-only variant for a subset (expect
  afe unconstrained -> prior-recovery null test).  NO Lines observations
  (the class rejects them); mask standard emission-line windows in the
  Spectrum arms exactly as a real quiescent fit would.
- Nuisances on: logmass, dust (modest tau), sigma_smooth, noise jitter —
  the standard quiescent config minus nebular parameters.

Fit config: `afe ~ Uniform(-0.2, +0.6)` (edge pileup is then diagnosable;
switch to truncated Normal(0, 0.2) only for the weak-data tier), init at
0.0 (solar), NUTS settings as in the v2 campaign.  Reuse the
`--dump-summary` recovery workflow for collection.

Metrics and gates:

- Joint (Z, afe) coverage: 68/95% credible-region coverage of the truth
  computed on the JOINT posterior (the marginal is misleading on the
  degeneracy ridge); report the ridge orientation d(afe)/d(Z).
- Null test: afe = 0 mocks must recover afe consistent with 0 with no
  pileup at -0.2 (pileup indicates prior-likelihood mismatch or a grid
  asymmetry).
- Nuisance leakage: afe posteriors must be stable to +/- the dust and
  sigma_smooth truths (alpha signatures are narrow-band; dust is
  broad-band — verify empirically).
- Library null: fit the SAME mocks with the n_afe=1 C3K null grid
  (`CSPBasis_afe` static branch); Z and SFH posteriors must match the
  alpha-model marginals when afe_truth = 0.

## 3. Phase C — afe evolution: current status and two routes

**Not yet possible with the delivered code**: `CSPBasis_afe` implements a
single scalar `theta["afe"]` applied to the whole SFH (one flux-cube
lerp per likelihood call).  `afe_val_06` quantifies the consequence: a
scalar fit to an evolving history lands at the light-weighted mean and
leaves percent-level structured residuals in the alpha windows.

Two implementation routes, in recommended order:

### Route 1 (cheap, chemically motivated): step / plateau-knee afe(t)

[alpha/Fe](t) is not an arbitrary function: core-collapse yields set a
plateau (~+0.3 to +0.4) at early times; SN Ia iron drags the ratio down
after a delay.  Parametrise with 2-3 free parameters:

    afeh(t) = afe_young + (afe_old - afe_young) * H(t_lb > t_alpha)

(step form; or a smooth tanh knee).  Implementation: split the SSP age
axis at t_alpha's Voronoi boundary, do TWO flux lerps (one per afe
value), and contract each against the age-masked halves of the existing
(n_z, n_age) weight matrix — cost ~4 plane reads, no 4-D einsum, and the
age mask can be handled with a differentiable soft split in t_alpha.
This drops into the current architecture with no change to
`_ssp_weights`.  Identifiability caveat: t_alpha and afe_old are
strongly degenerate with the SFH itself; consider fixing t_alpha to the
SN Ia delay scale (~0.4-1 Gyr after the SFH midpoint, computed from the
sampled SFH — a transform, prospector-style) and fitting only
(afe_old, afe_young) or even only (afe_plateau, Delta).

### Route 2 (general, expensive): free per-bin afeh

Mirror of `zh`: `theta["afeh"]` shape (n_time,), per-bin bilinear
scatter over the (afe, Z) plane, weight tensor
W3[a,z,y] = sum_bin A[bin,a] * Mz[bin,z] * S[bin,y], spectrum =
einsum("azy,azyw->w", W3, flux).  Full 4-D bandwidth per likelihood call
(~5x current) and n_time extra parameters the data will rarely
constrain.  Only worth it if Route 1's residuals prove inadequate on
deep spectra; consider the SVD compression (csp_svd) if it is.

### Evolution mock design (runnable as soon as Route 1 exists)

- Generate composites with the fig-06 generator on the real grid:
  truths (afe_old, afe_young, t_alpha) in {(+0.4, 0.0, 1 Gyr),
  (+0.4, +0.2, 2 Gyr), (+0.3, +0.3, -)} (the last = scalar null).
- Fit each mock THREE ways: scalar afe (measures the light-weighted-mean
  bias in a controlled way), step-afe (recovery of all 2-3 params), and
  scalar on the scalar-null mock (regression).
- Gate: step-afe must recover the scalar-null mock with
  afe_old ~= afe_young (no spurious evolution detection), and the
  Delta-afe posterior must exclude 0 only for the truly evolving mocks.
- Report the S/N threshold at which Delta-afe becomes detectable — that
  number decides whether Route 2 is ever worth building.

## 4. Does afe replace metallicity?

No.  In FSPS v4.0 and in ceridwen, [alpha/Fe] is an ADDITIONAL axis,
orthogonal to Z:

- `ssp_lgmet` / `theta["Z"]` (or `zh`) remains log10 of the absolute
  TOTAL metal mass fraction, exactly as before (project convention).
- `afe` re-partitions that fixed total Z between alpha elements and the
  Fe-peak: raising afe at fixed Z lowers [Fe/H] and raises [alpha/H].
  [Fe/H] is therefore a DERIVED quantity, not a sampled one.
- Both remain sampled: Z sets the overall metal content, afe the
  mixture.  Expect (and report) the anticorrelated joint posterior —
  marginal Z uncertainties legitimately widen once afe is free.
- FSPS's zlegend still exists (13 Z points for aMIST) at every afe
  plane; `SSPDataAfe.from_fsps` asserts it is identical across planes.
