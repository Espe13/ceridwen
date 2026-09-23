# Analytic marginalisation of the calibration polynomial: design

### Code read (evidence)
- `ceridwen/likelihood/poly_calibration.py:27-38` `chebyshev_design_matrix`: `chebvander` of the
  pixel wavelengths mapped to [-1,1] over the unmasked pixels, (n_pix, M+1), T_0..T_M.
- `poly_calibration.py:92-102` `solve`: `w = where(mask, inv_var, 0)`, `D = mu[:,None]*A`,
  `r = where(mask, y-mu, 0)`, `M = D^T W D + diag(reg^2)`, Jacobi column scaling
  `d = M_ii^-1/2`, response `1 + A c`.
- `poly_calibration.py:104-108` `calibrate`: the weights are the noise model's inv_var at the
  UNCALIBRATED mu.
- `likelihood.py:302-340` `DiagonalGaussianLikelihood.__call__`/`make_lnprobfn`; `runner.py:146-176`
  and `MultiObservationLikelihood.make_lnprobfn` (`likelihood.py:540-560`) both call
  `lhood(y, mu*calib, sig, mask, params=theta)`: any object with that `__call__` plugs in.
- `eline_marginal.py:53-92` `eline_marginal_loglike`: prior-width scaling `d_j = s_j`,
  `P = D M D + I`, Cholesky, `ln Z = -chi2/2 + v.v/2 - sum ln L_ii - lognorm`,
  posterior `mean = abar + d P^-1 d b`, `cov = d P^-1 d`. Same maths; I reuse this form.
- `fit.py:416-469` `_poly_calibration_for` / `_likelihood_for`: profiled mode refuses any
  CALIB_FAMILIES name (`spectrum_scaling`, `spectrum_calib`) on the same spectrum; logify and GP
  `noise` are refused for every spectrum already (`fit.py:447-455`).
- `csp/csp.py:990-996`: sampled `spectrum_scaling * (1 + spectrum_calib . P(x))` multiplies the
  prediction before the likelihood; `runner.py:171` multiplies the fixed `calibration` vector.
- `Spectrum` has no `upper_limit` kwarg (`observation/base.py:33-45` rejects unknown kwargs), so the
  upper-limit refusal is defensive (checked on the attribute in `_likelihood_for`).

### Derivation
Model: `y = mu (1 + A c) + n` on the unmasked pixels, i.e. `r0 := y - mu = D c + n`,
`D = diag(mu) A` (n x k, k = M+1), `n ~ N(0, C)`, `C = diag(1/w)`, `w` = noise-model inv_var at
the uncalibrated mu (as the profiled mode), prior `c ~ N(c0, Lambda)`, `Lambda = diag(s^2)`,
`c0 = 0`.

Marginal: `r0 ~ N(D c0, Sigma)`, `Sigma = C + D Lambda D^T`. With `r = r0 - D c0`:

- Woodbury: `Sigma^-1 = C^-1 - C^-1 D (Lambda^-1 + D^T C^-1 D)^-1 D^T C^-1 = C^-1 - C^-1 D M^-1 D^T C^-1`,
  so `r^T Sigma^-1 r = r^T C^-1 r - b^T M^-1 b`, `b = D^T C^-1 r`.
- Determinant lemma: `|C + D Lambda D^T| = |Lambda^-1 + D^T C^-1 D| |Lambda| |C| = |M| |Lambda| |C|`.
- Hence `ln N(r; 0, Sigma) = -1/2 r^T C^-1 r - 1/2 ln|2 pi C|  + 1/2 b^T M^-1 b - 1/2 ln|M| - 1/2 ln|Lambda|`
  `= ln L_diag(r) + 1/2 b^T M^-1 b - 1/2 ln|M| - 1/2 ln|Lambda|`. The prompt's formula holds.
- Signs: `M >= Lambda^-1` so `ln|M| + ln|Lambda| = ln|I + Lambda^1/2 D^T C^-1 D Lambda^1/2| >= 0`:
  the Occam term `-1/2(ln|M|+ln|Lambda|)` is <= 0, and `+1/2 b^T M^-1 b >= 0`. Limits: s -> 0 gives
  `ln|M Lambda| -> 0`, `b^T M^-1 b -> 0`: the uncalibrated diagonal likelihood. 
- Masked pixels: w_i = 0 there, so they drop from `r^T C^-1 r`, `b`, `D^T C^-1 D`; `ln|2 pi C|` is
  summed over unmasked pixels only (as `lnlike_diag_gaussian`). Equivalent to deleting them from
  the dense problem (tested).
- Conditional posterior of c given theta: `N(c0 + M^-1 b, M^-1)`.
- Stable evaluation (no n x n, O(n k^2)): with `S = diag(s)`, `G = D^T W D`,
  `P = S G S + I = S M S` (exact also for s_j = 0: coefficient pinned at c0), so
  `ln|M| + ln|Lambda| = ln|P|` and `b^T M^-1 b = (S b)^T P^-1 (S b)`.  P has eigenvalues >= 1
  (Cholesky always succeeds). For large s the entries of P grow, so it is additionally
  Jacobi-scaled as in `PolynomialCalibration.solve`: `e_j = P_jj^-1/2`, `Q = E P E` (unit
  diagonal), `ln|P| = 2 sum ln diag chol(Q) - 2 sum ln e_j`, `v = chol(Q)^-1 E S b`,
  `b^T M^-1 b = v.v`. Conditional mean `S E Q^-1 E S b`, covariance `S E Q^-1 E S`.
- Link to the profile mode: the conditional mean `M^-1 b` with `Lambda^-1 = diag(1/s^2)` is exactly
  `PolynomialCalibration.solve` with `reg = 1/s` (its matrix `D^T W D + diag(reg^2)`).
- Pointwise decomposition used for `LikelihoodOutput`: with `c_hat = M^-1 b`,
  `ln L_marg = ln L_diag(r - D c_hat) - 1/2 c_hat^T Lambda^-1 c_hat - 1/2 ln|P|`, so
  `lnl_pointwise`/`residuals`/`chi` are those of the conditional-mean calibrated model, and
  `lnl_total = sum(lnl_pointwise) - 1/2 c_hat^T Lambda^-1 c_hat - 1/2 ln|P|` (documented; the
  profile of the prior + Occam factor is a global, not per-pixel, term).

### API and behaviour
- `Spectrum(polynomial_order=M, polynomial_mode="marginalize", polynomial_prior_sigma=s)`;
  `polynomial_mode` default `"profile"`; `s` scalar or (M+1,), finite, >= 0 (0 pins a coefficient;
  s = inf refused, as is a missing s). All validation at construction.
- `polynomial_regularization` given (non-zero) with marginalize: teaching error (`reg = 1/s`).
- `polynomial_prior_sigma` given with profile mode: teaching error (it would be silently ignored).
- `polynomial_mode="marginalize"` with `polynomial_order=0`: error (nothing to marginalise).
- New module `ceridwen/likelihood/poly_marginal.py`: `PolynomialMarginal` (static, hashable like
  `PolynomialCalibration`), kernel `poly_marginal_loglike`, and `PolyMarginalGaussianLikelihood`
  (frozen dataclass, `__call__`, `make_lnprobfn`, `conditional`, pytree registration, repr).
  A separate class keeps `DiagonalGaussianLikelihood` and its pytree untouched (T4 identity and
  easy merge with G1).
- Refused at construction: `logify_spectrum=True`, a GP `noise`, `marginalize_elines=True` on the
  same spectrum (the lines are then scaled by the polynomial: `y = mu(1+Ac) + (1+Ac) A_l alpha` is
  bilinear in (c, alpha), not jointly Gaussian-linear; eline_marginal.py builds its line design
  from the uncalibrated profiles times the fixed/sampled calibration only, `eline_marginal.py:552-554`).
- Refused at setup (`_likelihood_for`, needs the model): outlier mixture on that spectrum,
  upper limits, `spectrum_calib[_name]` (two polynomials), and `spectrum_scaling[_name]` unless
  s_0 = 0 (then T_0 is pinned and the grey level is the sampled scaling alone: the response
  `s (1 + sum_{m>=1} c_m T_m)` is identifiable since (s, c) -> (s, s c) is invertible; with
  s_0 > 0 the likelihood depends on s only through s(1+c_0), a ridge only the prior breaks).
- Noise weights: as the profile mode, at the uncalibrated model (exact when the noise model has
  no model-anchored term: `noise_floor`, `log_f_calib`).
- Post-processing: `PostProcess` applies the conditional-mean response per draw (the same
  `__calib_<name>` path the profile mode uses) and can return the coefficient means/sds.
- Result file: `likelihood_json` records mode, order, prior widths (via `config()`).
