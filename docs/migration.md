# Upgrading to 2.0

CERIDWEN 2.0 changes two parameterisations so that every sampled number reaches the data.
Code written for 1.x raises a `ValueError` that names the fix and prints the converted
values; nothing is reinterpreted silently.

## Step SFH: one SFR per bin

`sfh_interp="step"` holds the SFR constant within each bin. It now takes one SFR per bin,
`theta["sfh"]` of shape `(n_time - 1,)`, as Prospector's `FastStepBasis` does, and the
continuity prior acts on the log SFR ratios of adjacent bins: `n_time - 2` of them.
`logsfr_ratios_to_sfh(r, sfh_times_yr=csp.sfh_times)` returns these bin SFRs when it is
given `n_time - 2` ratios, normalised so that Σ SFR_i Δt_i = 1 M☉, so `logmass` is still the
formed mass.

In 1.x a step CSP took one SFR per node and used only the mean of each bin's two nodes.
`n_time` node values then set `n_time - 1` bin values, one alternating combination of the
nodes changed no output, and the continuity prior acted on node ratios rather than on the
bins the data constrain.

To keep the number of free ratios, add one lookback node:

```python
import jax.numpy as jnp
from ceridwen.model import logsfr_ratios_to_sfh

# 1.x: 6 nodes, 5 ratios between nodes
# lookback = jnp.linspace(0.0, 12.0, 6)
# 2.0: 7 nodes, 6 bins, 5 ratios between adjacent bins
lookback = jnp.linspace(0.0, 12.0, 7)
n_ratios = lookback.size - 2
```

A per-node SFR given directly (`theta["sfh"]`, `fixed=`, a constant transform) is refused in
step mode. Its bin means give the 1.x model exactly:

```python
sfh_bins = 0.5 * (sfh_nodes[:-1] + sfh_nodes[1:])
```

`sfh_interp="linear"` (FSPS's tabular SFH, linear between nodes) is unchanged: one SFR per
node and `n_time - 1` ratios. `SedModel` refuses an `sfh` transform whose output length does
not match the CSP when the model is built.

## Metallicity history: one value per bin

`theta["logzsol_bins"]` (shape `(n_time - 1,)`, log10(Z/Z_sun) of the grid, index 0 = the
youngest bin) replaces the per-node `theta["logzsol_hist"]`. The metallicity is constant
within each bin. `logzsol_hist` entered the model only through its bin means, with the same
invisible combination as the SFH above; it now raises wherever it appears (theta, `fixed=`,
priors, `free_param_init`, transforms) and prints

```python
logzsol_bins = 0.5 * (logzsol_hist[:-1] + logzsol_hist[1:])
```

which gives the same model. A prior on the nodes has no per-bin equivalent, so set the
priors per bin. With `gas_tied=True` the gas metallicity follows `logzsol_bins[0]`.
Constant metallicity (`zh_const=True`, `theta["logzsol"]`) is unchanged.

## Results from 1.x

Result files written by 1.x stay readable with `ceridwen.resultfile`. A step fit made with
1.x sampled a different prior (ratios between nodes), so its posterior is not that of the
same fit in 2.0: refit to compare. A 1.x model with a per-node step SFH or `logzsol_hist`
cannot be rebuilt; the errors above give the 2.0 equivalent.
