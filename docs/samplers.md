# Samplers

`fitSED(model, sampler=...)` supports nested sampling and NUTS, with optional
VI preconditioning for NUTS. Each uses the same log-posterior and returns a
`SamplingResult` compatible with `PostProcess`, plotting and result files.
The examples use `model` from the [Quick start](quickstart.md).

| sampler | `sampler=` | gradients | evidence | on a laptop CPU |
|---|---|---|---|---|
| nested sampling (default) | `"nested"` (or `"ns"`, `"nss"`) | no | yes, `result.log_evidence` | minutes to tens of minutes |
| NUTS | `"nuts"` (or `"hmc"`) | yes | no | hours; use a GPU |
| VI-preconditioned NUTS | `"nuts"` with `vi="tril"` or `vi="iaf"` | yes | no | hours; use a GPU |

## Nested sampling

Nested sampling is the default. It draws live points from the prior, uses no
gradients and estimates the Bayesian evidence (log Z). Every free parameter
requires a proper prior.

```python
import jax
from ceridwen import fitSED

result = fitSED(
    model,
    sampler="nested",
    sampler_kwargs={"num_live": 400, "num_delete": 80, "logZ_tol": -5.0},
    rng_key=jax.random.PRNGKey(42),
    output_dir="./my_fit",
)
print(f"log Z = {result.log_evidence:.2f} +/- {result.log_evidence_err:.2f}")
```

Pass these settings in `sampler_kwargs`:

- `num_live`: number of live points, default 500. Increasing this generally
  improves posterior resolution and evidence precision at a proportional cost.
- `num_delete`: points replaced per iteration, default `num_live // 5`.
- `num_inner_steps`: MCMC steps per replacement, default 30.
- `logZ_tol`: stopping threshold for the remaining evidence,
  ln(Z_live / Z), default −5.
- `slice_kernel`: `"lanes"` (default), `"stock"` or `"carry"`.

`"stock"` uses `blackjax.nss` unchanged. `"carry"` evaluates each stepping-out
edge once rather than twice under `vmap`. `"lanes"` runs replacement chains in
a shared loop, skips likelihood evaluation for candidates below the prior
slice level and uses idle batch slots to evaluate candidates in advance. It
implements Dance et al. (2025, arXiv:2503.17405), with code by Liu Hao
(@potatoist314).

`"carry"` and `"lanes"` use the same random numbers and evaluations as
`"stock"`. On CPUs, `"carry"` is bitwise identical to `"stock"`. `"lanes"`
is bitwise identical on Apple Silicon; on x86, XLA's arithmetic ordering
introduces differences in the last bits. Long x86 runs can consequently
follow different trajectories while agreeing statistically.

There is one intentional difference with blackjax 1.6. When all moves in a
replacement chain are rejected, `"lanes"` assigns the replacement particle
the birth contour of the particle it replaces. Blackjax 1.7 uses the same
rule. Version 1.6 retained the starting particle's birth contour, which can
bias ln Z on a flat likelihood. This affects weights and evidence only for
chains without an accepted move.

GPU arithmetic can produce different trajectories as batch composition
changes. Runs need not be bitwise reproducible across processes, including
two `"stock"` runs with the same key on A100s. Compare their results
statistically.

For a 19-parameter JADES photometry and emission-line model with
`num_live=300` and `num_delete=64`, `"lanes"` was 3.5–3.8 times faster than
`"stock"` on an A100. Posteriors agreed, and ln Z differed by at most 2e-5.
A single measurement on a busy 11-core laptop CPU gave a sevenfold speedup
per iteration.

The progress bar counts dead points. Runtime grows with `num_live`. On a
shared 11-core laptop CPU, the Quick start model with `num_live=100`,
`num_delete=20` and `logZ_tol=-2.0` took 12 minutes with `"lanes"` and
21–29 minutes with `"stock"`. Both used 67,800 likelihood calls. With 400
live points, `"stock"` processed about 1.3 dead points per second and took hours.

### Checkpoints and resuming

Set `checkpoint_dir=` or `$CERIDWEN_CHECKPOINT_DIR` to save
`ns_checkpoint_<pid>.pkl` periodically. `checkpoint_interval_s` defaults to
1200 seconds. To resume an interrupted run, pass the checkpoint as
`resume_from=` with the same `rng_key` and settings. It restores the live
and dead points and can reproduce an uninterrupted run.

Loading checks the settings, parameter shapes and live-point log-likelihoods
(relative tolerance 1e-6) against the model. You may change `slice_kernel`
when resuming because the kernels use the same moves and random numbers,
subject to the numerical and version differences described above.

```python
result = fitSED(model, sampler="nested", output_dir="./my_fit",
                rng_key=jax.random.PRNGKey(42),
                sampler_kwargs={"checkpoint_dir": "./ckpt",
                                "resume_from": "./ckpt/ns_checkpoint_12345.pkl"})
```

## NUTS

The No-U-Turn sampler uses gradients from the JAX forward model. It is suited
to GPU fits with many parameters. Defaults are 4 chains, 2000 samples per
chain, 1500 warmup steps, a dense mass matrix and target acceptance 0.95.
Bounded parameters are automatically mapped to an unconstrained space.

```python
result = fitSED(
    model,
    sampler="nuts",
    sampler_kwargs={"num_chains": 4, "num_samples": 2000},
    rng_key=jax.random.PRNGKey(42),
    output_dir="./my_fit",
)
```

Warmup and sampling each run as a compiled loop without progress output.
Ctrl-C does not interrupt a loop; stop the process with `kill`. An interrupted
NUTS run does not save a result.

## VI-preconditioned NUTS

Variational inference fits a transport map before NUTS samples in the
transformed space (Hoffman et al. 2019). Use `vi="tril"` for a full-rank
Gaussian or `vi="iaf"` for an inverse autoregressive flow (NeuTra).
With VI, warmup defaults to 200 steps and a diagonal mass matrix.

```python
result = fitSED(
    model,
    sampler="nuts",
    vi="tril",
    sampler_kwargs={"num_chains": 4, "num_samples": 2000},
    rng_key=jax.random.PRNGKey(42),
    output_dir="./my_fit",
)

# VI convergence: -ELBO should drop and then plateau. It can be negative (the likelihood is
# normalised), so the axis is linear.
import matplotlib.pyplot as plt
plt.figure(); plt.plot(result.raw["vi_losses"])
plt.xlabel("VI iteration"); plt.ylabel(r"$-\mathrm{ELBO}$")
plt.savefig("vi_losses.png")
```

## Starting at the MAP

`fitSED(..., optimize=True)` first finds a MAP estimate using L-BFGS
(`ceridwen.optimize.map_fit`). It starts from `model.theta_init` and
`n_starts` prior draws. NUTS and its VI map start from the MAP; nested sampling
still draws live points from the prior. Both save the MAP under `/map` in
the result file. You can also call `map_fit(model)` separately and pass
the returned `MAPResult.theta` as `free_param_init`.

```python
from ceridwen.optimize import map_fit

best = map_fit(model, n_starts=16, rng_key=jax.random.PRNGKey(1))
result = fitSED(model, sampler="nuts", optimize=True,
                optimize_kwargs={"n_starts": 16}, output_dir="./my_fit")
```

## NUTS on a CPU: measured times

These measurements used the Quick start model: twelve photometric bands,
a 600-pixel spectrum and nine free parameters. They ran on shared 11-core
laptop CPUs with other jobs active, so use them as approximate timings.

| run | measured |
|---|---|
| `map_fit(model, n_starts=16)` | 108.5 s |
| VI-preconditioned NUTS (`vi="tril"`), 4 x 2000 | VI 72 s; warmup 596 s (200 steps, step size 0.0017); sampling not finished after 1 h 54 min |
| NUTS after the MAP (`optimize=True`), 4 x 2000 | not finished after 80 min |

Nested sampling is generally the practical choice on a CPU; NUTS benefits from a GPU.
