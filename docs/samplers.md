# Samplers

`fitSED(model, sampler=...)` runs one of two samplers on the same log-posterior and returns
the same `SamplingResult`, so everything after the fit (`PostProcess`, the figures, the result
file) is identical. The examples below continue the [Quick start](quickstart.md): `model` is
the model built there.

| sampler | `sampler=` | gradients | evidence | on a laptop CPU |
|---|---|---|---|---|
| nested sampling (default) | `"nested"` (or `"ns"`, `"nss"`) | no | yes, `result.log_evidence` | minutes to tens of minutes |
| NUTS | `"nuts"` (or `"hmc"`) | yes | no | hours; use a GPU |
| VI-preconditioned NUTS | `"nuts"` with `vi="tril"` or `vi="iaf"` | yes | no | hours; use a GPU |

## Nested sampling

The default. It draws its live points from the prior, needs no gradients and returns the
Bayesian evidence (log Z) for model comparison. Every free parameter needs a proper prior.

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

The settings, passed in `sampler_kwargs`:

- `num_live`: live points (fitSED's default 500). More live points give a smoother posterior and
  a more precise log Z, and cost proportionally more.
- `num_delete`: live points replaced per iteration (default `num_live // 5`).
- `num_inner_steps`: inner MCMC steps per replacement (fitSED's default 30).
- `logZ_tol`: the run stops when the evidence left in the live points, ln(Z_live / Z), falls
  below this (default -5).

The sampler shows a progress bar of dead points. Its cost scales with `num_live`: on a shared
11-core laptop CPU, the Quick start model at `num_live=100, num_delete=20, logZ_tol=-2.0`
took 21-29 min in two runs (67 800 likelihood calls), and at `num_live=400` it advanced about 1.3 dead points
per second, i.e. hours (measured 2026-09-24).

### Checkpoints and resuming

With `checkpoint_dir=` (or `$CERIDWEN_CHECKPOINT_DIR`), the run writes
`ns_checkpoint_<pid>.pkl` every `checkpoint_interval_s` seconds (default 1200). After a kill,
pass that file as `resume_from=` with the same `rng_key` and settings: the run continues from
the saved live and dead points and reproduces the uninterrupted run. It refuses a checkpoint
whose settings, parameter shapes or live-point log-likelihoods (to a relative 1e-6) do not
match the model.

```python
result = fitSED(model, sampler="nested", output_dir="./my_fit",
                rng_key=jax.random.PRNGKey(42),
                sampler_kwargs={"checkpoint_dir": "./ckpt",
                                "resume_from": "./ckpt/ns_checkpoint_12345.pkl"})
```

## NUTS

The No-U-Turn sampler uses the gradient of the log-posterior, which the JAX forward model
provides. It suits a GPU and many parameters. Defaults: 4 chains, 2000 samples each, 1500
warmup steps with a dense mass matrix, target acceptance 0.95; bounded priors are sampled in
an unconstrained space automatically.

```python
result = fitSED(
    model,
    sampler="nuts",
    sampler_kwargs={"num_chains": 4, "num_samples": 2000},
    rng_key=jax.random.PRNGKey(42),
    output_dir="./my_fit",
)
```

Warmup and sampling are each one compiled loop, so NUTS prints nothing while it runs, and
Ctrl-C does not stop it inside a loop; stop it with `kill`. A killed NUTS run writes no result.

## VI-preconditioned NUTS

Variational inference first fits a transport map to the posterior; NUTS then samples in the
whitened space (Hoffman et al. 2019). `vi="tril"` is a full-rank Gaussian, `vi="iaf"` an
inverse autoregressive flow (NeuTra). With VI the warmup defaults to 200 steps with a
diagonal mass matrix.

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

`fitSED(..., optimize=True)` first maximises the same log-posterior with L-BFGS
(`ceridwen.optimize.map_fit`, from `model.theta_init` plus `n_starts` prior draws), starts NUTS
(and its VI map) there, and stores the MAP under `/map` in the result file. Nested sampling
draws its live points from the prior, so there the MAP is only recorded. `map_fit(model)` on
its own returns a `MAPResult` whose `.theta` can be passed as `free_param_init`.

```python
from ceridwen.optimize import map_fit

best = map_fit(model, n_starts=16, rng_key=jax.random.PRNGKey(1))
result = fitSED(model, sampler="nuts", optimize=True,
                optimize_kwargs={"n_starts": 16}, output_dir="./my_fit")
```

## NUTS on a CPU: measured times

Measured on shared 11-core laptop CPUs on 2026-09-23/24 with the Quick start model (a
12-band photometry + 600-pixel spectrum fit, 9 free parameters). The machines were running
other jobs, so these times are upper limits, but they show the order of magnitude:

| run | measured |
|---|---|
| `map_fit(model, n_starts=16)` | 108.5 s |
| VI-preconditioned NUTS (`vi="tril"`), 4 x 2000 | VI 72 s; warmup 596 s (200 steps, step size 0.0017); sampling not finished after 1 h 54 min |
| NUTS after the MAP (`optimize=True`), 4 x 2000 | not finished after 80 min |

On a CPU, use nested sampling. Run NUTS on a GPU.
