import jax.numpy as jnp
import jax

import numpy as np
from pathlib import Path


from ceridwen.constants import C_AA_S as CLIGHT_AA_S
MDUST_PREFACTOR = 3.21e-3 / (4.0 * jnp.pi)
_DUSTEM_CACHE = {}


class DustEmission:

    def __init__(self, duste_model="DL07",
                 dust_file=None, spec_lambda=None, **kwargs):
        """Dust emission templates ('DL07' or 'THEMIS') interpolated onto spec_lambda [Angstrom].

        Parameters
        ----------
        dust_file : str -- data root containing dust/dustem/
        kwargs : duste_qpah, duste_umin, duste_gamma defaults
        """
        self.duste_model = duste_model

        self.duste_qpah = None
        self.duste_umin = None
        self.duste_gamma = None

        self.qpaharr = None
        self.uminarr = None

        self.dustem2_dustem = None

        self.dust_file = None
        self.spec_lambda = None

        self.dwargs = kwargs

        self.duste_qpah = kwargs.pop("duste_qpah", 3.5)
        self.duste_umin = kwargs.pop("duste_umin", 1.0)
        self.duste_gamma = kwargs.pop("duste_gamma", 0.01)

        if self.duste_model == "DL07":
            self.qpaharr = jnp.array([0.47,1.12,1.77,2.50,3.19,3.90,4.58])
            self.uminarr = jnp.array([0.1,0.15,0.2,0.3,0.4,0.5,0.7,0.8,1.0,1.2,1.5,
                                      2.0,2.5,3.0,4.0,5.0,7.0,8.0,12.0,15.0,20.0,25.0])
            self.nqpah_dustem = self.qpaharr.size
            self.numin_dustem = self.uminarr.size
        elif self.duste_model == "THEMIS":
            self.qpaharr = jnp.array([0.02, 0.06, 0.10, 0.14, 0.17, 0.20, 0.24,
                                      0.28, 0.32, 0.36, 0.40]) / 2.2 * 100
            self.uminarr = jnp.array([
                0.1, 0.12, 0.15, 0.17, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6,
                0.7, 0.8, 1.0, 1.2, 1.5, 1.7, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0,
                6.0, 7.0, 8.0, 10.0, 12.0, 15.0, 17.0, 20.0, 25.0, 30.0,
                35.0, 40.0, 50.0, 80.0
            ])
            self.nqpah_dustem = self.qpaharr.size
            self.numin_dustem = self.uminarr.size
        else:
            raise ValueError("Invalid duste_model. Choose 'DL07' or 'THEMIS'.")

        if dust_file is None or spec_lambda is None:
            raise ValueError("If `duste=True`, both `dust_file` and `spec_lambda` must be provided.")

        self.dust_file = dust_file
        self.spec_lambda = spec_lambda

        self.load_dust_emission(dust_file, spec_lambda)

        nu = CLIGHT_AA_S / jnp.asarray(spec_lambda)
        dnu = jnp.diff(nu)
        self._trap_w = jnp.concatenate([
            jnp.array([0.5 * dnu[0]]),
            0.5 * (dnu[:-1] + dnu[1:]),
            jnp.array([0.5 * dnu[-1]]),
        ])

    def __repr__(self):
        def format_array(arr):
            if arr is None:
                return "None"
            if arr.ndim == 1 and len(arr) <= 5:
                return f"[{', '.join(map(str, arr))}]"
            return f"\n    " + "\n    ".join(map(str, arr))

        attributes = {
            "Duste model": self.duste_model,
            "DUST qPAH": self.duste_qpah,
            "DUST Umin": self.duste_umin,
            "DUST Gamma": self.duste_gamma,
            f"qpaharr ({self.duste_model})": format_array(self.qpaharr),
            f"uminarr ({self.duste_model})": format_array(self.uminarr),
            "dust_file": self.dust_file,
            "spec_lambda": self.spec_lambda.shape if self.spec_lambda is not None else None,
        }

        if self.dwargs:
            attributes["Extra parameters (dwargs)"] = self.dwargs

        attr_str = "\n".join(f"  {k:<30}: {v}" for k, v in attributes.items() if v is not None)
        return f"\nDustEmission Model:\n{'='*50}\n{attr_str}\n{'='*50}"

    def get_default_params(self):
        """Return dict of default dust-emission fit parameters."""
        return {
            "duste_qpah": jnp.asarray(self.duste_qpah),
            "duste_umin": jnp.asarray(self.duste_umin),
            "duste_gamma": jnp.asarray(self.duste_gamma),
        }

    def load_dust_emission(self, dust_file=None, spec_lambda=None):

        if dust_file is None:
            dust_file = self.dust_file
        if spec_lambda is None:
            spec_lambda = self.spec_lambda

        dust_model_params = {
            "DL07": (7, 1001, 22),
            "THEMIS": (11, 576, 37),
        }

        nqpah_dustem, ndim_dustem, numin_dustem = dust_model_params[self.duste_model]

        spec_np = np.asarray(spec_lambda, dtype=np.float64)
        key = (self.duste_model, str(dust_file), spec_np.shape, spec_np.tobytes())
        hit = _DUSTEM_CACHE.get(key)
        if hit is not None:
            self.dustem2_dustem = hit
            return
        jj = int(np.searchsorted(spec_np / 1E4, 1, side='left'))
        dustem2_dustem = np.zeros((len(spec_lambda), nqpah_dustem, numin_dustem * 2))

        for k in range(nqpah_dustem):
            filename = Path(dust_file) / "dust" / "dustem" / f"{self.duste_model}_MW3.1_{'100' if k == 10 else f'{k}0'}.dat"

            if not filename.exists():
                raise FileNotFoundError(f"Error opening dust emission file: {filename}. File does not exist.")

            with filename.open('r') as f:
                next(f)
                next(f)

                try:
                    table = np.loadtxt(f, max_rows=ndim_dustem)
                except Exception:
                    raise RuntimeError(f"Error reading dust emission file: {filename}")
            if table.shape != (ndim_dustem, numin_dustem * 2 + 1):
                raise RuntimeError(f"{filename}: expected {ndim_dustem} rows x "
                                   f"{numin_dustem * 2 + 1} columns, got {table.shape}")
            lambda_dustem = table[:, 0] * 1E4
            dustem_dustem = table[:, 1:]
            for j in range(numin_dustem * 2):
                dustem2_dustem[jj:, k, j] = np.interp(spec_np[jj:], lambda_dustem, dustem_dustem[:, j])

        self.dustem2_dustem = jnp.array(dustem2_dustem)
        _DUSTEM_CACHE[key] = self.dustem2_dustem

    def update_dust_params(self, duste_qpah = 3.5, duste_umin = 1.0, duste_gamma = 0.01):
        """Set the default dust-emission parameters."""
        self.duste_qpah = duste_qpah
        self.duste_umin = duste_umin
        self.duste_gamma = duste_gamma

    def compute_dust_emission(self, spec_attn, spec_dustfree, spec_lambda, diffuse_curve,
                                        duste_qpah, duste_umin, duste_gamma):
        """Return (spec_attn + dust emission, dust mass, dust emission); diffuse_curve is exp(-tau_diffuse)."""
        tiny = 1e-70
        w = self._trap_w
        dc = diffuse_curve.ravel()
        f_attn = spec_attn.ravel()
        f_free = spec_dustfree.ravel()

        lbold = jnp.dot(f_attn, w)
        lboln = jnp.dot(f_free, w)

        qlo = jnp.clip(jnp.searchsorted(self.qpaharr, duste_qpah) - 1, 0, self.nqpah_dustem - 2)
        dq  = jnp.clip((duste_qpah - self.qpaharr[qlo]) / (self.qpaharr[qlo + 1] - self.qpaharr[qlo]), 0, 1)

        ulo = jnp.clip(jnp.searchsorted(self.uminarr, duste_umin) - 1, 0, self.numin_dustem - 2)
        du  = jnp.clip((duste_umin - self.uminarr[ulo]) / (self.uminarr[ulo + 1] - self.uminarr[ulo]), 0, 1)

        gamma = jnp.clip(duste_gamma, 0.0, 1.0)

        w00 = (1 - dq) * (1 - du)
        w10 = dq * (1 - du)
        w01 = (1 - dq) * du
        w11 = dq * du

        i_lo = 2 * ulo
        i_hi = 2 * (ulo + 1)

        D = self.dustem2_dustem  # (n_wave, n_qpah, 2*n_umin): Umin/Umax pairs interleaved

        dumin = (w00 * D[:, qlo, i_lo]     + w10 * D[:, qlo + 1, i_lo] +
                 w11 * D[:, qlo + 1, i_hi] + w01 * D[:, qlo, i_hi])

        dumax = (w00 * D[:, qlo, i_lo + 1]     + w10 * D[:, qlo + 1, i_lo + 1] +
                 w11 * D[:, qlo + 1, i_hi + 1] + w01 * D[:, qlo, i_hi + 1])

        mduste = jnp.maximum((1 - gamma) * dumin + gamma * dumax, tiny).ravel()
        norm   = jnp.dot(mduste, w)
        mduste_norm = mduste / norm

        labs0  = lboln - lbold
        duste0 = jnp.maximum(mduste_norm * labs0, tiny)

        duste0_atten = duste0 * dc
        absorbed_1 = jnp.dot(duste0 * (1.0 - dc), w)
        duste1 = jnp.maximum(mduste_norm * absorbed_1, tiny)

        duste1_atten = duste1 * dc

        tduste = duste0_atten + duste1_atten
        specdust = f_attn + tduste
        mdust = MDUST_PREFACTOR * labs0 / norm

        return specdust, mdust, tduste
