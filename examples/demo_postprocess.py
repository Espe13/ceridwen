"""
demo_postprocess.py -- posterior post-processing of a finished fit.

    python examples/demo_postprocess.py demo_1_output/ceridwen_result.h5

Rebuilds the SedModel exactly as examples/demo_1_mock_test.py built it (same
grid, observations, transforms, redshift), runs ceridwen.PostProcess on the
saved samples and writes <output>/post.npz.  Everything printed comes from the
same forward model that was fitted.
"""
import sys
import pathlib

import numpy as np

import jax.numpy as jnp

from ceridwen import SSPData, CSPBasis, SedModel, Cosmology, PostProcess, load_postprocess, read_result_h5
from ceridwen.observation import Photometry
from ceridwen.model import logsfr_ratios_to_sfh

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from demo_1_mock_test import SSP_FILE, N_TIME, ZRED, FILTERS   # noqa: E402  the fit's settings


def build_model(result_path=None):
    """The SedModel of demo_1_mock_test.py: same grid, grid nodes, physics
    switches, transform and redshift.  Priors are not needed here."""
    ssp = SSPData.load(str(SSP_FILE))
    csp = CSPBasis(ssp, lookback_time=jnp.linspace(0.0, 12.0, N_TIME),
                   zh_const=True, sfh_interp="step",
                   add_dust=False, add_diffuse_dust=True, add_neb=False,
                   verbose=False, cosmo=Cosmology.planck18())
    sfh_times_yr = np.array(csp.sfh_times)
    phot = Photometry(filters=FILTERS, name="phot")
    if result_path is not None:                      # the fitted data, for the figures
        d = read_result_h5(result_path)["obs"]["phot"]
        phot = Photometry(filters=FILTERS, flux=d["flux"], uncertainty=d["uncertainty"],
                          mask=d["mask"], name="phot")
    return SedModel(
        csp, observations=[phot],
        transforms={"sfh": lambda th, _t=sfh_times_yr:
                    logsfr_ratios_to_sfh(th["logsfr_ratios"], sfh_times_yr=_t)},
        free_param_init={"logsfr_ratios": jnp.zeros(N_TIME - 1),
                         "logmass": jnp.array([10.0])},
        zred=ZRED,
    )


def A_V(s):
    """V-band attenuation of this draw, from the dust-free and full spectra."""
    i = s.index_of(5500.0)
    return 2.5 * np.log10(s.dustfree[i] / s.full[i])


def main(result_path):
    model = build_model(result_path)                 # the fit's model, rebuilt
    pp = PostProcess(model, result_path, n_samples=2000, seed=0,
                     derived={"A_V": A_V,
                              "L_opt": lambda s: s.luminosity(s.full, 4000.0, 7000.0)})
    out = pp.run()
    saved = pp.save(pathlib.Path(result_path).with_name("post.npz"))
    figs = pp.figures(pathlib.Path(result_path).with_name("figures"), title="demo 1 (post-processed)")

    q = lambda a: np.percentile(a, [16, 50, 84])
    print(f"logmass     : {q(out['theta']['logmass'])}")
    print(f"sfr10       : {q(out['extras']['sfh']['sfr10'])}  M_sun/yr")
    print(f"MUV         : {q(out['extras']['uv']['MUV'])}")
    print(f"log nion    : {q(np.log10(out['extras']['ionizing']['nion']))}")
    print(f"A_V         : {q(out['derived']['A_V'])}")
    print(f"best fit    : logL = {out['bestfit']['log_likelihood']:.2f}, "
          f"logmass = {out['bestfit']['theta']['logmass']:.3f}")
    print(f"written     : {saved}")
    print(f"figures     : {', '.join(str(p) for p in figs.values())}")
    back = load_postprocess(saved)
    assert back["extras"]["sfh"]["sfr10"].shape == out["extras"]["sfh"]["sfr10"].shape


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
