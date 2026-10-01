"""E6: ``SSPData.ssp_flux`` is a read-only host NumPy array, and the device keeps only the
CSP's float32 cube.

Reference: the previous behaviour, a float64 device array cast by XLA on the device
(``jnp.array(jnp.asarray(flux64), dtype=float32)``).  The CSP's cube must be byte-identical to
it (XLA's convert flushes float32 subnormals; a NumPy cast would not), the content hash must
not change, and after building a CSP no float64 grid-sized array may be alive on the device.
"""
import dataclasses
import gc
import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _gridfixture import require_test_grid  # noqa: E402

jax.config.update("jax_enable_x64", True)

from ceridwen import SSPData, CSPBasis, Cosmology   # noqa: E402

GRID = require_test_grid()


def _grid_sized_f64_on_device(shape):
    n = int(np.prod(shape))
    return [a for a in jax.live_arrays() if a.dtype == jnp.float64 and a.size >= n]


def test_load_gives_read_only_host_numpy():
    ssp = SSPData.load(str(GRID))
    assert isinstance(ssp.ssp_flux, np.ndarray) and ssp.ssp_flux.dtype == np.float64
    assert not ssp.ssp_flux.flags.writeable
    with pytest.raises(ValueError):
        ssp.ssp_flux[0, 0, 0] = 1.0
    s32 = SSPData.load(str(GRID), flux_dtype="float32")
    assert isinstance(s32.ssp_flux, np.ndarray) and s32.ssp_flux.dtype == np.float32
    assert s32.chash == ssp.chash


def test_jax_input_is_moved_to_the_host_with_the_same_hash():
    ssp = SSPData.load(str(GRID))
    j = SSPData(ssp.ssp_lgmet, ssp.ssp_lg_age_gyr, ssp.ssp_wave, jnp.asarray(ssp.ssp_flux),
                ssp_resolution=ssp.ssp_resolution, log10_zsun=ssp.log10_zsun)
    assert isinstance(j.ssp_flux, np.ndarray) and not j.ssp_flux.flags.writeable
    assert j.chash == ssp.chash
    r = dataclasses.replace(ssp, ssp_flux=jnp.asarray(ssp.ssp_flux) * 1.0)
    assert isinstance(r.ssp_flux, np.ndarray)
    w = np.array(ssp.ssp_flux)                     # a writeable caller array stays writeable
    SSPData(ssp.ssp_lgmet, ssp.ssp_lg_age_gyr, ssp.ssp_wave, w,
            ssp_resolution=ssp.ssp_resolution, log10_zsun=ssp.log10_zsun)
    assert w.flags.writeable


def test_csp_cube_byte_identical_and_no_float64_cube_on_device():
    ssp = SSPData.load(str(GRID))
    ref = np.asarray(jnp.array(jnp.asarray(ssp.ssp_flux), dtype=jnp.float32))
    gc.collect()
    # other tests of the session may hold grid-sized float64 arrays: only new ones count
    before = {id(a) for a in _grid_sized_f64_on_device(ssp.ssp_flux.shape)}
    csp = CSPBasis(ssp, lookback_time=jnp.linspace(0.0, 5.0, 4), add_neb=False,
                   verbose=False, cosmo=Cosmology.planck18())
    gc.collect()
    assert np.asarray(csp.flux).tobytes() == ref.tobytes()
    assert csp.flux.dtype == jnp.float32
    new = [a for a in _grid_sized_f64_on_device(ssp.ssp_flux.shape) if id(a) not in before]
    assert new == []
