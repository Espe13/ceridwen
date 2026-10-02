"""TempleDisk (ceridwen/agn/disk.py): AGN_PLAN §4 B3, B4, B8 for ``fagn_bbb``, plus the
setup-time guards and the corrected-mode template range.

Component-level: a stub ``AGNContext`` on a synthetic grid, no CSP (the CSP wiring and its
byte-identity at zero AGN are the hook's tests).
"""
import numpy as np
import pytest

import jax
import jax.numpy as jnp

from ceridwen.agn import AGNContext
from ceridwen.agn.disk import NORM_WINDOW, TempleDisk, load_temple21

# 1 A .. 1e8 A like the CERIDWEN grids, with 2 A pixels across 5000-6000 A so the
# 5480-5520 A window holds an even number (20) of pixels, as on the test grid.
_LOG = np.geomspace(1.0, 1e8, 3000)
WAVE = np.concatenate([_LOG[_LOG < 5000.0], np.arange(5001.0, 6000.0, 2.0), _LOG[_LOG > 6000.0]])
N_WIN = int(np.sum((WAVE >= NORM_WINDOW[0]) & (WAVE <= NORM_WINDOW[1])))


def _galaxy(wave, tilt=0.0):
    """Smooth positive stand-in for the no-dust young+old spectrum (L_sun/Hz per M_formed),
    with a little structure in the 5500 A window so the median picks distinct pixels."""
    x = np.log10(wave / 5500.0)
    g = 1e-8 * np.exp(-0.5 * (x / 0.8) ** 2) * (1.0 + tilt * x)
    g *= 1.0 + 0.01 * np.sin(wave / 3.0)
    return g


GAL = _galaxy(WAVE)


def _ctx(gal=GAL, wave=WAVE, mode="prospector"):
    return AGNContext(wave=jnp.asarray(wave), L_bol_star=jnp.asarray(1.0),
                      gal_nodust_young_old=jnp.asarray(gal),
                      diffuse_curve=jnp.ones(len(wave)), mode=mode)


@pytest.fixture(scope="module", params=["prospector", "corrected"])
def disk(request):
    return TempleDisk(WAVE, mode=request.param)


def test_window_has_even_pixel_count():
    assert N_WIN == 20


def test_template_resampling_is_np_interp():
    """Prospector mode = np.interp on the model grid (constant ends); corrected = the same
    inside the template range and zero outside it."""
    w_t, f_t = load_temple21()
    assert (w_t[0], w_t[-1]) == (91.0, 1e8)
    ref = np.interp(WAVE, w_t, f_t)
    p = TempleDisk(WAVE, mode="prospector")
    c = TempleDisk(WAVE, mode="corrected")
    f = TempleDisk(WAVE, mode="fsps")
    assert np.array_equal(np.asarray(p.template), ref)
    assert np.array_equal(np.asarray(f.template), ref)
    inside = (WAVE >= 91.0) & (WAVE <= 1e8)
    assert np.array_equal(np.asarray(c.template)[inside], ref[inside])
    assert np.all(np.asarray(c.template)[~inside] == 0.0)
    assert np.all(np.asarray(p.template)[WAVE < 91.0] == f_t[0])   # constant blue end


def test_bundled_template_is_prospectors_file():
    """The data rows of temple21.txt are byte-for-byte Prospector's temple.txt (91f8e13):
    sha256 of everything below the provenance header."""
    import hashlib
    from ceridwen.agn.disk import TEMPLE21_FILE
    lines = open(TEMPLE21_FILE, "rb").read().split(b"\n")
    body = b"\n".join(lines[13:])
    assert hashlib.sha256(body).hexdigest() == \
        "ab622a7e9b62d76bb16388484774fb0eac88ce6440546f1f258eaa24bc1398d2"


def test_normalisation_definition(disk):
    """fagn_bbb is F_nu,disc / F_nu,gal at 5500 A in Prospector's sense: the median of the
    pixel ratios over [5480, 5520] A equals fagn_bbb."""
    win = (WAVE >= NORM_WINDOW[0]) & (WAVE <= NORM_WINDOW[1])
    for f in (1e-3, 1.0, 30.0):
        d = np.asarray(disk.rest_spectrum({"fagn_bbb": f}, _ctx()))
        assert np.median(d[win] / GAL[win]) == pytest.approx(f, rel=1e-14)


def test_B3_zero_fagn_bbb_adds_exactly_zero(disk):
    """B3 at component level: fagn_bbb = 0 gives +0.0 everywhere, so adding it to any
    stellar spectrum is byte-identical."""
    d = np.asarray(disk.rest_spectrum({"fagn_bbb": 0.0}, _ctx()))
    assert d.tobytes() == np.zeros_like(WAVE).tobytes()
    stellar = jnp.asarray(GAL)
    assert np.asarray(stellar + d).tobytes() == np.asarray(stellar).tobytes()
    # default (key absent) is 0 too
    assert np.asarray(disk.rest_spectrum({}, _ctx())).tobytes() == d.tobytes()


def test_B4_exact_linearity_in_fagn_bbb(disk):
    """B4: disc(k f) == k disc(f) exactly for k a power of two (no rounding), and to
    one ulp for any k."""
    ctx = _ctx()
    d1 = np.asarray(disk.rest_spectrum({"fagn_bbb": 0.37}, ctx))
    for k in (2.0, 0.5, 1024.0):
        dk = np.asarray(disk.rest_spectrum({"fagn_bbb": 0.37 * k}, ctx))
        assert np.array_equal(dk, k * d1)
    d3 = np.asarray(disk.rest_spectrum({"fagn_bbb": 3.0 * 0.37}, ctx))
    np.testing.assert_allclose(d3, 3.0 * d1, rtol=4e-16, atol=0)


def test_scales_with_the_galaxy(disk):
    """fagn_bbb is relative: doubling the galaxy doubles the disc, exactly."""
    a = np.asarray(disk.rest_spectrum({"fagn_bbb": 1.0}, _ctx(GAL)))
    b = np.asarray(disk.rest_spectrum({"fagn_bbb": 1.0}, _ctx(2.0 * GAL)))
    assert np.array_equal(b, 2.0 * a)


def test_zero_galaxy_is_finite_and_zero(disk):
    """An all-zero galaxy (e.g. an empty SFH), or one with a zero pixel in the window,
    gives a zero disc and finite gradients instead of Prospector's division by zero."""
    for gal in (np.zeros_like(GAL), np.where(np.abs(WAVE - 5501.0) < 0.5, 0.0, GAL)):
        f = lambda fb, g: jnp.sum(disk.rest_spectrum({"fagn_bbb": fb}, _ctx(g)))
        assert float(f(1.0, gal)) == 0.0
        gfb, gg = jax.grad(f, argnums=(0, 1))(1.0, jnp.asarray(gal))
        assert np.isfinite(float(gfb)) and np.all(np.isfinite(np.asarray(gg)))


def test_setup_refuses_grid_without_window():
    coarse = np.geomspace(1.0, 1e8, 200)
    assert not np.any((coarse >= 5480) & (coarse <= 5520))
    with pytest.raises(ValueError, match="no pixel in"):
        TempleDisk(coarse, mode="prospector")
    with pytest.raises(ValueError, match="mode"):
        TempleDisk(WAVE, mode="bogus")
    with pytest.raises(ValueError, match="ascending"):
        TempleDisk(WAVE[::-1], mode="corrected")


# --------------------------------------------------------------------------- B8
def _loss(disk, gal):
    def f(fb):
        return jnp.sum(jnp.log(disk.rest_spectrum({"fagn_bbb": fb}, _ctx(gal))[WAVE > 100.0]))
    return f


def test_B8_jit_matches_eager(disk):
    th = {"fagn_bbb": jnp.asarray(0.7)}
    eager = np.asarray(disk.rest_spectrum(th, _ctx()))
    jitted = np.asarray(jax.jit(lambda t, c: disk.rest_spectrum(t, c))(th, _ctx()))
    assert np.array_equal(eager, jitted)


@pytest.mark.parametrize("W", [64, 100])
def test_B8_vmap_matches_loop(disk, W):
    """vmap over fagn_bbb AND the galaxy (the normalisation is traced) at the production
    batch widths."""
    rng = np.random.default_rng(0)
    fbs = 10 ** rng.uniform(-5, 2, W)
    tilts = rng.uniform(-0.3, 0.3, W)
    gals = np.stack([_galaxy(WAVE, t) for t in tilts])
    fn = jax.jit(jax.vmap(lambda fb, g: disk.rest_spectrum({"fagn_bbb": fb}, _ctx(g))))
    batched = np.asarray(fn(jnp.asarray(fbs), jnp.asarray(gals)))
    single = jax.jit(lambda fb, g: disk.rest_spectrum({"fagn_bbb": fb}, _ctx(g)))
    loop = np.stack([np.asarray(single(fbs[i], gals[i])) for i in range(W)])
    np.testing.assert_allclose(batched, loop, rtol=1e-15, atol=0)


@pytest.mark.parametrize("fb", [1e-5, 1e-3, 1.0, 30.0, 100.0])
def test_B8_grad_fagn_bbb_finite_and_matches_fd(disk, fb):
    """Prior range LogUniform(1e-5, 100): d/d fagn_bbb of sum(log disc) is n_pix / fagn_bbb."""
    f = _loss(disk, GAL)
    g = float(jax.grad(f)(fb))
    n = int(np.sum(np.asarray(disk.template)[WAVE > 100.0] > 0))
    assert np.isfinite(g)
    assert g == pytest.approx(n / fb, rel=1e-12)
    h = 1e-6 * fb
    fd = (float(f(fb + h)) - float(f(fb - h))) / (2 * h)
    assert g == pytest.approx(fd, rel=1e-6)


def test_B8_grad_of_median_is_piecewise(disk):
    """d disc / d galaxy reaches only the two middle pixels of the 20-pixel window (each
    with weight 1/2 in the median), and equals central differences there."""
    def f(g):
        return disk.rest_spectrum({"fagn_bbb": 1.0}, _ctx(g))[np.argmin(np.abs(WAVE - 1e4))]
    gg = np.asarray(jax.grad(f)(jnp.asarray(GAL)))
    nz = np.flatnonzero(gg)
    assert len(nz) == 2
    assert np.all((WAVE[nz] >= NORM_WINDOW[0]) & (WAVE[nz] <= NORM_WINDOW[1]))
    for i in nz:
        h = 1e-7 * GAL[i]
        gp, gm = GAL.copy(), GAL.copy()
        gp[i] += h
        gm[i] -= h
        fd = (float(f(jnp.asarray(gp))) - float(f(jnp.asarray(gm)))) / (2 * h)
        assert gg[i] == pytest.approx(fd, rel=1e-6)
