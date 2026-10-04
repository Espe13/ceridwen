"""SKIRTOR + polar dust in agn_mode="cigale" vs the REAL pcigale module (session TORUS).

T-PC-1  ``pcigale.sed_modules.skirtor2016.Module`` (CIGALE 29cb909f, run in its own reference env
        through a subprocess) vs ``SkirtorTorus.native_parts`` / ``nuclear_parts`` / ``derived``:
        36 cases (3 laws x 3 E(B-V) x random nodes x type-1 / boundary / type-2 inclinations),
        rel <= 1e-10 where > 1e-6 x peak (PLAN_AGN2 L1/L3 target: 1e-6).

Needs ``$CERIDWEN_PCIGALE_PYTHON`` (default ``agn2_ref/envs/cigale/bin/python``) and
``$CERIDWEN_PCIGALE_SRC`` (default ``agn2_ref/src/cigale``, with its database built); skips
loudly otherwise.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from ceridwen.agn import AGNContext  # noqa: E402

REF = Path.home() / "Desktop/PhD/Tools/ceridwen_workspace/agn2_ref"
PY = Path(os.environ.get("CERIDWEN_PCIGALE_PYTHON", REF / "envs/cigale/bin/python"))
SRC = Path(os.environ.get("CERIDWEN_PCIGALE_SRC", REF / "src/cigale"))

DUMP = r'''
import numpy as np, sys
from pcigale.sed_modules.skirtor2016 import Module
rng = np.random.default_rng(7)
out, rows = {}, []
for law in (0, 1, 2):
    for ebv in (0.0, 0.3, 1.0):
        t = int(rng.choice([3, 5, 7, 9, 11])); pl = float(rng.choice([0., .5, 1., 1.5]))
        q = float(rng.choice([0., .5, 1., 1.5])); oa = int(rng.choice(range(10, 90, 10)))
        R = int(rng.choice([10, 20, 30]))
        for i in (0, 90 - oa, min(90, 100 - oa), 90):
            m = Module(name="skirtor2016", t=t, pl=pl, q=q, oa=oa, R=R, Mcl=0.97, i=i,
                       disk_type=1, delta=0.0, fracAGN=0.1, lambda_fracAGN="0/0", law=law,
                       EBV=ebv, temperature=float(rng.choice([50., 100., 300.])),
                       emissivity=1.6)
            n = len(rows)
            s = m.SKIRTOR2016
            out[f"{n}/wl"], out[f"{n}/disk"] = s.wl, s.disk
            out[f"{n}/dust"], out[f"{n}/polar"] = s.dust, s.polar_dust
            out[f"{n}/info"] = np.array([m.lumin_disk, m.lumin_polar_dust, m.lumin_intrin_disk,
                                         m.l_agn_2500A])
            rows.append((t, pl, q, oa, R, i, law, ebv, m.temperature))
out["cases"] = np.array(rows, dtype=float)
np.savez(sys.argv[1], **out)
'''

WAVE = np.geomspace(1.0, 1e8, 8000)
LAWS = {0: "smc", 1: "calzetti", 2: "gaskell"}


def _rel(m, r):
    m, r = np.asarray(m, np.float64), np.asarray(r, np.float64)
    big = np.abs(r) > 1e-6 * np.abs(r).max()
    return float(np.max(np.abs(m[big] - r[big]) / np.abs(r[big]))) if big.any() else 0.0


def test_T_PC_1_vs_pcigale(tmp_path):
    if not PY.exists() or not (SRC / "pcigale/data/skirtor2016/parameters.pickle").exists():
        pytest.skip(f"MISSING pcigale env {PY} or built database under {SRC}: T-PC-1 did NOT run")
    from ceridwen.agn.skirtor import SkirtorTorus, default_skirtor_path
    from ceridwen.agn.polar import PolarDust
    if not default_skirtor_path().exists():
        pytest.skip(f"MISSING {default_skirtor_path()}: T-PC-1 did NOT run")
    npz = tmp_path / "ref.npz"
    script = tmp_path / "dump.py"
    script.write_text(DUMP)
    env = dict(os.environ, PYTHONPATH=str(SRC))
    subprocess.run([str(PY), str(script), str(npz)], check=True, env=env, cwd=str(tmp_path),
                   capture_output=True)
    z = np.load(npz)
    sks, worst = {}, 0.0
    fagn, ldust = 0.1, 9.0
    P = ldust * (1.0 / (1.0 - fagn) - 1.0)
    for n, (t, pl, q, oa, R, i, law, ebv, T) in enumerate(z["cases"]):
        sk = sks.setdefault(R, SkirtorTorus(WAVE, mode="cigale", R=R))
        pd = PolarDust(WAVE, mode="cigale", law=LAWS[int(law)])
        th = {k: jnp.asarray(float(v)) for k, v in dict(
            fagn=fagn, skirtor_t=t, skirtor_pl=pl, skirtor_q=q, skirtor_oa=oa, skirtor_i=i,
            polar_ebv=ebv, polar_temp=T, polar_emis=1.6).items()}
        wl = z[f"{n}/wl"]
        assert np.array_equal(wl, sk.wl_nm)
        ref = {"disk": z[f"{n}/disk"], "dust": z[f"{n}/dust"] - z[f"{n}/polar"],
               "polar": z[f"{n}/polar"]}
        nat = sk.native_parts(th, pd)
        ctx = AGNContext(jnp.asarray(WAVE), jnp.asarray(1.0), None, jnp.ones(WAVE.size),
                         "cigale", jnp.asarray(ldust))
        p = sk.nuclear_parts(th, ctx, pd)
        lam = WAVE / 10.0
        for k, mine in (("disk", p["disk"] * p["polar_t"]), ("dust", p["dust"]),
                        ("polar", p["polar_ir"])):
            if np.any(ref[k]):
                worst = max(worst, _rel(nat[k], ref[k]))
                r = P * np.interp(lam, wl, ref[k], left=0, right=0) * lam ** 2 / 2.99792458e17
                worst = max(worst, _rel(mine, r))
            else:
                assert not np.any(np.asarray(mine))
        d = sk.derived(th, ctx, pd)
        info = z[f"{n}/info"]
        for g, r in ((d["disk_luminosity"] / P, info[0]), (d["polar_dust_luminosity"] / P, info[1]),
                     (d["accretion_power"] / (P * 0.493), info[2]),
                     (d["intrin_Lnu_2500A_30deg"] / P, info[3])):
            if r != 0:
                worst = max(worst, abs(float(g) / r - 1.0))
    print(f"T-PC-1: {len(z['cases'])} pcigale cases, worst rel {worst:.2e}")
    assert worst < 1e-10
