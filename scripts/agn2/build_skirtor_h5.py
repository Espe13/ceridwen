"""Convert the SKIRTOR 2016 templates (Stalevski+2012, 2016) into ``skirtor2016.h5`` for
``ceridwen.agn.skirtor.SkirtorTorus``.

The arrays written are exactly the ones CIGALE stores in its ``skirtor2016`` database
(``database_builder/skirtor2016/__init__.py`` at cigale 29cb909f, 2025.1), recomputed here from
the SKIRTOR source files, step by step:

1. read columns (lambda [um], direct, scattered, dust) of ``t{t}_p{pl}_q{q}_oa{oa}_R{R}_Mcl{Mcl}_
   i{i}_sed.dat`` (lambda F_lambda [W/m2]); lambda -> nm; ``disk = (direct + scattered) / lambda``,
   ``dust = dust / lambda`` (F_lambda per nm, up to a constant);
2. extend to 1e7 nm with the points 2e6, 4e6, 8e6, 1e7 nm: disk 1e-99 (and its last native point
   set to 1e-99), dust a power law through the last two native points (1e-99 if the last is 0);
3. above 3e4 nm, replace the grid by CIGALE's dense grid (the ``nebular_continuum`` wavelengths in
   [3e4, 1e7] nm), log-log interpolation from the points above 1e4 nm;
4. ``norm = trapz(dust, wl)``; ``disk /= norm``; ``dust /= norm`` (dust integrates to 1 W).

CIGALE's dense grid is not derivable from the SKIRTOR files (it is the union of CIGALE's stellar,
dust and nebular grids): it is read from a built pcigale database (``--cigale-db``, the directory
``pcigale/data/skirtor2016`` of pickles; every node is then also compared with CIGALE's own
arrays and the largest relative difference is recorded) or from a ``.npy`` (``--dense-wl``).
``--dev-dense N`` uses N log-spaced points instead and marks the file ``cigale_exact = False``
(development only; parity tests refuse such a file).

Layout (float64 throughout):

    axes/t (5,), axes/pl (4,), axes/q (4,), axes/oa (8,) [deg], axes/i (10,) [deg],
    axes/R (3,), attrs Mcl
    wl_nm (n_wl,)                         ascending, nm
    disk  (nR, nt, npl, nq, noa, ni, n_wl) F_lambda per nm, per unit torus-dust luminosity
    dust  (nR, nt, npl, nq, noa, ni, n_wl) idem; trapz(dust, wl_nm) = 1 for every node
    norm  (nR, nt, npl, nq, noa, ni)      CIGALE's ``norm`` (the raw dust integral)
    attrs: provenance (source directory, sha256 of the sorted per-file sha256 list, number of
           files, CIGALE commit the steps follow, dense-grid source, max rel diff vs CIGALE)

Usage::

    python scripts/agn2/build_skirtor_h5.py --raw-dir <dir of *_sed.dat> \
        --cigale-db <.../pcigale/data/skirtor2016> [--out $CERIDWEN_AGN_DATA/skirtor2016.h5]
"""
from __future__ import annotations

import argparse
import hashlib
import os
import pickle
import re
import sys
import time
from pathlib import Path

import numpy as np

CIGALE_COMMIT = "29cb909f (cigale 2025.1, database_builder/skirtor2016/__init__.py)"
NAME_RE = re.compile(r"^t(?P<t>[^_]+)_p(?P<pl>[^_]+)_q(?P<q>[^_]+)_oa(?P<oa>[^_]+)_R(?P<R>[^_]+)"
                     r"_Mcl(?P<Mcl>[^_]+)_i(?P<i>[^_]+)_sed\.dat$")
WL_EXT = np.array([2e6, 4e6, 8e6, 1e7])
DENSE_LO, DENSE_HI = 3e4, 1e7


def _trapz(y, x):
    return np.trapezoid(y, x=x)


def default_out():
    root = os.environ.get("CERIDWEN_AGN_DATA") or str(Path.home() / ".ceridwen" / "agn")
    return Path(root) / "skirtor2016.h5"


class _Entry:
    """Stand-in for ``pcigale.data.SimpleDatabaseEntry`` (unpickled without pcigale)."""

    def __setstate__(self, state):
        self.__dict__.update(state)


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == "SimpleDatabaseEntry":
            return _Entry
        return super().find_class(module, name)


def read_cigale_entry(db_dir, t, pl, q, oa, R, Mcl, i):
    """CIGALE's stored entry (``SimpleDatabase.get``: keys sorted, ``k=v`` joined by '_')."""
    keys = {"t": int(t), "pl": float(pl), "q": float(q), "oa": int(oa), "R": int(R),
            "Mcl": float(Mcl), "i": int(i)}
    base = "_".join(f"{k}={v}" for k, v in sorted(keys.items()))
    with open(Path(db_dir) / f"{base}.pickle", "rb") as f:
        return _Unpickler(f).load()


def cigale_steps(path, wl_den):
    """(wl, disk, dust, norm) of one SKIRTOR file, CIGALE's builder steps 1-4."""
    wl, disk, scatt, dust = np.genfromtxt(path, unpack=True, usecols=(0, 2, 3, 4))
    wl = wl * 1e3
    disk = disk + scatt
    disk = disk / wl
    dust = dust / wl
    disk_ext = np.zeros(len(WL_EXT)) + 1e-99
    if dust[-1] == 0:
        dust_ext = np.zeros(len(WL_EXT)) + 1e-99
    else:
        dust_ext = 10 ** (np.log10(dust[-1]) + np.log10(WL_EXT / wl[-1])
                          * np.log10(dust[-2] / dust[-1]) / np.log10(wl[-2] / wl[-1]))
    wl = np.append(wl, WL_EXT)
    disk[-1] = 1e-99
    disk = np.append(disk, disk_ext)
    dust = np.append(dust, dust_ext)
    idx = np.where(wl > 1e4)
    disk_den = 10 ** np.interp(np.log10(wl_den), np.log10(wl[idx]), np.log10(disk[idx]))
    dust_den = 10 ** np.interp(np.log10(wl_den), np.log10(wl[idx]), np.log10(dust[idx]))
    idx = np.where(wl < 3e4)
    wl = np.append(wl[idx], wl_den)
    disk = np.append(disk[idx], disk_den)
    dust = np.append(dust[idx], dust_den)
    norm = _trapz(dust, wl)
    return wl, disk / norm, dust / norm, norm


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scan(raw_dir):
    files = {}
    for p in sorted(Path(raw_dir).glob("*_sed.dat")):
        m = NAME_RE.match(p.name)
        if m is None:
            raise ValueError(f"{p}: not a SKIRTOR file name t*_p*_q*_oa*_R*_Mcl*_i*_sed.dat")
        files[tuple(m.group(k) for k in ("t", "pl", "q", "oa", "R", "Mcl", "i"))] = p
    if not files:
        raise FileNotFoundError(f"no *_sed.dat in {raw_dir}")
    return files


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--raw-dir", required=True, help="directory of SKIRTOR *_sed.dat files")
    ap.add_argument("--cigale-db", help="pcigale/data/skirtor2016 (built CIGALE database)")
    ap.add_argument("--dense-wl", help=".npy of CIGALE's dense grid in [3e4, 1e7] nm")
    ap.add_argument("--dev-dense", type=int, default=0,
                    help="DEV ONLY: N log-spaced dense points (file marked cigale_exact=False)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    files = scan(a.raw_dir)
    keys = list(files)
    strs = {k: sorted({key[j] for key in keys}, key=float)
            for j, k in enumerate(("t", "pl", "q", "oa", "R", "Mcl", "i"))}
    if len(strs["Mcl"]) != 1:
        raise ValueError(f"expected one Mcl value, got {strs['Mcl']}")
    shape = tuple(len(strs[k]) for k in ("R", "t", "pl", "q", "oa", "i"))
    n_expected = int(np.prod(shape))
    if len(files) != n_expected:
        raise ValueError(f"{len(files)} files do not fill the rectilinear grid {shape} "
                         f"({n_expected}); a node is missing")

    any_key = keys[0]
    if a.cigale_db:
        e = read_cigale_entry(a.cigale_db, *any_key)
        wl_db = np.asarray(e.wl, dtype=np.float64)
        wl_den = wl_db[wl_db >= DENSE_LO]
        dense_src = f"pcigale database {a.cigale_db}"
        exact = True
    elif a.dense_wl:
        wl_den = np.load(a.dense_wl).astype(np.float64)
        dense_src = f"{a.dense_wl} (sha256 {_sha256(a.dense_wl)})"
        exact = True
    elif a.dev_dense > 1:
        wl_den = np.geomspace(DENSE_LO, DENSE_HI, a.dev_dense)
        dense_src = f"DEV: {a.dev_dense} log-spaced points in [3e4, 1e7] nm (NOT CIGALE's grid)"
        exact = False
    else:
        raise SystemExit("give --cigale-db, --dense-wl or (development only) --dev-dense N")

    wl0 = cigale_steps(files[any_key], wl_den)[0]
    disk = np.empty(shape + (wl0.size,))
    dust = np.empty(shape + (wl0.size,))
    norm = np.empty(shape)
    hashes = []
    max_rel = 0.0
    t0 = time.time()
    order = [(iR, it, ip, iq, io_, ii)
             for iR in range(shape[0]) for it in range(shape[1]) for ip in range(shape[2])
             for iq in range(shape[3]) for io_ in range(shape[4]) for ii in range(shape[5])]
    for n, (iR, it, ip, iq, io_, ii) in enumerate(order):
        key = (strs["t"][it], strs["pl"][ip], strs["q"][iq], strs["oa"][io_], strs["R"][iR],
               strs["Mcl"][0], strs["i"][ii])
        p = files[key]
        hashes.append(f"{p.name} {_sha256(p)}")
        wl, dk, ds, nm = cigale_steps(p, wl_den)
        if not np.array_equal(wl, wl0):
            raise ValueError(f"{p}: wavelength grid differs from {files[any_key]}")
        disk[iR, it, ip, iq, io_, ii] = dk
        dust[iR, it, ip, iq, io_, ii] = ds
        norm[iR, it, ip, iq, io_, ii] = nm
        if a.cigale_db:
            e = read_cigale_entry(a.cigale_db, *key)
            for mine, theirs in ((dk, e.disk), (ds, e.dust), (wl, e.wl)):
                theirs = np.asarray(theirs, dtype=np.float64)
                big = np.abs(theirs) > 1e-6 * np.abs(theirs).max()
                rel = np.max(np.abs(mine[big] - theirs[big]) / np.abs(theirs[big]))
                max_rel = max(max_rel, float(rel))
            max_rel = max(max_rel, abs(nm - float(e.norm)) / abs(float(e.norm)))
        if n % 2000 == 0:
            print(f"  {n}/{len(order)} ({time.time() - t0:.0f} s)", flush=True)
    if not (np.all(np.isfinite(disk)) and np.all(np.isfinite(dust))):
        raise ValueError("non-finite template values")

    out = Path(a.out) if a.out else default_out()
    out.parent.mkdir(parents=True, exist_ok=True)
    import h5py
    tmp = out.with_suffix(".h5.tmp")
    manifest = "\n".join(hashes).encode()
    with h5py.File(tmp, "w") as f:
        g = f.create_group("axes")
        for k in ("t", "pl", "q", "oa", "i", "R"):
            g.create_dataset(k, data=np.array([float(s) for s in strs[k]], dtype=np.float64))
        f.create_dataset("wl_nm", data=wl0)
        f.create_dataset("disk", data=disk, compression="gzip", compression_opts=4)
        f.create_dataset("dust", data=dust, compression="gzip", compression_opts=4)
        f.create_dataset("norm", data=norm)
        f.attrs["format"] = "ceridwen skirtor2016 v1"
        f.attrs["Mcl"] = float(strs["Mcl"][0])
        f.attrs["axis_order"] = "R, t, pl, q, oa, i, wl"
        f.attrs["units"] = ("wl_nm: nm; disk, dust: F_lambda per nm normalised so that "
                            "trapz(dust, wl_nm) = 1 (CIGALE's 1 W of torus dust)")
        f.attrs["source_dir"] = str(Path(a.raw_dir).resolve())
        f.attrs["source_n_files"] = len(files)
        f.attrs["source_manifest_sha256"] = hashlib.sha256(manifest).hexdigest()
        f.attrs["builder_steps_follow"] = CIGALE_COMMIT
        f.attrs["dense_grid_source"] = dense_src
        f.attrs["cigale_exact"] = bool(exact)
        f.attrs["max_rel_vs_cigale_db"] = max_rel if a.cigale_db else -1.0
        f.attrs["created"] = time.strftime("%Y-%m-%d %H:%M:%S")
        f.attrs["script"] = "scripts/agn2/build_skirtor_h5.py"
    os.replace(tmp, out)
    print(f"wrote {out}: grid {dict(zip(('R', 't', 'pl', 'q', 'oa', 'i'), shape))}, "
          f"{wl0.size} wavelengths, cigale_exact={exact}"
          + (f", max rel vs CIGALE db {max_rel:.3e}" if a.cigale_db else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
