"""Convert the Feltre et al. (2016) AGN narrow-line-region grid to ``nlr_feltre16.h5``.

Input: the 16 ASCII files ``nlr_nebular_Z<digits>.txt`` of the public Feltre+16 NLR grid (one file
per gas metallicity Z; ``Z0001`` = 0.0001, ``Z014`` = 0.014; the file name is the only record of
Z, so ``Z017`` is read as 0.017, not the 0.01774 of Feltre+16 Table 1, which the public files do
not carry: they hold 16 Z incl. 0.017 and 0.020 and log U_S -5..-1 in 0.5 dex, wider than Table 1).
Download: https://www.iap.fr/neogal/ewExternalFiles/AGN_NLR_nebular_feltre16.tar.gz (the tarball
may be passed directly). Each has a ``#`` header naming the
columns ``log(Us) xid nh alpha <line> <line> ...`` and one row per (log Us, xi_d, n_H, alpha) node,
the line luminosities in erg/s as tabulated.

Output (``$CERIDWEN_AGN_DATA/nlr_feltre16.h5`` by default; ``--out`` to change it):

    /axes/Z, /axes/logz (= log10 Z, absolute), /axes/logu, /axes/xid, /axes/nh, /axes/lognh,
    /axes/alpha                            ascending, float64
    /L        (n_lines, nZ, nU, nxid, nnh, nalpha)  line luminosities as tabulated [erg/s]
    /ratio    (n_lines, nZ, nU, nxid, nnh, nalpha)  L / L(H beta)
    /lines/source_name   the column names of the source files
    /lines/fsps_name     the FSPS ``emlines_info.dat`` name of each line ("" = none)
    /lines/fsps_index    its row in that list (-1 = none)
    /lines/wave_vac      its vacuum wavelength from that list [A] (NaN = none)
    attrs: provenance (source directory, sha256 of every input file, the FSPS line list and its
    sha256, converter path, date), units, axis order.

Every node of the full Cartesian product must be present exactly once (else ValueError), every
luminosity finite and > 0, H beta among the columns. Lines are matched to FSPS BY NAME through
``FELTRE_TO_FSPS`` below, then checked against the FSPS list in use (the name must exist there and
its wavelength must agree with the line's nominal wavelength to 3 A); a mismatch is an error.

Usage::

    python scripts/agn2/build_nlr_h5.py <dir with nlr_nebular_Z*.txt> [--out PATH] [--sps-home DIR]
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import os
import re
import sys
from pathlib import Path

import numpy as np

# Feltre+16 column name -> (FSPS emlines_info.dat name, nominal wavelength [A] for the check).
# None: FSPS has no such line (N V 1240, Si III] 1883/1888); the line is kept in the file with
# fsps_name "" and fsps_index -1 so that the component can decide (see ceridwen/agn/nlr.py).
FELTRE_TO_FSPS = {
    # The grid gives the [O II] 3726+3729 doublet SUM only (Feltre+16 sec. 3: "[O ii]3726,3729
    # (hereafter simply [O ii]3727)"). The whole sum goes on '[O II] 3726', the convention of the
    # package's 'a42' AGN lines (ceridwen/agn/data/richardson14_a42.csv, after Prospector).
    "[OII]3727": ("[O II] 3726", 3727.1),
    "Hbeta": ("Ba-beta 4861", 4862.7),
    "[OIII]4959": ("[O III] 4959", 4960.3),
    "[OIII]5007": ("[O III] 5007", 5008.2),
    "[OI]6300": ("[O I] 6300", 6302.0),
    "[NII]6548": ("[N II] 6548", 6549.9),
    "Halpha": ("Ba-alpha 6563", 6564.6),
    "[NII]6584": ("[N II] 6584", 6585.3),
    "[SII]6717": ("[S II] 6716", 6718.3),
    "[SII]6731": ("[S II] 6731", 6732.7),
    "NV1240": (None, 1240.0),
    "CIV1548": ("C IV 1548", 1548.2),
    "CIV1551": ("C IV 1551", 1550.8),
    "HeII1640": ("He II 1640", 1640.4),
    "OIII]1661": ("O III] 1661", 1660.8),
    "OIII]1666": ("O III] 1666", 1666.2),
    "[SiIII]1883": (None, 1882.7),
    "SiIII]1888": (None, 1892.0),
    "[CIII]1907": ("[C III] 1907", 1906.7),
    "CIII]1910": ("C III] 1910", 1908.7),
}
HBETA = "Hbeta"
AXIS_COLUMNS = ("log(Us)", "xid", "nh", "alpha")
FILE_RE = re.compile(r"^nlr_nebular_Z(\d+)\.txt$")


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def z_from_filename(name: str) -> float:
    """``nlr_nebular_Z0001.txt`` -> 0.0001, ``nlr_nebular_Z014.txt`` -> 0.014."""
    m = FILE_RE.match(name)
    if not m:
        raise ValueError(f"{name}: not a Feltre+16 NLR file name (nlr_nebular_Z<digits>.txt)")
    return float("0." + m.group(1))


def read_feltre_file(path):
    """``(columns, data)`` of one file: ``columns`` the header names, ``data`` (n_rows, n_cols)."""
    with open(path) as f:
        head = f.readline()
    if not head.startswith("#"):
        raise ValueError(f"{path}: expected a '#' header line naming the columns")
    cols = head[1:].split()
    data = np.loadtxt(path, dtype=np.float64, comments="#", ndmin=2)
    if data.shape[1] != len(cols):
        raise ValueError(f"{path}: {data.shape[1]} columns of numbers, {len(cols)} names in header")
    if tuple(cols[:4]) != AXIS_COLUMNS:
        raise ValueError(f"{path}: first columns {cols[:4]}, expected {AXIS_COLUMNS}")
    return cols, data


def read_fsps_lines(sps_home):
    path = Path(sps_home) / "data" / "emlines_info.dat"
    wave, names = [], []
    with open(path) as f:
        for row in f:
            parts = row.split(",", 1)
            if len(parts) == 2:
                wave.append(float(parts[0]))
                names.append(parts[1].strip())
    return np.asarray(wave, dtype=np.float64), names, path


def build_grid(src_dir):
    """Read every file and return ``dict(Z, logu, xid, nh, alpha, columns, L, files)``;
    ``L`` (n_lines, nZ, nU, nxid, nnh, nalpha) in erg/s."""
    src_dir = Path(src_dir)
    files = sorted((p for p in src_dir.iterdir() if FILE_RE.match(p.name)),
                   key=lambda p: z_from_filename(p.name))
    if not files:
        raise ValueError(f"{src_dir}: no nlr_nebular_Z*.txt files")
    Z = np.asarray([z_from_filename(p.name) for p in files])
    if np.any(np.diff(Z) <= 0):
        raise ValueError(f"duplicate metallicities in {src_dir}: {Z}")
    tables, columns = [], None
    for p in files:
        cols, data = read_feltre_file(p)
        if columns is None:
            columns = cols
        elif cols != columns:
            raise ValueError(f"{p}: header {cols} differs from {files[0]}: {columns}")
        tables.append(data)
    axes = [np.unique(np.concatenate([t[:, k] for t in tables])) for k in range(4)]
    shape = tuple(a.size for a in axes)
    n_lines = len(columns) - 4
    L = np.full((n_lines, Z.size) + shape, np.nan)
    for iz, (p, t) in enumerate(zip(files, tables)):
        if t.shape[0] != int(np.prod(shape)):
            raise ValueError(f"{p}: {t.shape[0]} rows, the full grid {shape} has {np.prod(shape)}")
        idx = [np.searchsorted(axes[k], t[:, k]) for k in range(4)]
        for k in range(4):
            if not np.array_equal(axes[k][idx[k]], t[:, k]):
                raise ValueError(f"{p}: axis {AXIS_COLUMNS[k]} value off the union grid")
        flat = np.ravel_multi_index(idx, shape)
        if np.unique(flat).size != flat.size:
            raise ValueError(f"{p}: duplicated grid nodes")
        block = np.full((n_lines,) + shape, np.nan)
        block.reshape(n_lines, -1)[:, flat] = t[:, 4:].T
        L[:, iz] = block
    if not np.all(np.isfinite(L)) or np.any(L <= 0):
        raise ValueError("the grid has missing nodes or non-positive luminosities")
    return dict(Z=Z, logu=axes[0], xid=axes[1], nh=axes[2], alpha=axes[3],
                columns=columns[4:], L=L, files=files)


def match_lines(columns, sps_home):
    """Per column: (fsps_name, fsps_index, wave_vac) via ``FELTRE_TO_FSPS``, checked against the
    FSPS list in use (the name must exist, its wavelength within 3 A of the nominal one)."""
    fwave, fnames, fpath = read_fsps_lines(sps_home)
    pos = {n: i for i, n in enumerate(fnames)}
    out, bad = [], []
    for c in columns:
        if c not in FELTRE_TO_FSPS:
            bad.append(f"{c!r}: not in FELTRE_TO_FSPS (add it with its FSPS name)")
            continue
        name, lam = FELTRE_TO_FSPS[c]
        if name is None:
            out.append(("", -1, np.nan))
        elif name not in pos:
            bad.append(f"{c!r} -> {name!r}: no such name in {fpath}")
        elif abs(fwave[pos[name]] - lam) > 3.0:
            bad.append(f"{c!r} -> {name!r} at {fwave[pos[name]]:.2f} A, expected ~{lam} A")
        else:
            out.append((name, pos[name], float(fwave[pos[name]])))
    if bad:
        raise ValueError("Feltre+16 lines not matched to the FSPS line list: " + "; ".join(bad))
    return out, fpath


def write_h5(grid, matched, fsps_path, out, source=("", "")):
    """``source``: (path of the input directory or tarball, sha256 of the tarball or "")."""
    import h5py
    names = list(grid["columns"])
    if HBETA not in names:
        raise ValueError(f"no {HBETA!r} column; the ratios are normalised to H beta")
    L = grid["L"]
    ratio = L / L[names.index(HBETA)][None]
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".h5.tmp")
    with h5py.File(tmp, "w") as h:
        ax = h.create_group("axes")
        ax["Z"] = grid["Z"]
        ax["logz"] = np.log10(grid["Z"])
        ax["logu"] = grid["logu"]
        ax["xid"] = grid["xid"]
        ax["nh"] = grid["nh"]
        ax["lognh"] = np.log10(grid["nh"])
        ax["alpha"] = grid["alpha"]
        h["L"] = L
        h["ratio"] = ratio
        g = h.create_group("lines")
        g["source_name"] = np.asarray(names, dtype="S")
        g["fsps_name"] = np.asarray([m[0] for m in matched], dtype="S")
        g["fsps_index"] = np.asarray([m[1] for m in matched], dtype=np.int64)
        g["wave_vac"] = np.asarray([m[2] for m in matched], dtype=np.float64)
        a = h.attrs
        a["format"] = "ceridwen nlr_feltre16 v1"
        a["axis_order"] = "line, Z, logu, xid, nh, alpha"
        a["units_L"] = ("erg/s for an accretion luminosity of 1e45 erg/s (NEOGAL README; "
                        "linear in it); only the ratios are used")
        a["units_ratio"] = "L_line / L_Hbeta"
        a["hbeta_source_name"] = HBETA
        a["source"] = ("Feltre, Charlot & Gutkin 2016, MNRAS 456, 3354: AGN narrow-line-region "
                       "photoionisation grid (nlr_nebular_Z*.txt)")
        a["source_path"] = source[0]
        a["source_tarball_sha256"] = source[1]
        a["source_url"] = ("https://www.iap.fr/neogal/ewExternalFiles/"
                           "AGN_NLR_nebular_feltre16.tar.gz")
        a["source_files"] = ";".join(p.name for p in grid["files"])
        a["source_sha256"] = ";".join(_sha256(p) for p in grid["files"])
        a["fsps_emlines_file"] = str(fsps_path)
        a["fsps_emlines_sha256"] = _sha256(fsps_path)
        a["converter"] = "scripts/agn2/build_nlr_h5.py"
        a["created"] = datetime.datetime.now().isoformat(timespec="seconds")
    os.replace(tmp, out)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("src", help="directory with nlr_nebular_Z*.txt, or the NEOGAL tarball "
                    "AGN_NLR_nebular_feltre16.tar.gz")
    default_out = Path(os.environ.get("CERIDWEN_AGN_DATA", Path.home() / ".ceridwen" / "agn"))
    ap.add_argument("--out", default=str(default_out / "nlr_feltre16.h5"))
    ap.add_argument("--sps-home", default=os.environ.get("SPS_HOME"))
    args = ap.parse_args(argv)
    if not args.sps_home:
        sys.exit("set $SPS_HOME or pass --sps-home: lines are matched to FSPS's emlines_info.dat")
    src = Path(args.src)
    if src.is_file() and src.name.endswith((".tar.gz", ".tgz")):
        import tarfile
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            with tarfile.open(src) as tf:
                tf.extractall(tmp, filter="data")
            dirs = {p.parent for p in Path(tmp).rglob("nlr_nebular_Z*.txt")}
            if len(dirs) != 1:
                sys.exit(f"{src}: expected one directory of nlr_nebular_Z*.txt, found {dirs}")
            grid = build_grid(dirs.pop())
            matched, fpath = match_lines(grid["columns"], args.sps_home)
            out = write_h5(grid, matched, fpath, args.out,
                           source=(str(src.resolve()), _sha256(src)))
    else:
        grid = build_grid(src)
        matched, fpath = match_lines(grid["columns"], args.sps_home)
        out = write_h5(grid, matched, fpath, args.out,
                       source=(str(src.resolve()), ""))
    print(f"wrote {out}: {grid['L'].shape[0]} lines x grid "
          f"{tuple(grid['L'].shape[1:])} (Z, logU, xid, nH, alpha)")


if __name__ == "__main__":
    main()
