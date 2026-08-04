#!/usr/bin/env python
"""publish_grid_zenodo.py — publish an SSP grid to Zenodo and emit the
ceridwen REGISTRY entry.

Why this exists: users of ``CSPBasis_afe`` should never have to build the
alpha-enhanced grids themselves — that requires python-fsps compiled from
source with ``AFE_FLAG=1`` against FSPS v4.0 data.  Because the alpha
variant carries NO nebular model, nothing is read from ``$SPS_HOME`` at
fit time either, so a downloaded grid is the COMPLETE stellar input: with
this file published, fitting [alpha/Fe] needs no FSPS install at all.

Run this wherever the built grid and network access both exist (Tursa
login node or laptop after scp). Stdlib only — no requests, no numpy.

Two modes
---------
1. No token (default): compute sha256 + size, print the manual upload
   checklist and a ready-to-paste ``REGISTRY`` entry with the record id
   left as a placeholder.

       python scripts_afe/publish_grid_zenodo.py \
           --file ~/grids/ssp_data_afe_c3k.h5 --name amist_c3k_lr_chab_afe

2. With a token (``--token`` or ``$ZENODO_TOKEN``): drive the Zenodo API —
   create a new version of the ceridwen-grids deposit (``--deposit``,
   default 21221634, the record holding ssp_data.h5 / ssp_data_bpass.h5),
   upload the file under the canonical name ``<name>.h5``, and print the
   final REGISTRY entry with the NEW record id filled in.  The draft is
   left unpublished for a final look in the browser unless ``--publish``
   is passed.  (A new Zenodo version gets a new record id; old links and
   the concept DOI keep working.)

After publishing, paste the printed entry into
``ceridwen/ssps/grid_fetch.py`` and users are one call away:

    from ceridwen.ssps import fetch_grid, SSPDataAfe
    ssp = SSPDataAfe.load(fetch_grid("amist_c3k_lr_chab_afe"))
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path

API = "https://zenodo.org/api"


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def registry_entry(name: str, record_id, sha: str, size_mb: int,
                   notes: str) -> str:
    rid = record_id if record_id is not None else "<record_id>"
    return (
        f'    "{name}": {{\n'
        f'        "url": "https://zenodo.org/records/{rid}/files/'
        f'{name}.h5?download=1",\n'
        f'        "sha256": "{sha}",\n'
        f'        "size_mb": {size_mb},\n'
        f'        "notes": {notes!r},\n'
        f'    }},'
    )


def _req(method: str, url: str, token: str, data=None, headers=None):
    hdrs = {"Authorization": f"Bearer {token}"}
    if headers:
        hdrs.update(headers)
    body = None
    if data is not None and not isinstance(data, (bytes, bytearray)):
        body = json.dumps(data).encode()
        hdrs["Content-Type"] = "application/json"
    elif data is not None:
        body = data
    req = urllib.request.Request(url, data=body, headers=hdrs, method=method)
    with urllib.request.urlopen(req) as r:
        txt = r.read()
    return json.loads(txt) if txt else {}


def api_publish(path: Path, name: str, token: str, deposit: int,
                sha: str, publish: bool) -> int:
    """New version of ``deposit`` + upload; returns the new record id."""
    print(f"[zenodo] creating new version of deposit {deposit} ...")
    nv = _req("POST", f"{API}/deposit/depositions/{deposit}/actions/newversion",
              token)
    draft_url = nv["links"]["latest_draft"]
    draft = _req("GET", draft_url, token)
    new_id = draft["id"]
    bucket = draft["links"]["bucket"]

    fname = f"{name}.h5"
    print(f"[zenodo] uploading {path} as {fname} "
          f"({path.stat().st_size / 1e6:.0f} MB) ...")
    with open(path, "rb") as f:
        _req("PUT", f"{bucket}/{fname}", token, data=f.read(),
             headers={"Content-Type": "application/octet-stream"})

    up = _req("GET", f"{API}/deposit/depositions/{new_id}/files", token)
    remote = {e["filename"]: e["checksum"] for e in up}
    md5_local = hashlib.md5(path.read_bytes()).hexdigest()  # noqa: S324 — Zenodo reports md5
    if remote.get(fname) not in (md5_local, f"md5:{md5_local}"):
        raise SystemExit(f"[zenodo] checksum mismatch after upload: "
                         f"{remote.get(fname)} != md5:{md5_local}")
    print(f"[zenodo] upload verified (md5 {md5_local}).")

    if publish:
        _req("POST", f"{API}/deposit/depositions/{new_id}/actions/publish",
             token)
        print(f"[zenodo] PUBLISHED: https://zenodo.org/records/{new_id}")
    else:
        print(f"[zenodo] draft ready (NOT published): "
              f"https://zenodo.org/uploads/{new_id}\n"
              f"[zenodo] review in the browser, press Publish, and the "
              f"record id stays {new_id}.")
    return new_id


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--file", required=True, help="built grid .h5")
    ap.add_argument("--name", required=True,
                    help="registry key, e.g. amist_c3k_lr_chab_afe "
                         "(uploaded as <name>.h5)")
    ap.add_argument("--deposit", type=int, default=21221634,
                    help="existing ceridwen-grids record to version "
                         "(default 21221634)")
    ap.add_argument("--token", default=os.environ.get("ZENODO_TOKEN"),
                    help="Zenodo personal token (or $ZENODO_TOKEN); omit "
                         "for the manual checklist")
    ap.add_argument("--publish", action="store_true",
                    help="publish the draft immediately (default: leave "
                         "draft for review)")
    ap.add_argument("--notes", default="",
                    help="notes string for the REGISTRY entry")
    args = ap.parse_args()

    path = Path(args.file).expanduser()
    if not path.is_file():
        raise SystemExit(f"no such file: {path}")

    print(f"[grid] {path}")
    sha = sha256_of(path)
    size_mb = round(path.stat().st_size / 1e6)
    print(f"[grid] sha256 = {sha}")
    print(f"[grid] size   = {size_mb} MB")

    notes = args.notes or (
        "FSPS v4.0 alpha-MC (python-fsps >= 0.4.9.dev, AFE_FLAG=1), "
        "aMIST + C3K_LR, Chabrier IMF, [alpha/Fe] = {-0.2, 0.0, +0.2, "
        "+0.4, +0.6}. For CSPBasis_afe (no nebular; no FSPS needed at "
        "fit time)."
    ) if args.name == "amist_c3k_lr_chab_afe" else args.notes

    record_id = None
    if args.token:
        record_id = api_publish(path, args.name, args.token,
                                args.deposit, sha, args.publish)
    else:
        print(
            "\n[manual] no token given — upload by hand:\n"
            f"  1. https://zenodo.org/records/{args.deposit} -> "
            "'New version'\n"
            f"  2. add the file, renamed to: {args.name}.h5\n"
            "  3. Publish; note the NEW record id in the URL\n"
        )

    print("\nPaste into ceridwen/ssps/grid_fetch.py REGISTRY "
          "(replace the existing entry):\n")
    print(registry_entry(args.name, record_id, sha, size_mb, notes))
    if record_id is None:
        print("\n(replace <record_id> with the new record id from step 3)")


if __name__ == "__main__":
    main()
