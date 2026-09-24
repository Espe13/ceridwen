"""Per-observation parameter names: one naming rule for every parameter that belongs
to a single observation -- the outlier mixture, the noise nuisance terms, the spectrum
calibration and the spectrum GP likelihood.

A family has a *stem* per observation kind (``f_outlier_spec``, ``log_jitter_phot``,
``spectrum_scaling``).  Observation ``obs`` answers to ``<stem>_<obs.name>`` always, and to
the plain ``<stem>`` only when the model has exactly one observation of that kind; the plain
name with several is ambiguous and refused, as are both spellings for one observation and a
name that matches no observation (it would be sampled and never used).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

#: observation kind -> short suffix used in parameter names
KIND_SUFFIX = {"photometry": "phot", "spectrum": "spec", "lines": "lines"}


@dataclass(frozen=True)
class NameFamily:
    """``stem(kind_suffix) -> str`` for the kinds in ``kinds`` (a family that exists for one
    kind only, e.g. ``spectrum_scaling``, ignores the suffix).  Every name equal to ``root``
    or starting with ``root_`` belongs to the family and must match an observation."""
    label: str
    root: str
    stem: Callable[[str], str]
    kinds: tuple = ("photometry", "spectrum", "lines")

    def claims(self, name: str) -> bool:
        return name == self.root or name.startswith(self.root + "_")


def obs_kind(obs) -> Optional[str]:
    k = getattr(obs, "kind", None)
    return k if k in KIND_SUFFIX else None


def names_for(obs, family: NameFamily):
    """``(plain, own)`` names of ``obs`` in ``family``, or None when the family does not
    apply to its kind."""
    kind = obs_kind(obs)
    if kind is None or kind not in family.kinds:
        return None
    stem = family.stem(KIND_SUFFIX[kind])
    return stem, f"{stem}_{obs.name}"


def resolve_name(obs, family: NameFamily, given) -> Optional[str]:
    """The name in ``given`` that sets ``family`` for ``obs`` (own name first), or None."""
    nm = names_for(obs, family)
    if nm is None:
        return None
    return nm[1] if nm[1] in given else (nm[0] if nm[0] in given else None)


def check_names(observations, families, given, *, what="") -> dict:
    """Validate every name of ``families`` in ``given`` against ``observations``; return
    {valid name: obs}.  Raises ValueError for an ambiguous plain name, both spellings for one
    observation, or a name that matches no observation."""
    by_kind: dict = {}
    for o in observations:
        k = obs_kind(o)
        if k is not None:
            by_kind.setdefault(k, []).append(o)
    valid: dict = {}
    for fam in families:
        wanted = {n for n in given if fam.claims(n)}
        fam_valid = {}
        for kind in fam.kinds:
            obs = by_kind.get(kind, [])
            for o in obs:
                plain, own = names_for(o, fam)
                fam_valid[own] = o
                if len(obs) == 1:
                    fam_valid[plain] = o
                if len(obs) > 1 and plain in given:
                    raise ValueError(
                        f"{plain!r} is ambiguous: the model has {len(obs)} {kind} observations "
                        f"{[x.name for x in obs]}, and each takes its own {fam.label}; use "
                        f"{[names_for(x, fam)[1] for x in obs]}")
                if plain in given and own in given:
                    raise ValueError(f"both {plain!r} and {own!r} are set for {kind} "
                                     f"{o.name!r}; keep one")
        unknown = sorted(n for n in wanted if n not in fam_valid)
        if unknown:
            raise ValueError(
                f"{unknown} match no observation{what}, so they would be sampled without being "
                f"used; the {fam.label} names of this model are {sorted(fam_valid)}")
        valid.update(fam_valid)
    return valid


def family(label, root, kinds=("photometry", "spectrum", "lines"), per_kind=True):
    """A NameFamily whose stem is ``<root>_<kind suffix>`` (``per_kind``) or ``root``."""
    if per_kind:
        return NameFamily(label, root, lambda sfx, _r=root: f"{_r}_{sfx}", tuple(kinds))
    return NameFamily(label, root, lambda sfx, _r=root: _r, tuple(kinds))


#: the outlier mixture: f_outlier_<kind>[_<obs>], nsigma_outlier_<kind>[_<obs>]
OUTLIER_FAMILIES = (family("outlier fraction", "f_outlier"),
                    family("outlier width", "nsigma_outlier"))

#: the noise nuisance terms, one per observation: <root>_<kind>[_<obs>]
NOISE_ROOTS = ("log_err_scale", "log_jitter", "log_f_calib", "log_f_data")
NOISE_FAMILIES = tuple(family(f"noise term {r}", r) for r in NOISE_ROOTS)

#: the spectrum calibration, one per Spectrum: spectrum_scaling[_<obs>], ...
CALIB_ROOTS = ("spectrum_scaling", "spectrum_calib")
CALIB_FAMILIES = tuple(family(f"calibration {r}", r, kinds=("spectrum",), per_kind=False)
                       for r in CALIB_ROOTS)

#: the Gaussian-process likelihood of a spectrum, one per Spectrum:
#: log_gp_amp_spec[_<obs>] (ln a, a in units of sigma_eff), log_gp_length_spec[_<obs>]
#: (ln l, l in observed-frame Angstrom)
GP_ROOTS = ("log_gp_amp", "log_gp_length")
GP_FAMILIES = (family("GP amplitude log_gp_amp", "log_gp_amp", kinds=("spectrum",)),
               family("GP length scale log_gp_length", "log_gp_length", kinds=("spectrum",)))
