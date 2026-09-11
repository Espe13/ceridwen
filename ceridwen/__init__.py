import os
import jax
jax.config.update("jax_enable_x64", os.environ.get("CERIDWEN_X64", "1") != "0")

# CERIDWEN_MATMUL_PRECISION=high: TF32 matmuls (off by default)
_matmul_prec = os.environ.get("CERIDWEN_MATMUL_PRECISION")
if _matmul_prec:
    jax.config.update("jax_default_matmul_precision", _matmul_prec)


def _patch_tfp_jax_compat():
    """Re-add ``jax.interpreters.xla`` aliases that tensorflow-probability's JAX substrate imports."""
    import warnings
    try:
        import jax.interpreters.xla as _xla
        import jax.core as _core
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            for _name in ("pytype_aval_mappings", "abstractify"):
                if not hasattr(_xla, _name) and hasattr(_core, _name):
                    setattr(_xla, _name, getattr(_core, _name))
    except Exception:  # pragma: no cover
        pass


_patch_tfp_jax_compat()

from .dust import DustModel, DustEmission
from .neb import NebularModel
from .fit import fitSED, read_result_h5, load_result_h5, result_cosmology
from .postprocess import PostProcess, SpectrumSample, load_postprocess
from . import plotting

from .ssps import SSPData
from .csp import CSPBasis
from .model import SedModel
from .cosmology import Cosmology, DEFAULT_COSMO
from .broadening import Kinematics, Instrument, TIED, DEFAULT_KINEMATICS

try:
    from ._version import __version__
except ImportError:  # pragma: no cover
    __version__ = "0.0.0+unknown"

# keep separate from the _version import: _version.py has no __githash__
try:
    from ._buildstamp import __githash__, __dirty__, __build_time__
except ImportError:  # pragma: no cover
    __githash__ = None
    __dirty__ = None
    __build_time__ = None

__all__ = [
    "Kinematics", "Instrument", "TIED", "DEFAULT_KINEMATICS",
    "SSPData",
    "CSPBasis",
    "SedModel",
    "DustModel",
    "DustEmission",
    "NebularModel",
    "Cosmology",
    "DEFAULT_COSMO",
    "fitSED",
    "read_result_h5",
    "load_result_h5",
    "result_cosmology",
    "PostProcess",
    "SpectrumSample",
    "load_postprocess",
    "plotting",
    "__version__",
    "__githash__",
]
