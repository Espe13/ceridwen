import importlib.util
import os
import warnings as _warnings
import jax


def _apple_gpu_is_default_backend() -> str | None:
    """The default JAX backend when it is an Apple GPU (``"mtl"``, registered by jax-graft), else
    None.  Only asked when jax-graft is installed, so a CPU/CUDA install does not initialise its
    backend at import."""
    if importlib.util.find_spec("jax_graft") is None:
        return None
    try:
        backend = jax.default_backend()
    except Exception:                                         # noqa: BLE001  (plugin failed to load)
        return None
    return backend if backend in ("mtl", "mps") else None


# Precision.  float64 is the default (evidences, gradients and the sampler state need it).
# An Apple GPU has no float64: when jax-graft has made it JAX's default backend and
# CERIDWEN_X64 is unset, the package runs in float32 (docs/mac_gpu.md).  CERIDWEN_X64=0 forces
# float32 anywhere; CERIDWEN_X64=1 forces float64 (refused on an Apple GPU: use
# JAX_PLATFORMS=cpu for a float64 fit on such a Mac).
_x64_env = os.environ.get("CERIDWEN_X64")
_apple_gpu = _apple_gpu_is_default_backend()
if _x64_env is None:
    _x64 = _apple_gpu is None
elif _x64_env == "0":
    _x64 = False
else:
    _x64 = True
    if _apple_gpu is not None:
        raise RuntimeError(
            f"CERIDWEN_X64={_x64_env!r} asks for float64, but JAX's default backend is the Apple GPU "
            f"({_apple_gpu}), which has no float64.  Unset CERIDWEN_X64 to run in float32 on the GPU, "
            "or set JAX_PLATFORMS=cpu for a float64 fit on the CPU (docs/mac_gpu.md)")
jax.config.update("jax_enable_x64", _x64)
if not _x64:
    _warnings.warn(
        ("float32 mode: JAX's default backend is the Apple GPU (%s), which has no float64.  "
         "Evidences, gradients and the sampler state are float32 (docs/mac_gpu.md); for a float64 fit "
         "set JAX_PLATFORMS=cpu" % _apple_gpu) if _x64_env is None else
        "CERIDWEN_X64=0: float64 is OFF for this process (float32 mode: evidences, gradients and "
        "sampler state are float32).  See docs/mac_gpu.md for what this mode guarantees; float64 "
        "(unset) is the default for fits",
        stacklevel=2)

# CERIDWEN_MATMUL_PRECISION=high: TF32 matmuls (off by default)
_matmul_prec = os.environ.get("CERIDWEN_MATMUL_PRECISION")
if _matmul_prec:
    jax.config.update("jax_default_matmul_precision", _matmul_prec)
    _warnings.warn(f"CERIDWEN_MATMUL_PRECISION={_matmul_prec!r}: JAX's default matmul "
                   "precision is changed for this process (e.g. TF32 on GPU); results are not "
                   "byte-comparable with the default", stacklevel=2)


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
