import jax
jax.config.update("jax_enable_x64", True)

from .dust import DustModel, DustEmission
from .neb import NebularModel
from .fit import fitSED, read_result_h5

try:
    from ._version import __version__, __githash__
except ImportError:  # pragma: no cover - _version may be absent in a bare checkout
    __version__ = "0.0.0+unknown"
    __githash__ = None

__all__ = [
    "DustModel",
    "DustEmission",
    "NebularModel",
    "fitSED",
    "read_result_h5",
    "__version__",
]
