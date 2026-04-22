import jax
jax.config.update("jax_enable_x64", True)

from .dust import DustModel, DustEmission
from .neb import NebularModel
from .fit import fitSED, read_result_h5
try:
    from ._version import __version__, __githash__
except(ImportError):
    pass
