import jax
jax.config.update("jax_enable_x64", True)

from .dust import DustModel, DustEmission
from .neb import NebularModel
from .fit import fitSED, read_result_h5

try:
    from ._version import __version__
except ImportError:  # pragma: no cover - _version may be absent in a bare checkout
    __version__ = "0.0.0+unknown"

# Deploy/build stamp.  ``ceridwen/_buildstamp.py`` is written by the deploy
# tooling at push time and records the git short hash + dirty flag + UTC
# timestamp of the source tree the install was pushed from; greppable on the
# cluster to verify which ceridwen revision a job is running.  Absent in a
# bare/dev checkout, in which case the hash is None.
#
# WARNING: keep the ``_buildstamp`` import SEPARATE from the ``_version``
# import above.  ``_version.py`` does not define ``__githash__``, so combining
# them makes the import always raise ImportError and silently mask the real
# ``__version__`` as "0.0.0+unknown".
try:
    from ._buildstamp import __githash__, __dirty__, __build_time__
except ImportError:  # pragma: no cover - stamp absent in a dev checkout
    __githash__ = None
    __dirty__ = None
    __build_time__ = None

__all__ = [
    "DustModel",
    "DustEmission",
    "NebularModel",
    "fitSED",
    "read_result_h5",
    "__version__",
    "__githash__",
]
