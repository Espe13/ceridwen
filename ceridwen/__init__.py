import jax
jax.config.update("jax_enable_x64", True)


def _patch_tfp_jax_compat():
    """Restore symbols newer JAX removed from ``jax.interpreters.xla`` that the
    tensorflow-probability JAX substrate still imports.

    TFP references e.g. ``jax.interpreters.xla.pytype_aval_mappings``, which JAX
    moved to ``jax.core`` (same object) and then removed the old alias. We re-add
    the alias so ``tensorflow_probability.substrates.jax`` imports on JAX >= 0.7.
    Harmless on older JAX (the attributes already exist).
    """
    import warnings
    try:
        import jax.interpreters.xla as _xla
        import jax.core as _core
        # jax.core members are themselves deprecated; probing them warns. We
        # only need the object, so silence the deprecation during the copy.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            for _name in ("pytype_aval_mappings", "abstractify"):
                if not hasattr(_xla, _name) and hasattr(_core, _name):
                    setattr(_xla, _name, getattr(_core, _name))
    except Exception:  # pragma: no cover - never let the shim break import
        pass


_patch_tfp_jax_compat()

from .dust import DustModel, DustEmission
from .neb import NebularModel
from .fit import fitSED, read_result_h5
# NB: do NOT import .check here — it is run as `python -m ceridwen.check`, and
# importing it in the package __init__ triggers a runpy double-import warning.
# Use it via the CLI, or `from ceridwen.check import check_environment`.

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
