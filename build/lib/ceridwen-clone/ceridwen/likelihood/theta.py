"""
ceridwen/likelihood/theta.py
====================================
ThetaVector: a JAX-compatible flat parameter vector with named-parameter access.

Motivation
----------
``CSPBasis`` packs all free parameters into a single flat ``jnp.ndarray`` and
accesses them via integer index attributes::

    theta[self.sfh_idx]          # shape (n_time,)
    theta[self.gas_logz_idx]     # shape (1,)

``DiagonalNoiseModel`` and the rest of the likelihood module expect nuisance
parameters to be accessible by name, i.e. ``params["log_jitter"]``.

``ThetaVector`` is the bridge.  It wraps a flat 1-D JAX array and a static
Python-level mapping of parameter names → integer index tuples.  Both access
patterns work on the same object::

    tv = ThetaVector(flat_array, csp.name_to_idx)

    tv[csp.sfh_idx]        # integer-array indexing (CSPBasis path)
    tv["log_jitter"]       # string-key indexing (noise model path)

JAX design
----------
``ThetaVector`` is registered as a JAX PyTree whose **only leaf** is the
underlying flat array.  The name-to-index mapping lives in the **auxiliary**
(static, hashable) data.  Consequently:

* ``jax.jit``, ``jax.grad``, and ``jax.vmap`` trace through the flat array
  exactly as if ``theta`` were a plain ``jnp.ndarray``.
* Retracing is triggered only when the *shape* or *dtype* of the flat array
  changes — never when parameter *values* change between calls.
* The Python-level dispatch in ``__getitem__`` (string vs array key) happens
  at trace time on concrete Python objects, not on traced values.  JAX never
  sees the ``isinstance`` check; it only sees the resulting array slice
  ``self._data[concrete_integer_tuple]``.

Index storage convention
------------------------
Internally all index arrays are converted to plain Python tuples of ``int`` at
construction time.  This makes them hashable and therefore safe to store as
PyTree auxiliary data.  JAX uses auxiliary data as part of the JIT cache key,
so the tuple representation is correct and efficient.

Construction
------------
Build directly or via the ``CSPBasis`` helper ``make_theta_vector``::

    # From CSPBasis
    tv_init = csp.make_theta_vector(csp.theta_init)

    # Manual construction
    name_to_idx = {"sfh": jnp.arange(100), "log_jitter": jnp.array([100])}
    tv = ThetaVector(flat_array, name_to_idx)

Integration with make_lnprobfn
-------------------------------
Pass a ``ThetaVector`` as the initial theta to blackjax (or any other sampler).
The sampler's kernel will update the underlying flat array; JAX will
automatically reconstruct ``ThetaVector`` instances with the same
name-to-index mapping via the PyTree unflatten function.

No changes are required in ``DiagonalNoiseModel``, ``DiagonalGaussianLikelihood``,
``MultiObservationLikelihood``, or ``CSPBasis``.
"""

from __future__ import annotations

from typing import Union

import jax
import jax.numpy as jnp

Array = jax.Array


# ---------------------------------------------------------------------------
# Helper: convert any index-like object to a hashable Python tuple of ints
# ---------------------------------------------------------------------------

def _to_int_tuple(idx) -> tuple[int, ...]:
    """
    Convert a JAX/numpy index array, Python list, or int to a tuple of ints.

    This is the canonical form stored in ``ThetaVector._name_to_idx``.
    Tuples of ints are hashable and therefore safe as JAX PyTree auxiliary data.
    """
    try:
        # JAX / numpy arrays
        return tuple(int(i) for i in idx)
    except TypeError:
        # scalar int
        return (int(idx),)


# ===========================================================================
# ThetaVector
# ===========================================================================

class ThetaVector:
    """
    Flat 1-D JAX parameter array with both integer-index and string-key access.

    Parameters
    ----------
    data : Array, shape (n_params,)
        The flat parameter vector.  This is the only JAX-traced value;
        everything else is static Python.
    name_to_idx : dict[str, array-like]
        Maps each parameter name to its index or index array within ``data``.
        Values may be JAX arrays, numpy arrays, Python lists, or ints.
        They are converted to Python tuples of ints at construction time.

    Examples
    --------
    >>> tv = ThetaVector(
    ...     jnp.array([0.1, 0.2, 0.3, -4.6]),
    ...     {"sfh": jnp.arange(3), "log_jitter": jnp.array([3])},
    ... )
    >>> tv["sfh"]          # Array([0.1, 0.2, 0.3])
    >>> tv["log_jitter"]   # Array([-4.6])
    >>> tv[jnp.array([0, 2])]  # Array([0.1, 0.3])
    """

    __slots__ = ("_data", "_name_to_idx")

    def __init__(
        self,
        data: Array,
        name_to_idx: dict[str, object],
    ) -> None:
        object.__setattr__(self, "_data", data)
        # Convert all index arrays → plain Python tuples of ints (hashable).
        object.__setattr__(
            self,
            "_name_to_idx",
            {k: _to_int_tuple(v) for k, v in name_to_idx.items()},
        )

    # ------------------------------------------------------------------
    # Core access interface
    # ------------------------------------------------------------------

    def __getitem__(self, key: Union[str, Array]) -> Array:
        """
        Access parameters by name (str) or by integer / array index.

        Parameters
        ----------
        key : str or array-like
            If ``str``: looks up the pre-registered index tuple for that
            parameter name, then slices ``self._data``.
            Otherwise: passes ``key`` directly to ``self._data[key]``, which
            supports all JAX indexing modes (scalar int, slice, array).

        Returns
        -------
        Array
            The requested parameter value(s).

        Raises
        ------
        KeyError
            If ``key`` is a string not present in the name-to-index mapping.
        """
        if isinstance(key, str):
            idx = self._name_to_idx[key]  # Python tuple of ints — static
            return self._data[jnp.array(idx)]
        # Integer, slice, or JAX array index — pass through directly.
        return self._data[key]

    # ------------------------------------------------------------------
    # Shape / dtype passthrough (makes ThetaVector behave like an array
    # for code that inspects these attributes without indexing)
    # ------------------------------------------------------------------

    @property
    def shape(self) -> tuple[int, ...]:
        return self._data.shape

    @property
    def dtype(self):
        return self._data.dtype

    def __len__(self) -> int:
        return len(self._data)

    # ------------------------------------------------------------------
    # Attribute setting is disabled (treat as immutable like a frozen
    # dataclass — modifications must produce new ThetaVector instances)
    # ------------------------------------------------------------------

    def __setattr__(self, name: str, value) -> None:
        raise AttributeError(
            "ThetaVector is immutable.  "
            "To update theta, create a new ThetaVector with the new data array."
        )

    # ------------------------------------------------------------------
    # Convenience: named parameter names
    # ------------------------------------------------------------------

    @property
    def param_names(self) -> tuple[str, ...]:
        """All registered parameter names."""
        return tuple(self._name_to_idx.keys())

    def has(self, name: str) -> bool:
        """Return True if *name* is a registered parameter."""
        return name in self._name_to_idx

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        names_summary = ", ".join(
            f"{k}[{len(v)}]" for k, v in self._name_to_idx.items()
        )
        return (
            f"ThetaVector(shape={self.shape}, dtype={self.dtype}, "
            f"params=[{names_summary}])"
        )


# ---------------------------------------------------------------------------
# JAX PyTree registration
#
# Leaf:      self._data  (the only JAX-traced array)
# Auxiliary: a sorted tuple of (name, idx_tuple) pairs — static, hashable
#
# JAX uses auxiliary data as a cache key for JIT.  Using a sorted tuple of
# pairs ensures deterministic ordering and correct hash equality.
# ---------------------------------------------------------------------------

jax.tree_util.register_pytree_node(
    ThetaVector,
    flatten_func=lambda tv: (
        [tv._data],
        tuple(sorted(tv._name_to_idx.items())),   # static, hashable aux
    ),
    unflatten_func=lambda aux, leaves: ThetaVector(
        data=leaves[0],
        name_to_idx=dict(aux),                     # tuples are already ints
    ),
)


# ---------------------------------------------------------------------------
# CSPBasis integration helper
#
# Call this from user code after constructing a CSPBasis:
#
#     tv_init = make_theta_vector_from_csp(csp)
#
# This produces a ThetaVector wrapping csp.theta_init with all parameter
# names registered.  Pass tv_init as the starting point for the sampler.
# ---------------------------------------------------------------------------

def make_theta_vector_from_csp(csp) -> ThetaVector:
    """
    Construct a ``ThetaVector`` from a ``CSPBasis`` instance.

    Reads the ``*_idx`` attributes that ``CSPBasis.initialize_model_structure``
    registers on the model and builds the complete name-to-index mapping
    automatically.

    Parameters
    ----------
    csp : CSPBasis
        An initialised ``CSPBasis`` object.

    Returns
    -------
    ThetaVector
        Wraps ``csp.theta_init`` with all parameter names registered.

    Examples
    --------
    >>> csp = CSPBasis(sspdata, theta_dict)
    >>> tv = make_theta_vector_from_csp(csp)
    >>> tv["sfh"]               # the SFH slice of theta_init
    >>> tv[csp.sfh_idx]         # same slice, integer-array access
    >>> tv["log_jitter"]        # noise nuisance, if registered in theta_dict
    """
    name_to_idx: dict[str, object] = {}
    for attr in dir(csp):
        if attr.endswith("_idx") and not attr.startswith("__"):
            # Attribute names are of the form "<param_name>_idx"
            param_name = attr[:-4]  # strip "_idx"
            name_to_idx[param_name] = getattr(csp, attr)

    return ThetaVector(data=csp.theta_init, name_to_idx=name_to_idx)
