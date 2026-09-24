"""Guard against importing this repository folder as the package ``ceridwen``.

This file is never imported when ceridwen is used normally: the package is the ``ceridwen/``
folder *inside* this repository.  It runs only when Python is started from the folder that
*contains* a clone named ``ceridwen``: the clone then sits first on ``sys.path`` and, without
this file, Python would import it as an empty namespace package (``ceridwen.__version__``
raises ``AttributeError``, ``ceridwen.ssps`` is "not found").

So here: import the ceridwen that is installed (``pip install ceridwen`` or
``pip install -e <clone>``), exactly as if this folder were not in the way; if none is
installed, raise one clear error.  Not part of the wheel.
"""
import importlib.machinery as _machinery
import importlib.util as _util
import os as _os
import sys as _sys

_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PARENT = _os.path.dirname(_HERE)


def _installed_spec():
    """The spec Python would find for ``ceridwen`` if this folder were not on sys.path."""
    path = [p for p in _sys.path if _os.path.abspath(p or _os.curdir) != _PARENT]
    for finder in _sys.meta_path:
        find = getattr(finder, "find_spec", None)
        if find is None:
            continue
        try:
            spec = (_machinery.PathFinder.find_spec(__name__, path)
                    if finder is _machinery.PathFinder else find(__name__, None))
        except Exception:                                   # a foreign finder: not ours to judge
            continue
        if spec is None or spec.origin in (None, "namespace"):
            continue
        if _os.path.dirname(_os.path.abspath(spec.origin)) == _HERE:
            continue                                        # this file
        return spec
    return None


_spec = _installed_spec() if __name__ == "ceridwen" else None
if __name__ != "ceridwen":
    pass                                   # the clone folder under another name: stay inert
elif _spec is None:
    raise ImportError(
        f"'import {__name__}' found the folder {_HERE}, which is a clone of the ceridwen "
        f"repository, not the package: {_PARENT}, the folder that contains the clone, is on "
        f"sys.path (it is the current folder, or the folder of the script you ran), and no "
        f"installed ceridwen was found.  Install it "
        f"(pip install ceridwen, or pip install -e {_HERE}), or run from inside the clone "
        f"or from any other folder.")
else:
    # A module may replace itself in sys.modules while it is being imported; the import
    # system then returns the replacement.  Load the installed package under the same name.
    _real = _util.module_from_spec(_spec)
    _sys.modules[__name__] = _real
    _spec.loader.exec_module(_real)
