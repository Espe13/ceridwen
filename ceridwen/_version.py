# Single source of truth for the version is pyproject.toml ([project].version).
# This module is kept only because ceridwen/__init__.py imports __version__ from
# it as a fallback; keep the two in sync, or switch to
# importlib.metadata.version("ceridwen") to derive it automatically.
__version__ = version = "0.1.1"
__version_tuple__ = version_tuple = (0, 1, 1)
