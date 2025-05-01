from .dust import DustModel, DustEmission

try:
    from ._version import __version__, __githash__
except(ImportError):
    pass
