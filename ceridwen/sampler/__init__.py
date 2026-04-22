# ceridwen/sampler/__init__.py

from .priors import (
    Prior,
    Uniform,
    TopHat,
    Normal,
    #MultiVariateNormal,
    ClippedNormal,
    LogNormal,
    StudentT,
)
from .runner import SamplerAdapter, SamplingResult, run_sampler
from .nested import BlackJAXNestedSamplerAdapter
from .nuts   import BlackJAXNUTSAdapter
from .vi     import (
    VariationalMap, TriLMap, IAFMap, TrainedMap, train_vi, make_vi_map,
)

__all__ = [
    # Priors
    "Prior",
    "Uniform",
    "TopHat",
    "Normal",
    #"MultiVariateNormal",
    "ClippedNormal",
    "LogNormal",
    "StudentT",
    # Sampler protocol
    "SamplerAdapter",
    "SamplingResult",
    "run_sampler",
    # Backends
    "BlackJAXNestedSamplerAdapter",
    "BlackJAXNUTSAdapter",
    # Variational-inference preconditioning (NeuTra HMC)
    "VariationalMap",
    "TriLMap",
    "IAFMap",
    "TrainedMap",
    "train_vi",
    "make_vi_map",
]
