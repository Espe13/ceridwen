# All package metadata now lives in pyproject.toml ([project] table).
# This shim exists only so that legacy tooling / `pip install -e .` on older
# pip versions still works; setuptools reads the metadata from pyproject.toml.
from setuptools import setup

setup()
