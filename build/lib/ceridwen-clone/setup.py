from setuptools import setup, find_packages

setup(
    name='ceridwen',
    version='0.1.0',
    description="SED fitting with HMC",
    author='Amanda Stoffers',
    author_email="aas208@cam.ac.uk",
    packages=find_packages(where="."), 
    include_package_data=True,
    install_requires=[
        # JAX >= 0.4.30 required: the code uses jnp.trapezoid, the modern
        # jax.tree_util namespace, and post-2024 PRNG conventions.
        'jax>=0.4.30',
        'matplotlib',
        'numpy',
        'scipy',
    ],
    python_requires=">=3.10",
        classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)