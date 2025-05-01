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
        'jax',
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