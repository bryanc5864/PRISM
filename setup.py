from setuptools import setup, find_packages

setup(
    name="prism-scrnaseq",
    version="1.0.0",
    description="PRISM: Progenitor Resolution via Invariance-Sensitive Modeling",
    author="Bryan Cheng",
    packages=find_packages(),
    package_data={
        "prism": ["../configs/*.yaml"],
    },
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0",
        "scanpy>=1.9",
        "anndata>=0.9",
        "numpy>=1.24",
        "scipy>=1.10",
        "scikit-learn>=1.2",
        "numpyro>=0.13",
        "jax>=0.4",
        "matplotlib>=3.7",
        "seaborn>=0.12",
        "umap-learn>=0.5",
        "pyyaml>=6.0",
        "tqdm>=4.65",
        "einops>=0.6",
        "pandas>=1.5",
        "harmonypy>=0.0.9",
    ],
    entry_points={
        "console_scripts": [
            "prism-run=prism.cli:main",
        ],
    },
)
