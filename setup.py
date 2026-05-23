from setuptools import setup, find_packages

setup(
    name="amino_lattice",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "rdkit",
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "matplotlib",
        "tqdm",
    ],
    python_requires=">=3.9",
)
