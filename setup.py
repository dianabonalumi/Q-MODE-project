from setuptools import setup, find_packages

setup(
    name="qmode",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "rdkit",
        "numpy",
        "pandas",
        "matplotlib",
        "biopython",
        "qiskit",
        "qiskit-aer",
    ],
    python_requires=">=3.9",
)
