from setuptools import setup, find_packages

setup(
    name="fid_susc_alg",
    version="0.1.0",
    description="Fidelity Susceptibility and QFI computation with QSVT simulation.",
    author="Luca Manzi",
    packages=find_packages(),
    install_requires=["numpy", "scipy", "matplotlib"],
    python_requires=">=3.8",
)
