from setuptools import setup, find_packages

setup(
    name="fluorescence-bpm",
    version="0.1.0",
    author="Your Name",
    description="Fluorescence microscopy simulation via beam propagation method",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "torch",
        "scikit-image",
        "tifffile",
        "tqdm",
    ],
)
