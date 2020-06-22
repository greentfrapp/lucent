import lucent
from setuptools import setup, find_packages

version = lucent.__version__

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="torch-lucent",
    packages=find_packages(exclude=[]),
    version=version,
    description=(
        "Lucid for PyTorch. "
        "Collection of infrastructure and tools for research in "
        "neural network interpretability."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="The Lucent Authors",
    author_email="limsweekiat@gmail.com",
    url="https://github.com/greentfrapp/lucent",
    license="Apache License 2.0",
    keywords=[
        "pytorch",
        "tensor",
        "machine learning",
        "neural networks",
        "convolutional neural networks",
        "feature visualization",
        "optimization",
    ],
    install_requires=[
        "torch>=1.5.0",
        "torchvision",
        "kornia",
        "tqdm",
        "numpy",
        "ipython",
        "pillow",
        "future",
        "decorator",
        "pytest",
        "pytest-mock",
        "coverage",
        "coveralls",
        "scikit-learn"
    ],
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
