from setuptools import setup, find_packages

setup(
    name="beam",
    version="1.0.0",
    description="Background Elimination via Additive Modeling",
    author="Daniel Muthukrishna, Drew Tufto"
    author_email="daniel.muthukrishna@gmail.com, ajtufto@gmail.com"
    url="https://github.com/daniel-muthukrishna/beam",
    packages=find_packages(include=["beam", "beam.*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.3",
        "numpy>=1.19.0",
        "matplotlib>=3.3.0",
        "tqdm>=4.50.0",
        "astropy>=4.0",
        "Pillow>=8.0.0",
        "pyyaml>=5.4.0",
        "einops",
        "torchdiffeq",
        "wandb",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.10.0",
            "black>=21.5b2",
            "isort>=5.8.0",
            "mypy>=0.812",
            "flake8>=3.9.0",
        ],
    },
)
