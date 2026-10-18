"""Setup script for Multimodal Protein Function Prediction."""
from setuptools import setup, find_packages

setup(
    name="protein-function-prediction",
    version="1.0.0",
    description="Multimodal deep learning for protein function prediction",
    author="MSc Bioinformatics Research",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torch-geometric>=2.4.0",
        "transformers>=4.30.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "networkx>=3.1",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "plotly>=5.15.0",
        "streamlit>=1.28.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "requests>=2.31.0",
        "biopython>=1.81",
    ],
    extras_require={
        "dev": ["black", "isort", "mypy", "pytest"],
        "xai": ["shap>=0.42.0", "captum>=0.6.0"],
    },
)
