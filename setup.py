"""
Driver Drowsiness Detection System - Setup Script

Install with:
    pip install -e .

Or for development:
    pip install -e ".[dev]"
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="drowsiness-detection",
    version="2.0.0",
    author="Ahmed",
    description="Real-time driver drowsiness detection using computer vision",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Ahmed-Hereworking/Driver-Drowsiness-Detection",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "opencv-python>=4.5.0",
        "dlib>=19.22.0",
        "PyYAML>=6.0",
        "pygame>=2.1.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "ruff>=0.1.0",
            "mypy>=1.0.0",
        ],
        "web": [
            "streamlit>=1.20.0",
            "plotly>=5.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "drowsiness-detect=main:main",
        ],
    },
)
