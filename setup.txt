#!/usr/bin/env python3
"""
Kaleidoscope AI - Software Ingestion & Mimicry System
Package setup script
"""

from setuptools import setup, find_packages
import os

# Read requirements
with open("requirements.txt", "r") as f:
    requirements = [line.strip() for line in f.readlines() if not line.startswith('#')]

# Read long description from README
with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="kaleidoscope-ai",
    version="0.1.0",
    description="Software Ingestion & Mimicry System powered by LLMs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Kaleidoscope AI Team",
    author_email="info@kaleidoscope.ai",
    url="https://github.com/kaleidoscope-ai/kaleidoscope",
    packages=find_packages(exclude=["tests", "tests.*"]),
    entry_points={
        "console_scripts": [
            "kaleidoscope=kaleidoscope:main",
        ],
    },
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=requirements,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    package_data={
        "kaleidoscope": [
            "templates/*",
            "static/css/*",
            "static/js/*",
            "static/img/*",
        ],
    },
)
