from setuptools import setup, find_packages

# Read the README file for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="curve-algebra",
    version="0.1.0",
    author="Your Name",
    description="A functional, composable Python library for mathematical curves with a focus on financial calculations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/curve-algebra",
    py_modules=["curve", "finance", "visualize"],
    python_requires=">=3.7",
    install_requires=[
        "matplotlib>=3.0.0",
        "numpy>=1.18.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Office/Business :: Financial",
    ],
)
