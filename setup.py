import setuptools

from pylate.__version__ import __version__

with open(file="README.md", mode="r", encoding="utf-8") as fh:
    long_description = fh.read()

base_packages = [
    "sentence-transformers >= 3.0.1",
    "datasets >= 2.20.0",
    "accelerate >= 0.31.0",
    "voyager >= 2.0.9",
    "sqlitedict >= 2.1.0",
    "pandas >= 2.2.1",
]


dev = [
    "ruff >= 0.4.9",
    "pytest-cov >= 5.0.0",
    "pytest >= 8.2.1",
    "pandas >= 2.2.1",
    "mkdocs-material == 9.5.32",
    "mkdocs-awesome-pages-plugin == 2.9.3",
    "mkdocs-jupyter == 0.24.8",
    "mkdocs_charts_plugin == 0.0.10",
    "numpydoc == 1.8.0",
]

eval = ["ranx >= 0.3.16", "beir >= 2.0.0"]

api = ["fastapi >= 0.114.1", "uvicorn >= 0.30.6", "batched >= 0.1.2"]

setuptools.setup(
    name="pylate",
    version=f"{__version__}",
    license="",
    author="LightON",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lightonai/pylate",
    keywords=[],
    packages=setuptools.find_packages(),
    install_requires=base_packages,
    extras_require={
        "eval": base_packages + eval,
        "api": base_packages + api,
        "dev": base_packages + dev + eval + api,
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
