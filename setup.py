import setuptools

from giga_cherche.__version__ import __version__

with open(file="README.md", mode="r", encoding="utf-8") as fh:
    long_description = fh.read()

base_packages = [
    "sentence-transformers >= 3.0.1",
    "datasets >= 2.20.0",
    "accelerate >= 0.31.0",
    "weaviate-client >= 4.6.4",
]

dev = ["ruff >= 0.4.9"]

eval = ["ranx >= 0.3.16", "beir >= 2.0.0"]

setuptools.setup(
    name="giga_cherche",
    version=f"{__version__}",
    license="",
    author="LightON",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lightonai/giga-cherche",
    keywords=[],
    packages=setuptools.find_packages(),
    install_requires=base_packages,
    extras_require={"dev": base_packages + dev + eval, "eval": base_packages + eval},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
