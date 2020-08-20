# -*- coding: utf-8 -*-

# Learn more: https://github.com/kennethreitz/setup.py

from setuptools import setup, find_packages


# def read_requirements():
#     """parse requirements from requirements.txt."""
#     reqs_path = os.path.join("", "requirements.txt")
#     with open(reqs_path, "r") as f:
#         requirements = [line.rstrip() for line in f]
#     return requirements


with open("README.md", encoding="utf-8") as f:
    readme = f.read()

with open("LICENSE", encoding="utf-8") as f:
    license = f.read()

setup(
    name="medner_j",
    version="0.1.0",
    description="MedNER-J: Japanese Disease Extractor based on BERT+CRF.",
    long_description=readme,
    author="Shogo Ujiie",
    author_email="ujiie@is.naist.jp",
    url="https://github.com/sociocom/MedNER-J/tree/package",
    license=license,
    packages=find_packages(exclude=("tests", "docs")),
    install_requires=[
        "mecab-python3>=0.996.5",
        "torch==1.5.1",
        "transformers==2.11.0",
        "allennlp==1.0.0",
        "dnorm_j @ git+https://github.com/sociocom/DNorm-J.git"
    ],
    entry_points={
        "console_scripts": [
            "mednerj = medner_j.__main__:main"
            ]
        }
)
