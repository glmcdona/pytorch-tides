import sys
import os
from setuptools import setup, find_packages

# Note to self to build and upload skip existing:
#   python setup.py sdist bdist_wheel
#   twine upload dist/* --skip-existing
with open("readme.md", "r") as fh:
    long_description = fh.read()

setup(
    name='pytorch-tides',
    version='0.0.1',
    author='Geoff McDonald',
    author_email='glmcdona@gmail.com',
    packages=find_packages(exclude=['tests*']),
    url='https://pypi.python.org/pypi/pytorch-tides/',
    license='MIT',
    description='Global tide chart prediction model using PyTorch.',
    long_description=open('README.md').read(),
    install_requires=[
        "pytest",
        "numpy",
        "scipy",
        "pandas",
        "matplotlib",
        "torch",
        "sklearn"
    ],
    test_suite='nose2.collector.collector',
    tests_require=['nose2'],
)
