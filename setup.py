"""Setup module for GazeML."""

from setuptools import setup, find_packages
# from pipenv import find_install_requires

setup(
        name='gazeml',
        version='0.1',
        description='Data-driven gaze estimation using machine learning.',

        author='Seonwook Park',
        author_email='spark@inf.ethz.ch',

        packages=find_packages(exclude=[]),
        python_requires='>=3.5',
        install_requires=[],
)
