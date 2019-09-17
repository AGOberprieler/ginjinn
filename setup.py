#!/usr/bin/env python

from setuptools import setup, find_packages
import re

# get version from init file
with open('ginjinn/__init__.py', 'r') as f:
    VERSION=re.search(
        r"^__version__ = ['\"]([^'\"]*)['\"]",
        f.read(),
        re.M
    ).group(1)

DESCRIPTION='Object detection pipeline for the extraction of structures from herbarium specimens'

def install_requires():
    '''Get requirements from requirements.txt'''
    # with open('requirements.txt') as f:
    #     return f.read().splitlines()
    return []

setup(
    name='ginjinn',
    version=VERSION,
    url='https://github.com/AGOberprieler/ginjinn',
    author='Tankred Ott',
    author_email='tankred.ott@ur.de',
    description=DESCRIPTION,
    packages=find_packages(),
    install_requires=install_requires(),
    entry_points={
        'console_scripts': [
            'ginjinn = ginjinn.__main__:main',
        ]
    },
    package_data={
        'ginjinn': [
            'data_files/*.yaml',
            'data_files/tf_config_templates/*.config',
            'data_files/*',
        ],
    }
)