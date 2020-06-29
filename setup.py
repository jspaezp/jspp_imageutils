#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['Click>=7.0', ]

setup_requirements = ['pytest-runner', 'Cython', 'numpy', ]

test_requirements = ['pytest>=3', 'Cython', 'numpy', ]

setup(
    author="J. Sebastian Paez",
    author_email='jpaezpae@purdue.edu',
    python_requires='>=3.5',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="A small python package that bundles several utilities to handle images, specially for the purpose of preparing microscopy data for deep learning",
    entry_points={
        'console_scripts': [
            'jspp_imageutils=jspp_imageutils.cli:main',
        ],
    },
    install_requires=requirements,
    license="Apache Software License 2.0",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='jspp_imageutils',
    name='jspp_imageutils',
    packages=find_packages(include=['jspp_imageutils', 'jspp_imageutils.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/jspaezp/jspp_imageutils',
    version='0.2.0',
    zip_safe=False,
)
