# Copyright (C) 2021 by the ufl4rom authors
#
# This file is part of ufl4rom.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from setuptools import find_packages, setup

setup(name="ufl4rom",
      description="A collection of UFL utilities for reduced order modelling codes",
      long_description="A collection of UFL utilities for reduced order modelling codes",
      author="Francesco Ballarin (and contributors)",
      author_email="francesco.ballarin@unicatt.it",
      version="0.0.dev1",
      license="GNU Library or Lesser General Public License (LGPL)",
      url="https://github.com/RBniCS/ufl4rom",
      classifiers=[
          "Development Status :: 3 - Alpha",
          "Intended Audience :: Developers",
          "Intended Audience :: Science/Research",
          "Programming Language :: Python :: 3",
          "Programming Language :: Python :: 3.6",
          "Programming Language :: Python :: 3.7",
          "Programming Language :: Python :: 3.8",
          "Programming Language :: Python :: 3.9",
          "License :: OSI Approved :: GNU Library or Lesser General Public License (LGPL)",
          "Topic :: Scientific/Engineering :: Mathematics",
          "Topic :: Software Development :: Libraries :: Python Modules",
      ],
      packages=find_packages(),
      include_package_data=True,
      install_requires=["fenics-ufl"],
      tests_require=["pytest", "pytest-flake8"]
      )
