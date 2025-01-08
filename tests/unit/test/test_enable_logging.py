# Copyright (C) 2021-2025 by the ufl4rom authors
#
# This file is part of ufl4rom.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Unit tests for ufl4rom.test.enable_logging module."""

import logging

import ufl4rom.test

test_logger = logging.getLogger("tests/unit/test/test_enable_logging.py")
enable_logger = ufl4rom.test.enable_logging({test_logger: logging.DEBUG})


@enable_logger
def test_enable_logging() -> None:
    """Add a few debug lines to a logger."""
    test_logger.log(logging.DEBUG, "Hello, logger!")
