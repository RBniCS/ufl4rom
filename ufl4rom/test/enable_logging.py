# Copyright (C) 2021-2024 by the ufl4rom authors
#
# This file is part of ufl4rom.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Test utility to enable logging."""

import functools
import logging
import sys
import typing

OriginalTestType = typing.Callable[..., typing.Any]


def enable_logging(
    loggers_and_levels: typing.Dict[logging.Logger, int]
) -> typing.Callable[[OriginalTestType], OriginalTestType]:
    """Return a decorator that enables logging."""
    logging.basicConfig(stream=sys.stdout)

    def enable_logging_decorator(original_test: OriginalTestType) -> OriginalTestType:
        """Implement a decorator that enables logging."""
        @functools.wraps(original_test)
        def decorated_test(*args: typing.Any, **kwargs: typing.Any) -> None:  # noqa: ANN401
            """Set logging levels before running the test, and unset the afterwards."""
            for (logger, level) in loggers_and_levels.items():
                logger.setLevel(level)
            original_test(*args, **kwargs)
            for logger in loggers_and_levels.keys():
                logger.setLevel(logging.NOTSET)
        return decorated_test

    return enable_logging_decorator
