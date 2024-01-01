# Copyright (C) 2021-2024 by the ufl4rom authors
#
# This file is part of ufl4rom.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Add a name to ufl.Coefficient."""

import re
import typing

import ufl
import ufl.functionspace


class NamedCoefficient(ufl.Coefficient):  # type: ignore[misc, no-any-unimported]
    """An ufl.Coefficient with an additional name attribute."""

    def __init__(  # type: ignore[no-any-unimported]
        self, name: str, function_space: ufl.functionspace.AbstractFunctionSpace, count: typing.Optional[int] = None
    ) -> None:
        super().__init__(function_space, count)
        self._name = name

        # Neglect the count argument when preparing the representation string, as we aim to
        # get a representation which is independent on the internal counter
        self._repr = "NamedCoefficient({}, {})".format(
            repr(self._name), repr(self._ufl_function_space))
        self._repr = re.sub(" +", " ", self._repr)
        self._repr = re.sub(r"\[ ", "[", self._repr)
        self._repr = re.sub(r" \]", "]", self._repr)

    def __str__(self) -> str:
        """Represent the coefficient by its name."""
        return self._name
