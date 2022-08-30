# Copyright (C) 2021-2022 by the ufl4rom authors
#
# This file is part of ufl4rom.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Add a name to ufl.Constant and its specialization offered by backends."""

import re
import typing

import ufl

from ufl4rom.utils.backends import (
    DolfinConstant, DolfinScalarType, DolfinxConstant, DolfinxScalarType, FiredrakeConstant, FiredrakeScalarType)


class NamedConstant(ufl.Constant):  # type: ignore[misc, no-any-unimported]
    """An ufl.Constant with an additional name attribute."""

    def __init__(  # type: ignore[no-any-unimported]
        self, name: str, domain: ufl.AbstractDomain, shape: typing.Tuple[int, ...] = (),
        count: typing.Optional[int] = None
    ) -> None:
        super().__init__(domain, shape, count)
        self._name = name

        # Neglect the count argument when preparing the representation string, as we aim to
        # get a representation which is independent on the internal counter
        self._repr = "NamedConstant({}, {}, {})".format(
            repr(self._name), repr(self._ufl_domain), repr(self._ufl_shape))
        self._repr = re.sub(" +", " ", self._repr)
        self._repr = re.sub(r"\[ ", "[", self._repr)
        self._repr = re.sub(r" \]", "]", self._repr)

    def __str__(self) -> str:  # pragma: no cover
        """Return a string representation which is independent on internal counters."""
        return self._name


class DolfinNamedConstant(DolfinConstant):
    """A dolfin.Constant with constructor arguments in a slighlty different order."""

    def __init__(  # type: ignore[no-any-unimported]
        self, name: str, value: typing.Union[DolfinScalarType, typing.Iterable[DolfinScalarType]],
        cell: typing.Optional[ufl.Cell] = None
    ) -> None:
        super().__init__(value, cell, name)


class DolfinxNamedConstant(DolfinxConstant):
    """A dolfinx.Constant with an additional name attribute."""

    def __init__(  # type: ignore[no-any-unimported]
        self, name: str, value: typing.Union[DolfinxScalarType, typing.Iterable[DolfinxScalarType]],
        domain: ufl.AbstractDomain
    ) -> None:
        super().__init__(domain, value)
        self._name = name

        # Neglect the count argument when preparing the representation string, as we aim to
        # get a representation which is independent on the internal counter
        self._repr = "DolfinxNamedConstant({}, {}, {})".format(
            repr(self._name), repr(self.value), repr(self._ufl_domain))
        self._repr = re.sub(" +", " ", self._repr)
        self._repr = re.sub(r"\[ ", "[", self._repr)
        self._repr = re.sub(r" \]", "]", self._repr)

    def __str__(self) -> str:
        """Represent the constant by its name."""
        return self._name


class FiredrakeNamedConstant(FiredrakeConstant):
    """A firedrake.Constant with an additional name attribute."""

    def __init__(  # type: ignore[no-any-unimported]
        self, name: str, value: typing.Union[FiredrakeScalarType, typing.Iterable[FiredrakeScalarType]],
        domain: typing.Optional[ufl.AbstractDomain] = None
    ) -> None:
        super().__init__(value, domain)
        self._name = name

        # Neglect the count argument when preparing the representation string, as we aim to
        # get a representation which is independent on the internal counter
        self._repr = "FiredrakeNamedConstant({}, {}, {})".format(
            repr(self._name), repr(self.values()), repr(self._ufl_function_space._ufl_domain))
        self._repr = re.sub(" +", " ", self._repr)
        self._repr = re.sub(r"\[ ", "[", self._repr)
        self._repr = re.sub(r" \]", "]", self._repr)

    def __str__(self) -> str:
        """Represent the constant by its name."""
        return self._name
