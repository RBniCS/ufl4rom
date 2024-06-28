# Copyright (C) 2021-2024 by the ufl4rom authors
#
# This file is part of ufl4rom.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Add a name to ufl.Constant and its specialization offered by backends."""

import re
import sys
import typing

import ufl

from ufl4rom.utils.backends import DolfinxConstant, DolfinxScalarType, FiredrakeConstant, FiredrakeScalarType

if sys.version_info >= (3, 11):
    import typing as typing_extensions
else:  # pragma: no cover
    import typing_extensions


class NamedConstant(ufl.Constant):  # type: ignore[misc, no-any-unimported]
    """An ufl.Constant with an additional name attribute."""

    def __init__(  # type: ignore[no-any-unimported]
        self, name: str, domain: ufl.AbstractDomain, shape: tuple[int, ...] = (),
        count: typing.Optional[int] = None
    ) -> None:
        super().__init__(domain, shape, count)
        self._name = name

        # Neglect the count argument when preparing the representation string, as we aim to
        # get a representation which is independent on the internal counter
        self._repr = f"NamedConstant({self._name!r}, {self._ufl_domain!r}, {self._ufl_shape!r})"
        self._repr = re.sub(" +", " ", self._repr)
        self._repr = re.sub(r"\[ ", "[", self._repr)
        self._repr = re.sub(r" \]", "]", self._repr)

    def __str__(self) -> str:  # pragma: no cover
        """Return a string representation which is independent on internal counters."""
        return self._name


class NamedConstantValue(ufl.constantvalue.ConstantValue):  # type: ignore[misc, no-any-unimported]
    """An ufl.constantvalue.ConstantValue with an additional name attribute."""

    def __init__(
        self, name: str, shape: tuple[int, ...] = ()
    ) -> None:
        super().__init__()
        self._name = name
        assert not hasattr(self, "_ufl_shape")
        self._ufl_shape = shape

        # Represent the constant value by its name and its shape
        self._repr = f"NamedConstantValue({self._name!r}, {self._ufl_shape!r})"
        self._repr = re.sub(" +", " ", self._repr)
        self._repr = re.sub(r"\[ ", "[", self._repr)
        self._repr = re.sub(r" \]", "]", self._repr)

    def __str__(self) -> str:  # pragma: no cover
        """Return the name of the constant value as a string representation."""
        return self._name

    def __repr__(self) -> str:  # pragma: no cover
        """Return string representation this object can be reconstructed from."""
        return self._repr

    @property
    def ufl_shape(self) -> tuple[int, ...]:  # pragma: no cover
        """Shape of the constant value."""
        return self._ufl_shape


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
        self._repr = f"DolfinxNamedConstant({self._name!r}, {self.value!r}, {self._ufl_domain!r})"
        self._repr = re.sub(" +", " ", self._repr)
        self._repr = re.sub(r"\[ ", "[", self._repr)
        self._repr = re.sub(r" \]", "]", self._repr)

    def __str__(self) -> str:
        """Represent the constant by its name."""
        return self._name


class FiredrakeNamedConstant(FiredrakeConstant):
    """A firedrake.Constant with an additional name attribute."""

    def __new__(  # type: ignore[no-any-unimported]
        cls: type[typing_extensions.Self], name: str,
        value: typing.Union[FiredrakeScalarType, typing.Iterable[FiredrakeScalarType]],
        domain: typing.Optional[ufl.AbstractDomain] = None
    ) -> typing_extensions.Self:
        """Create a new constant."""
        return typing.cast(FiredrakeNamedConstant, FiredrakeConstant.__new__(cls, value, domain))

    def __init__(  # type: ignore[no-any-unimported]
        self, name: str, value: typing.Union[FiredrakeScalarType, typing.Iterable[FiredrakeScalarType]],
        domain: typing.Optional[ufl.AbstractDomain] = None
    ) -> None:
        super().__init__(value, domain, name)
        assert domain is None, "Giving Constants a domain has been deprecated in firedrake"
        self._name = name

        # Neglect the count argument when preparing the representation string, as we aim to
        # get a representation which is independent on the internal counter
        self._repr = f"FiredrakeNamedConstant({self._name!r}, {self.values()!r})"
        self._repr = re.sub(" +", " ", self._repr)
        self._repr = re.sub(r"\[ ", "[", self._repr)
        self._repr = re.sub(r" \]", "]", self._repr)

    def __str__(self) -> str:
        """Represent the constant by its name."""
        return self._name
