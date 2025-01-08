# Copyright (C) 2021-2025 by the ufl4rom authors
#
# This file is part of ufl4rom.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Import specialization of UFL classes from dolfinx and firedrake backends."""

import sys
import typing

import ufl

if sys.version_info >= (3, 11):
    import typing as typing_extensions
else:  # pragma: no cover
    import typing_extensions

try:
    import dolfinx
except ImportError:
    DolfinxScalarType = float

    class DolfinxConstant(ufl.Constant):  # type: ignore[misc, no-any-unimported]
        """Mock dolfinx.fem.Constant class."""

        def __init__(  # type: ignore[no-any-unimported]
            self, domain: ufl.AbstractDomain,
            value: typing.Union[DolfinxScalarType, typing.Iterable[DolfinxScalarType]]
        ) -> None:  # pragma: no cover
            raise RuntimeError("Cannot use a dolfinx constant when dolfinx is not installed")

    class DolfinxFunction(ufl.Coefficient):  # type: ignore[misc, no-any-unimported]
        """Mock dolfinx.fem.Function class."""

        @property
        def name(self) -> str:  # pragma: no cover
            """Get function name."""
            raise RuntimeError("Cannot use a dolfinx function when dolfinx is not installed")
else:
    import dolfinx.fem
    import petsc4py.PETSc

    DolfinxConstant = dolfinx.fem.Constant  # type: ignore
    DolfinxFunction = dolfinx.fem.Function  # type: ignore
    DolfinxScalarType = petsc4py.PETSc.ScalarType  # type: ignore

try:
    import firedrake
except ImportError:
    FiredrakeScalarType = float

    class FiredrakeConstant(ufl.constantvalue.ConstantValue):  # type: ignore[misc, no-any-unimported]
        """Mock firedrake.Constant class."""

        def __new__(  # type: ignore[no-any-unimported]
            cls: type[typing_extensions.Self],
            value: typing.Union[FiredrakeScalarType, typing.Iterable[FiredrakeScalarType]],
            domain: typing.Optional[ufl.AbstractDomain] = None, name: typing.Optional[str] = None
        ) -> typing_extensions.Self:  # pragma: no cover
            """Create a new constant."""
            raise RuntimeError("Cannot use a firedrake constant when firedrake is not installed")

        def __init__(  # type: ignore[no-any-unimported]
            self, value: typing.Union[FiredrakeScalarType, typing.Iterable[FiredrakeScalarType]],
            domain: typing.Optional[ufl.AbstractDomain] = None, name: typing.Optional[str] = None
        ) -> None:  # pragma: no cover
            raise RuntimeError("Cannot use a firedrake constant when firedrake is not installed")

    class FiredrakeFunction(ufl.Coefficient):  # type: ignore[misc, no-any-unimported]
        """Mock firedrake.Function class."""

        def name(self) -> str:  # pragma: no cover
            """Get function name."""
            raise RuntimeError("Cannot use a firedrake function when firedrake is not installed")
else:
    import petsc4py.PETSc

    FiredrakeConstant = firedrake.Constant  # type: ignore
    FiredrakeFunction = firedrake.Function  # type: ignore
    FiredrakeScalarType = petsc4py.PETSc.ScalarType  # type: ignore
