# Copyright (C) 2021-2022 by the ufl4rom authors
#
# This file is part of ufl4rom.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Import specialization of UFL classes from dolfin, dolfinx and firedrake backends."""

import typing

import ufl

try:
    import dolfin
except ImportError:
    DolfinScalarType = float

    class DolfinConstant(ufl.Constant):  # type: ignore[misc, no-any-unimported]
        """Mock dolfin.Constant class."""

        def __init__(  # type: ignore[no-any-unimported]
            self, value: typing.Union[DolfinScalarType, typing.Iterable[DolfinScalarType]],
            cell: typing.Optional[ufl.Cell] = None, name: typing.Optional[str] = None
        ) -> None:  # pragma: no cover
            raise RuntimeError("Cannot use a dolfin constant when dolfin is not installed")

    class DolfinFunction(ufl.Coefficient):  # type: ignore[misc, no-any-unimported]
        """Mock dolfin.Function class."""

        def name(self) -> str:  # pragma: no cover
            """Get function name."""
            raise RuntimeError("Cannot use a dolfin function when dolfin is not installed")
else:
    DolfinConstant = dolfin.Constant  # type: ignore
    DolfinFunction = dolfin.Function  # type: ignore
    DolfinScalarType = float  # type: ignore

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
            raise RuntimeError("Cannot use a dolfinx function when dolfin is not installed")
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

    class FiredrakeConstant(ufl.Constant):  # type: ignore[misc, no-any-unimported]
        """Mock firedrake.Constant class."""

        def __init__(  # type: ignore[no-any-unimported]
            self, value: typing.Union[FiredrakeScalarType, typing.Iterable[FiredrakeScalarType]],
            domain: typing.Optional[ufl.AbstractDomain] = None
        ) -> None:  # pragma: no cover
            raise RuntimeError("Cannot use a firedrake constant when firedrake is not installed")

    class FiredrakeFunction(ufl.Coefficient):  # type: ignore[misc, no-any-unimported]
        """Mock firedrake.Function class."""

        def name(self) -> str:  # pragma: no cover
            """Get function name."""
            raise RuntimeError("Cannot use a firedrake function when dolfin is not installed")
else:
    import petsc4py.PETSc

    FiredrakeConstant = firedrake.Constant  # type: ignore
    FiredrakeFunction = firedrake.Function  # type: ignore
    FiredrakeScalarType = petsc4py.PETSc.ScalarType  # type: ignore
