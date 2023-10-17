# Copyright (C) 2021-2023 by the ufl4rom authors
#
# This file is part of ufl4rom.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Unit tests for ufl4rom.utils.name module with forms on a scalar finite element."""

import ufl
import ufl.finiteelement
import ufl.pullback
import ufl.sobolevspace

import ufl4rom.utils


def test_name_vector_1() -> None:
    """Test a basic vector advection-diffusion-reaction parametrized form, with all parametrized coefficients."""
    cell = ufl.triangle
    dim = cell.geometric_dimension()
    scalar_element = ufl.finiteelement.FiniteElement(
        "Lagrange", cell, 1, (), ufl.pullback.identity_pullback, ufl.sobolevspace.H1)
    vector_element = ufl.finiteelement.FiniteElement(
        "Lagrange", cell, 1, (dim, ), ufl.pullback.identity_pullback, ufl.sobolevspace.H1)
    tensor_element = ufl.finiteelement.FiniteElement(
        "Lagrange", cell, 1, (dim, dim), ufl.pullback.identity_pullback, ufl.sobolevspace.H1)

    domain = ufl.Mesh(vector_element)
    scalar_function_space = ufl.FunctionSpace(domain, scalar_element)
    vector_function_space = ufl.FunctionSpace(domain, vector_element)
    tensor_function_space = ufl.FunctionSpace(domain, tensor_element)

    u = ufl.TrialFunction(vector_function_space)
    v = ufl.TestFunction(vector_function_space)
    f1 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 1, scalar", scalar_function_space)
    f2 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 2, vector", vector_function_space)
    f3 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 3, tensor", tensor_function_space)

    a1 = (
        ufl.inner(f3 * ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.inner(ufl.grad(u) * f2, v) * ufl.dx
        + f1 * ufl.inner(u, v) * ufl.dx
    )
    assert ufl4rom.utils.name(a1) == "4d7ccb1fdb7fd88183e133c23cb62b2a00380a5a"


def test_name_vector_2() -> None:
    """In this case the diffusivity tensor is given by the product of two expressions."""
    cell = ufl.triangle
    dim = cell.geometric_dimension()
    scalar_element = ufl.finiteelement.FiniteElement(
        "Lagrange", cell, 1, (), ufl.pullback.identity_pullback, ufl.sobolevspace.H1)
    vector_element = ufl.finiteelement.FiniteElement(
        "Lagrange", cell, 1, (dim, ), ufl.pullback.identity_pullback, ufl.sobolevspace.H1)
    tensor_element = ufl.finiteelement.FiniteElement(
        "Lagrange", cell, 1, (dim, dim), ufl.pullback.identity_pullback, ufl.sobolevspace.H1)

    domain = ufl.Mesh(vector_element)
    scalar_function_space = ufl.FunctionSpace(domain, scalar_element)
    vector_function_space = ufl.FunctionSpace(domain, vector_element)
    tensor_function_space = ufl.FunctionSpace(domain, tensor_element)

    u = ufl.TrialFunction(vector_function_space)
    v = ufl.TestFunction(vector_function_space)
    f1 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 1, scalar", scalar_function_space)
    f2 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 2, vector", vector_function_space)
    f3 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 3, tensor", tensor_function_space)
    f4 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 4, tensor", tensor_function_space)

    a2 = (
        ufl.inner(f3 * f4 * ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.inner(ufl.grad(u) * f2, v) * ufl.dx
        + f1 * ufl.inner(u, v) * ufl.dx
    )
    assert ufl4rom.utils.name(a2) == "dea7c3bf30df8007e185657959b9f7b267e68ef0"


def test_name_vector_3() -> None:
    """We try now with a more complex expression for each coefficient."""
    cell = ufl.triangle
    dim = cell.geometric_dimension()
    scalar_element = ufl.finiteelement.FiniteElement(
        "Lagrange", cell, 1, (), ufl.pullback.identity_pullback, ufl.sobolevspace.H1)
    vector_element = ufl.finiteelement.FiniteElement(
        "Lagrange", cell, 1, (dim, ), ufl.pullback.identity_pullback, ufl.sobolevspace.H1)
    tensor_element = ufl.finiteelement.FiniteElement(
        "Lagrange", cell, 1, (dim, dim), ufl.pullback.identity_pullback, ufl.sobolevspace.H1)

    domain = ufl.Mesh(vector_element)
    scalar_function_space = ufl.FunctionSpace(domain, scalar_element)
    vector_function_space = ufl.FunctionSpace(domain, vector_element)
    tensor_function_space = ufl.FunctionSpace(domain, tensor_element)

    u = ufl.TrialFunction(vector_function_space)
    v = ufl.TestFunction(vector_function_space)
    f1 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 1, scalar", scalar_function_space)
    f2 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 2, vector", vector_function_space)
    f3 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 3, tensor", tensor_function_space)
    f4 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 4, tensor", tensor_function_space)

    a3 = (
        ufl.inner(ufl.det(f3) * (f4 + f3 * f3) * f1, ufl.grad(v)) * ufl.dx + ufl.inner(ufl.grad(u) * f2, v) * ufl.dx
        + f1 * ufl.inner(u, v) * ufl.dx
    )
    assert ufl4rom.utils.name(a3) == "bb823bfbe14b6bf97bf91a92799228b1f05b2579"


def test_name_vector_4() -> None:
    """We add a term depending on the mesh size."""
    cell = ufl.triangle
    dim = cell.geometric_dimension()
    scalar_element = ufl.finiteelement.FiniteElement(
        "Lagrange", cell, 1, (), ufl.pullback.identity_pullback, ufl.sobolevspace.H1)
    vector_element = ufl.finiteelement.FiniteElement(
        "Lagrange", cell, 1, (dim, ), ufl.pullback.identity_pullback, ufl.sobolevspace.H1)
    tensor_element = ufl.finiteelement.FiniteElement(
        "Lagrange", cell, 1, (dim, dim), ufl.pullback.identity_pullback, ufl.sobolevspace.H1)

    domain = ufl.Mesh(vector_element)
    scalar_function_space = ufl.FunctionSpace(domain, scalar_element)
    vector_function_space = ufl.FunctionSpace(domain, vector_element)
    tensor_function_space = ufl.FunctionSpace(domain, tensor_element)

    u = ufl.TrialFunction(vector_function_space)
    v = ufl.TestFunction(vector_function_space)
    f1 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 1, scalar", scalar_function_space)
    f2 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 2, vector", vector_function_space)
    f3 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 3, tensor", tensor_function_space)
    h = ufl.CellDiameter(domain)

    a4 = (
        ufl.inner(f3 * h * ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.inner(ufl.grad(u) * f2 * h, v) * ufl.dx
        + f1 * h * ufl.inner(u, v) * ufl.dx
    )
    assert ufl4rom.utils.name(a4) == "76e4f07b1533077ae6c746907975357e642fd19e"


def test_name_vector_5() -> None:
    """Starting from form 4, use parenthesis to change the UFL tree."""
    cell = ufl.triangle
    dim = cell.geometric_dimension()
    scalar_element = ufl.finiteelement.FiniteElement(
        "Lagrange", cell, 1, (), ufl.pullback.identity_pullback, ufl.sobolevspace.H1)
    vector_element = ufl.finiteelement.FiniteElement(
        "Lagrange", cell, 1, (dim, ), ufl.pullback.identity_pullback, ufl.sobolevspace.H1)
    tensor_element = ufl.finiteelement.FiniteElement(
        "Lagrange", cell, 1, (dim, dim), ufl.pullback.identity_pullback, ufl.sobolevspace.H1)

    domain = ufl.Mesh(vector_element)
    scalar_function_space = ufl.FunctionSpace(domain, scalar_element)
    vector_function_space = ufl.FunctionSpace(domain, vector_element)
    tensor_function_space = ufl.FunctionSpace(domain, tensor_element)

    u = ufl.TrialFunction(vector_function_space)
    v = ufl.TestFunction(vector_function_space)
    f1 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 1, scalar", scalar_function_space)
    f2 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 2, vector", vector_function_space)
    f3 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 3, tensor", tensor_function_space)
    h = ufl.CellDiameter(domain)

    a5 = (
        ufl.inner((f3 * h) * ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.inner(ufl.grad(u) * (f2 * h), v) * ufl.dx
        + (f1 * h) * ufl.inner(u, v) * ufl.dx
    )
    assert ufl4rom.utils.name(a5) == "5e0dbde03943684ba9a1e0a9e89db0365fe8eeee"


def test_name_vector_6() -> None:
    """We change the coefficients to be non-parametrized."""
    cell = ufl.triangle
    dim = cell.geometric_dimension()
    scalar_element = ufl.finiteelement.FiniteElement(
        "Lagrange", cell, 1, (), ufl.pullback.identity_pullback, ufl.sobolevspace.H1)
    vector_element = ufl.finiteelement.FiniteElement(
        "Lagrange", cell, 1, (dim, ), ufl.pullback.identity_pullback, ufl.sobolevspace.H1)
    tensor_element = ufl.finiteelement.FiniteElement(
        "Lagrange", cell, 1, (dim, dim), ufl.pullback.identity_pullback, ufl.sobolevspace.H1)

    domain = ufl.Mesh(vector_element)
    scalar_function_space = ufl.FunctionSpace(domain, scalar_element)
    vector_function_space = ufl.FunctionSpace(domain, vector_element)
    tensor_function_space = ufl.FunctionSpace(domain, tensor_element)

    u = ufl.TrialFunction(vector_function_space)
    v = ufl.TestFunction(vector_function_space)
    f1 = ufl4rom.utils.NamedCoefficient("non-parametrized coefficient 1, scalar", scalar_function_space)
    f2 = ufl4rom.utils.NamedCoefficient("non-parametrized coefficient 2, vector", vector_function_space)
    f3 = ufl4rom.utils.NamedCoefficient("non-parametrized coefficient 3, tensor", tensor_function_space)

    a6 = (
        ufl.inner(f3 * ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.inner(ufl.grad(u) * f2, v) * ufl.dx
        + f1 * ufl.inner(u, v) * ufl.dx
    )
    assert ufl4rom.utils.name(a6) == "3a345b4dd17a21c595b8ffba677332b38550e181"


def test_name_vector_7() -> None:
    """A part of the diffusion coefficient is parametrized, while advection-reaction are not parametrized."""
    cell = ufl.triangle
    dim = cell.geometric_dimension()
    scalar_element = ufl.finiteelement.FiniteElement(
        "Lagrange", cell, 1, (), ufl.pullback.identity_pullback, ufl.sobolevspace.H1)
    vector_element = ufl.finiteelement.FiniteElement(
        "Lagrange", cell, 1, (dim, ), ufl.pullback.identity_pullback, ufl.sobolevspace.H1)
    tensor_element = ufl.finiteelement.FiniteElement(
        "Lagrange", cell, 1, (dim, dim), ufl.pullback.identity_pullback, ufl.sobolevspace.H1)

    domain = ufl.Mesh(vector_element)
    scalar_function_space = ufl.FunctionSpace(domain, scalar_element)
    vector_function_space = ufl.FunctionSpace(domain, vector_element)
    tensor_function_space = ufl.FunctionSpace(domain, tensor_element)

    u = ufl.TrialFunction(vector_function_space)
    v = ufl.TestFunction(vector_function_space)
    f1 = ufl4rom.utils.NamedCoefficient("non-parametrized coefficient 1, scalar", scalar_function_space)
    f2 = ufl4rom.utils.NamedCoefficient("non-parametrized coefficient 2, vector", vector_function_space)
    f3 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 3, tensor", tensor_function_space)
    f4 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 4, tensor", tensor_function_space)
    f5 = ufl4rom.utils.NamedCoefficient("non-parametrized coefficient 5, tensor", tensor_function_space)

    a7 = (
        ufl.inner(f5 * (f3 * f4) * ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.inner(ufl.grad(u) * f2, v) * ufl.dx
        + f1 * ufl.inner(u, v) * ufl.dx
    )
    assert ufl4rom.utils.name(a7) == "0357df9b11126d862868edef06adc3fcd944bf67"


def test_name_vector_8() -> None:
    """Test a case similar to form 7, but hwere the order of the matrix multiplication is different."""
    cell = ufl.triangle
    dim = cell.geometric_dimension()
    scalar_element = ufl.finiteelement.FiniteElement(
        "Lagrange", cell, 1, (), ufl.pullback.identity_pullback, ufl.sobolevspace.H1)
    vector_element = ufl.finiteelement.FiniteElement(
        "Lagrange", cell, 1, (dim, ), ufl.pullback.identity_pullback, ufl.sobolevspace.H1)
    tensor_element = ufl.finiteelement.FiniteElement(
        "Lagrange", cell, 1, (dim, dim), ufl.pullback.identity_pullback, ufl.sobolevspace.H1)

    domain = ufl.Mesh(vector_element)
    scalar_function_space = ufl.FunctionSpace(domain, scalar_element)
    vector_function_space = ufl.FunctionSpace(domain, vector_element)
    tensor_function_space = ufl.FunctionSpace(domain, tensor_element)

    u = ufl.TrialFunction(vector_function_space)
    v = ufl.TestFunction(vector_function_space)
    f1 = ufl4rom.utils.NamedCoefficient("non-parametrized coefficient 1, scalar", scalar_function_space)
    f2 = ufl4rom.utils.NamedCoefficient("non-parametrized coefficient 2, vector", vector_function_space)
    f3 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 3, tensor", tensor_function_space)
    f4 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 4, tensor", tensor_function_space)
    f5 = ufl4rom.utils.NamedCoefficient("non-parametrized coefficient 5, tensor", tensor_function_space)

    a8 = (
        ufl.inner(f3 * f5 * f4 * ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.inner(ufl.grad(u) * f2, v) * ufl.dx
        + f1 * ufl.inner(u, v) * ufl.dx
    )
    assert ufl4rom.utils.name(a8) == "ea72a5674f04d7051627861e0723684a791b60f3"


def test_name_vector_9() -> None:
    """Test a form similar to form 7, with a coefficient replaced by a constant."""
    cell = ufl.triangle
    dim = cell.geometric_dimension()
    scalar_element = ufl.finiteelement.FiniteElement(
        "Lagrange", cell, 1, (), ufl.pullback.identity_pullback, ufl.sobolevspace.H1)
    vector_element = ufl.finiteelement.FiniteElement(
        "Lagrange", cell, 1, (dim, ), ufl.pullback.identity_pullback, ufl.sobolevspace.H1)
    tensor_element = ufl.finiteelement.FiniteElement(
        "Lagrange", cell, 1, (dim, dim), ufl.pullback.identity_pullback, ufl.sobolevspace.H1)

    domain = ufl.Mesh(vector_element)
    scalar_function_space = ufl.FunctionSpace(domain, scalar_element)
    vector_function_space = ufl.FunctionSpace(domain, vector_element)
    tensor_function_space = ufl.FunctionSpace(domain, tensor_element)

    u = ufl.TrialFunction(vector_function_space)
    v = ufl.TestFunction(vector_function_space)
    f1 = ufl4rom.utils.NamedCoefficient("non-parametrized coefficient 1, scalar", scalar_function_space)
    f2 = ufl4rom.utils.NamedCoefficient("non-parametrized coefficient 2, vector", vector_function_space)
    f3 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 3, tensor", tensor_function_space)
    f4 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 4, tensor", tensor_function_space)
    c1 = ufl4rom.utils.NamedConstant("constant 1, tensor", domain, shape=(dim, dim))

    a9 = (
        ufl.inner(c1 * (f3 * f4) * ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.inner(ufl.grad(u) * f2, v) * ufl.dx
        + f1 * ufl.inner(u, v) * ufl.dx
    )
    assert ufl4rom.utils.name(a9) == "38c3ca9cbb94bae5474abfdd1cf45bb818bfd4af"


def test_name_vector_10() -> None:
    """Test a form similar to form 8, but the order of the matrix multiplication is different."""
    cell = ufl.triangle
    dim = cell.geometric_dimension()
    scalar_element = ufl.finiteelement.FiniteElement(
        "Lagrange", cell, 1, (), ufl.pullback.identity_pullback, ufl.sobolevspace.H1)
    vector_element = ufl.finiteelement.FiniteElement(
        "Lagrange", cell, 1, (dim, ), ufl.pullback.identity_pullback, ufl.sobolevspace.H1)
    tensor_element = ufl.finiteelement.FiniteElement(
        "Lagrange", cell, 1, (dim, dim), ufl.pullback.identity_pullback, ufl.sobolevspace.H1)

    domain = ufl.Mesh(vector_element)
    scalar_function_space = ufl.FunctionSpace(domain, scalar_element)
    vector_function_space = ufl.FunctionSpace(domain, vector_element)
    tensor_function_space = ufl.FunctionSpace(domain, tensor_element)

    u = ufl.TrialFunction(vector_function_space)
    v = ufl.TestFunction(vector_function_space)
    f1 = ufl4rom.utils.NamedCoefficient("non-parametrized coefficient 1, scalar", scalar_function_space)
    f2 = ufl4rom.utils.NamedCoefficient("non-parametrized coefficient 2, vector", vector_function_space)
    f3 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 3, tensor", tensor_function_space)
    f4 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 4, tensor", tensor_function_space)
    c1 = ufl4rom.utils.NamedConstant("constant 1, tensor", domain, shape=(dim, dim))

    a10 = (
        ufl.inner(f3 * c1 * f4 * ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.inner(ufl.grad(u) * f2, v) * ufl.dx
        + f1 * ufl.inner(u, v) * ufl.dx
    )
    assert ufl4rom.utils.name(a10) == "618d12fafea8615f3edf958ffb6aeb86349327ee"


def test_name_vector_11() -> None:
    """Test form similar to form 1, but where each term is multiplied by a Function, which represents the solution of\
    a parametrized problem."""
    cell = ufl.triangle
    dim = cell.geometric_dimension()
    scalar_element = ufl.finiteelement.FiniteElement(
        "Lagrange", cell, 1, (), ufl.pullback.identity_pullback, ufl.sobolevspace.H1)
    vector_element = ufl.finiteelement.FiniteElement(
        "Lagrange", cell, 1, (dim, ), ufl.pullback.identity_pullback, ufl.sobolevspace.H1)
    tensor_element = ufl.finiteelement.FiniteElement(
        "Lagrange", cell, 1, (dim, dim), ufl.pullback.identity_pullback, ufl.sobolevspace.H1)

    domain = ufl.Mesh(vector_element)
    scalar_function_space = ufl.FunctionSpace(domain, scalar_element)
    vector_function_space = ufl.FunctionSpace(domain, vector_element)
    tensor_function_space = ufl.FunctionSpace(domain, tensor_element)

    u = ufl.TrialFunction(vector_function_space)
    v = ufl.TestFunction(vector_function_space)
    s1 = ufl4rom.utils.NamedCoefficient("solution of parametrized problem 1, scalar", scalar_function_space)
    s2 = ufl4rom.utils.NamedCoefficient("solution of parametrized problem 2, vector", vector_function_space)
    s3 = ufl4rom.utils.NamedCoefficient("solution of parametrized problem 3, tensor", tensor_function_space)

    a11 = (
        ufl.inner(s3 * ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.inner(ufl.grad(u) * s2, v) * ufl.dx
        + s1 * ufl.inner(u, v) * ufl.dx
    )
    assert ufl4rom.utils.name(a11) == "27406321f198c978e9dce3bac8234efa1278fa6e"


def test_name_vector_12() -> None:
    """Test a form similar to form 11, but where each term is multiplied by a component of a solution of a parametrized\
    problem, resulting in an Indexed coefficient."""
    cell = ufl.triangle
    dim = cell.geometric_dimension()
    vector_element = ufl.finiteelement.FiniteElement(
        "Lagrange", cell, 1, (dim, ), ufl.pullback.identity_pullback, ufl.sobolevspace.H1)

    domain = ufl.Mesh(vector_element)
    vector_function_space = ufl.FunctionSpace(domain, vector_element)

    u = ufl.TrialFunction(vector_function_space)
    v = ufl.TestFunction(vector_function_space)
    s1 = ufl4rom.utils.NamedCoefficient("solution of parametrized problem 1", vector_function_space)

    a12 = s1[0] * ufl.inner(u, v) * ufl.dx
    assert ufl4rom.utils.name(a12) == "e858f87e6c5247729e9edc0506bb4c4fcb4b23e4"


def test_name_vector_13() -> None:
    """Test a form similar to form 11, but where each term is multiplied by a Function, which does not represent the\
    solution of a parametrized problem."""
    cell = ufl.triangle
    dim = cell.geometric_dimension()
    scalar_element = ufl.finiteelement.FiniteElement(
        "Lagrange", cell, 1, (), ufl.pullback.identity_pullback, ufl.sobolevspace.H1)
    vector_element = ufl.finiteelement.FiniteElement(
        "Lagrange", cell, 1, (dim, ), ufl.pullback.identity_pullback, ufl.sobolevspace.H1)
    tensor_element = ufl.finiteelement.FiniteElement(
        "Lagrange", cell, 1, (dim, dim), ufl.pullback.identity_pullback, ufl.sobolevspace.H1)

    domain = ufl.Mesh(vector_element)
    scalar_function_space = ufl.FunctionSpace(domain, scalar_element)
    vector_function_space = ufl.FunctionSpace(domain, vector_element)
    tensor_function_space = ufl.FunctionSpace(domain, tensor_element)

    u = ufl.TrialFunction(vector_function_space)
    v = ufl.TestFunction(vector_function_space)
    k1 = ufl4rom.utils.NamedCoefficient("auxiliary known function 1, scalar", scalar_function_space)
    k2 = ufl4rom.utils.NamedCoefficient("auxiliary known function 2, vector", vector_function_space)
    k3 = ufl4rom.utils.NamedCoefficient("auxiliary known function 3, tensor", tensor_function_space)

    a13 = (
        ufl.inner(k3 * ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.inner(ufl.grad(u) * k2, v) * ufl.dx
        + k1 * ufl.inner(u, v) * ufl.dx
    )
    assert ufl4rom.utils.name(a13) == "a690f711c57fc02e39bf36a06aa678ec8161b383"


def test_name_vector_14() -> None:
    """Test a form similar to form 12, but where each term is multiplied by a component of a Function which does not\
    represent the solution of a parametrized problem."""
    cell = ufl.triangle
    dim = cell.geometric_dimension()
    vector_element = ufl.finiteelement.FiniteElement(
        "Lagrange", cell, 1, (dim, ), ufl.pullback.identity_pullback, ufl.sobolevspace.H1)

    domain = ufl.Mesh(vector_element)
    vector_function_space = ufl.FunctionSpace(domain, vector_element)

    u = ufl.TrialFunction(vector_function_space)
    v = ufl.TestFunction(vector_function_space)
    k1 = ufl4rom.utils.NamedCoefficient("auxiliary known function 1", vector_function_space)

    a14 = k1[0] * ufl.inner(u, v) * ufl.dx
    assert ufl4rom.utils.name(a14) == "68a7774070de80683080d668345269c680a63d7c"


def test_name_vector_15() -> None:
    """Test a form similar to form 11, but where each term is multiplied by the gradientor partial derivative of\
    a Function, which represents the solution of a parametrized problem."""
    cell = ufl.triangle
    dim = cell.geometric_dimension()
    scalar_element = ufl.finiteelement.FiniteElement(
        "Lagrange", cell, 1, (), ufl.pullback.identity_pullback, ufl.sobolevspace.H1)
    vector_element = ufl.finiteelement.FiniteElement(
        "Lagrange", cell, 1, (dim, ), ufl.pullback.identity_pullback, ufl.sobolevspace.H1)

    domain = ufl.Mesh(vector_element)
    scalar_function_space = ufl.FunctionSpace(domain, scalar_element)
    vector_function_space = ufl.FunctionSpace(domain, vector_element)

    u = ufl.TrialFunction(vector_function_space)
    v = ufl.TestFunction(vector_function_space)
    s1 = ufl4rom.utils.NamedCoefficient("solution of parametrized problem 1, scalar", scalar_function_space)
    s2 = ufl4rom.utils.NamedCoefficient("solution of parametrized problem 2, vector", vector_function_space)

    a15 = (
        ufl.inner(ufl.grad(s2) * ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.inner(ufl.grad(u) * ufl.grad(s1), v) * ufl.dx
        + s1.dx(0) * ufl.inner(u, v) * ufl.dx
    )
    assert ufl4rom.utils.name(a15) == "8173d81b4aa22b52dc8e18ea015f04fc50f7645e"


def test_name_vector_16() -> None:
    """Test a form similar to form 12, but where each term is multiplied by the gradient or partial derivative of\
    a component of a solution of a parametrized problem."""
    cell = ufl.triangle
    dim = cell.geometric_dimension()
    vector_element = ufl.finiteelement.FiniteElement(
        "Lagrange", cell, 1, (dim, ), ufl.pullback.identity_pullback, ufl.sobolevspace.H1)

    domain = ufl.Mesh(vector_element)
    vector_function_space = ufl.FunctionSpace(domain, vector_element)

    u = ufl.TrialFunction(vector_function_space)
    v = ufl.TestFunction(vector_function_space)
    s1 = ufl4rom.utils.NamedCoefficient("solution of parametrized problem 1", vector_function_space)

    a16 = ufl.inner(ufl.grad(s1[0]), u) * v[0] * ufl.dx + s1[0].dx(0) * ufl.inner(u, v) * ufl.dx
    assert ufl4rom.utils.name(a16) == "b1fd42d1e0b54fd297a2b2c787e0d0df7111317a"


def test_name_vector_17() -> None:
    """Test a form is similar to form 13, but where each term is multiplied by a the gradient or partial derivative of\
    a Function, which does not represent the solution of a parametrized problem."""
    cell = ufl.triangle
    dim = cell.geometric_dimension()
    scalar_element = ufl.finiteelement.FiniteElement(
        "Lagrange", cell, 1, (), ufl.pullback.identity_pullback, ufl.sobolevspace.H1)
    vector_element = ufl.finiteelement.FiniteElement(
        "Lagrange", cell, 1, (dim, ), ufl.pullback.identity_pullback, ufl.sobolevspace.H1)

    domain = ufl.Mesh(vector_element)
    scalar_function_space = ufl.FunctionSpace(domain, scalar_element)
    vector_function_space = ufl.FunctionSpace(domain, vector_element)

    u = ufl.TrialFunction(vector_function_space)
    v = ufl.TestFunction(vector_function_space)
    k1 = ufl4rom.utils.NamedCoefficient("auxiliary known function 1, scalar", scalar_function_space)
    k2 = ufl4rom.utils.NamedCoefficient("auxiliary known function 2, vector", vector_function_space)

    a17 = (
        ufl.inner(ufl.grad(k2) * ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.inner(ufl.grad(u) * ufl.grad(k1), v) * ufl.dx
        + k1.dx(0) * ufl.inner(u, v) * ufl.dx
    )
    assert ufl4rom.utils.name(a17) == "b06397e64909858b76a0f27023e30b854fb3896c"


def test_name_vector_18() -> None:
    """Test a form similar to form 14, but where each term is multiplied by the gradient or partial derivative of\
    a component of a Function which does not represent the solution of a parametrized problem."""
    cell = ufl.triangle
    dim = cell.geometric_dimension()
    vector_element = ufl.finiteelement.FiniteElement(
        "Lagrange", cell, 1, (dim, ), ufl.pullback.identity_pullback, ufl.sobolevspace.H1)

    domain = ufl.Mesh(vector_element)
    vector_function_space = ufl.FunctionSpace(domain, vector_element)

    u = ufl.TrialFunction(vector_function_space)
    v = ufl.TestFunction(vector_function_space)
    k1 = ufl4rom.utils.NamedCoefficient("auxiliary known function 1", vector_function_space)

    a18 = ufl.inner(ufl.grad(k1[0]), u) * v[0] * ufl.dx + k1[0].dx(0) * ufl.inner(u, v) * ufl.dx
    assert ufl4rom.utils.name(a18) == "853ebe55df8ad40564691df453b75c6298d56768"
