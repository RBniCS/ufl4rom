# Copyright (C) 2021-2024 by the ufl4rom authors
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
    dim = 2
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
    assert ufl4rom.utils.name(a1) == "995f2ee481c6b1c6734365e13747b1fb9bb8d5c0"


def test_name_vector_2() -> None:
    """In this case the diffusivity tensor is given by the product of two expressions."""
    cell = ufl.triangle
    dim = 2
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
    assert ufl4rom.utils.name(a2) == "8d6d724534788a1c72af37f24e098d58db996a87"


def test_name_vector_3() -> None:
    """We try now with a more complex expression for each coefficient."""
    cell = ufl.triangle
    dim = 2
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
    assert ufl4rom.utils.name(a3) == "91e93fb8c7a197f2fc1e704e0a60ac0c2d9c164a"


def test_name_vector_4() -> None:
    """We add a term depending on the mesh size."""
    cell = ufl.triangle
    dim = 2
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
    assert ufl4rom.utils.name(a4) == "df4a6b8c7621dbaf35ad614640267d236b1237b2"


def test_name_vector_5() -> None:
    """Starting from form 4, use parenthesis to change the UFL tree."""
    cell = ufl.triangle
    dim = 2
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
    assert ufl4rom.utils.name(a5) == "97dea8b4bab81d823ab99553bca574432c89a5d3"


def test_name_vector_6() -> None:
    """We change the coefficients to be non-parametrized."""
    cell = ufl.triangle
    dim = 2
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
    assert ufl4rom.utils.name(a6) == "1b51aeb4f9a8f407d4c16db86290b518ddbbec1e"


def test_name_vector_7() -> None:
    """A part of the diffusion coefficient is parametrized, while advection-reaction are not parametrized."""
    cell = ufl.triangle
    dim = 2
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
    assert ufl4rom.utils.name(a7) == "9fa2219f43ab12049190e21ef29a520a952e61c5"


def test_name_vector_8() -> None:
    """Test a case similar to form 7, but hwere the order of the matrix multiplication is different."""
    cell = ufl.triangle
    dim = 2
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
    assert ufl4rom.utils.name(a8) == "1383df8391d4e0a4f9f0a8e3f51bcf481da17bb1"


def test_name_vector_9() -> None:
    """Test a form similar to form 7, with a coefficient replaced by a constant."""
    cell = ufl.triangle
    dim = 2
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
    assert ufl4rom.utils.name(a9) == "5babf74d869ea0d42d8442b1e3fd754ef9a74b31"


def test_name_vector_10() -> None:
    """Test a form similar to form 8, but the order of the matrix multiplication is different."""
    cell = ufl.triangle
    dim = 2
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
    assert ufl4rom.utils.name(a10) == "4468a1dd9aaded7f7200996dea5362aae17c4497"


def test_name_vector_11() -> None:
    """Test form similar to form 1, but where each term is multiplied by a solution."""
    cell = ufl.triangle
    dim = 2
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
    assert ufl4rom.utils.name(a11) == "8b6f43ef8651d58691a26a96dffd54318ec33dc6"


def test_name_vector_12() -> None:
    """Test a form similar to form 11, but where each term is multiplied by a component of a solution.

    The extraction of a component results in an Indexed coefficient.
    """
    cell = ufl.triangle
    dim = 2
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
    """Test a form similar to form 11, but where each term is multiplied by a function.

    In contrast to form 11, the function does not represent the solution of a parametrized problem.
    """
    cell = ufl.triangle
    dim = 2
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
    assert ufl4rom.utils.name(a13) == "5544b3e92ad8232018f9878a20c12a78e678922f"


def test_name_vector_14() -> None:
    """Test a form similar to form 12, but where each term is multiplied by a component of a function.

    In contrast to form 12, the function does not represent the solution of a parametrized problem.
    """
    cell = ufl.triangle
    dim = 2
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
    """Test a form similar to form 11, but where each term is multiplied by the gradient of a solution."""
    cell = ufl.triangle
    dim = 2
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
    assert ufl4rom.utils.name(a15) == "1caa4c6ebfcca238e20e117e20cfbb216d0e7f99"


def test_name_vector_16() -> None:
    """Test a form similar to form 12, but where each term is multiplied by the gradient of a solution."""
    cell = ufl.triangle
    dim = 2
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
    """Test a form is similar to form 13, but where each term is multiplied by the gradient of a function.

    In contrast to form 17, the function does not represent the solution of a parametrized problem.
    """
    cell = ufl.triangle
    dim = 2
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
    assert ufl4rom.utils.name(a17) == "dcfd7eac71d7d79d2c8a8254b1ad7dee668f94d6"


def test_name_vector_18() -> None:
    """Test a form similar to form 14, but where each term is multiplied by the gradient of a component of a function.

    In contrast to form 14, the function does not represent the solution of a parametrized problem.
    """
    cell = ufl.triangle
    dim = 2
    vector_element = ufl.finiteelement.FiniteElement(
        "Lagrange", cell, 1, (dim, ), ufl.pullback.identity_pullback, ufl.sobolevspace.H1)

    domain = ufl.Mesh(vector_element)
    vector_function_space = ufl.FunctionSpace(domain, vector_element)

    u = ufl.TrialFunction(vector_function_space)
    v = ufl.TestFunction(vector_function_space)
    k1 = ufl4rom.utils.NamedCoefficient("auxiliary known function 1", vector_function_space)

    a18 = ufl.inner(ufl.grad(k1[0]), u) * v[0] * ufl.dx + k1[0].dx(0) * ufl.inner(u, v) * ufl.dx
    assert ufl4rom.utils.name(a18) == "853ebe55df8ad40564691df453b75c6298d56768"
