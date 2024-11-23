# Copyright (C) 2021-2024 by the ufl4rom authors
#
# This file is part of ufl4rom.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Unit tests for ufl4rom.utils.name module with forms on a scalar finite element."""


import pytest
import ufl
import ufl.finiteelement
import ufl.pullback
import ufl.sobolevspace

import ufl4rom.utils


def test_name_scalar_1() -> None:
    """Test a basic advection-diffusion-reaction parametrized form, with all parametrized coefficients."""
    cell = ufl.triangle
    dim = 2
    scalar_element = ufl.finiteelement.FiniteElement(
        "Lagrange", cell, 1, (), ufl.pullback.identity_pullback, ufl.sobolevspace.H1)
    vector_element = ufl.finiteelement.FiniteElement(
        "Lagrange", cell, 1, (dim, ), ufl.pullback.identity_pullback, ufl.sobolevspace.H1)

    domain = ufl.Mesh(vector_element)
    scalar_function_space = ufl.FunctionSpace(domain, scalar_element)

    u = ufl.TrialFunction(scalar_function_space)
    v = ufl.TestFunction(scalar_function_space)
    f1 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 1", scalar_function_space)
    f2 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 2", scalar_function_space)
    f3 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 3", scalar_function_space)

    a1 = f3 * f2 * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + f2 * u.dx(0) * v * ufl.dx + f1 * u * v * ufl.dx
    assert ufl4rom.utils.name(a1) == "29a2c37955642711c70c20aa631185f5cd47f041"


def test_name_scalar_2() -> None:
    """We move the diffusion coefficient inside the ufl.inner product.

    This requires splitting it into the two arguments of the ufl.inner product.
    """
    cell = ufl.triangle
    dim = 2
    scalar_element = ufl.finiteelement.FiniteElement(
        "Lagrange", cell, 1, (), ufl.pullback.identity_pullback, ufl.sobolevspace.H1)
    vector_element = ufl.finiteelement.FiniteElement(
        "Lagrange", cell, 1, (dim, ), ufl.pullback.identity_pullback, ufl.sobolevspace.H1)

    domain = ufl.Mesh(vector_element)
    scalar_function_space = ufl.FunctionSpace(domain, scalar_element)

    u = ufl.TrialFunction(scalar_function_space)
    v = ufl.TestFunction(scalar_function_space)
    f1 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 1", scalar_function_space)
    f2 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 2", scalar_function_space)
    f3 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 3", scalar_function_space)

    a2 = ufl.inner(f3 * ufl.grad(u), f2 * ufl.grad(v)) * ufl.dx + f2 * u.dx(0) * v * ufl.dx + f1 * u * v * ufl.dx
    assert ufl4rom.utils.name(a2) == "1a57764c5384bd8280d60bf6408c95b1b89cfc1e"


def test_name_scalar_3() -> None:
    """Test a form containing a sum of coefficients."""
    cell = ufl.triangle
    dim = 2
    scalar_element = ufl.finiteelement.FiniteElement(
        "Lagrange", cell, 1, (), ufl.pullback.identity_pullback, ufl.sobolevspace.H1)
    vector_element = ufl.finiteelement.FiniteElement(
        "Lagrange", cell, 1, (dim, ), ufl.pullback.identity_pullback, ufl.sobolevspace.H1)

    domain = ufl.Mesh(vector_element)
    scalar_function_space = ufl.FunctionSpace(domain, scalar_element)

    u = ufl.TrialFunction(scalar_function_space)
    v = ufl.TestFunction(scalar_function_space)
    f1 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 1", scalar_function_space)
    f2 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 2", scalar_function_space)
    f3 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 3", scalar_function_space)

    a3 = (
        f3 * f2 * (1 + f1 * f2) * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + f2 * (1 + f2 * f3) * u.dx(0) * v * ufl.dx
        + f3 * (1 + f1 * f2) * u * v * ufl.dx
    )
    assert ufl4rom.utils.name(a3) == "6c8d199c4c4fef59fe4fca44b0e8096f34bbba88"


def test_name_scalar_4() -> None:
    """We use now a diffusivity tensor and a vector convective field.

    This cases tests if matrix and vector coefficients are correctly detected.
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

    u = ufl.TrialFunction(scalar_function_space)
    v = ufl.TestFunction(scalar_function_space)
    f1 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 1, scalar", scalar_function_space)
    f2 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 2, scalar", scalar_function_space)
    f3 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 3, scalar", scalar_function_space)
    f4 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 4, vector", vector_function_space)
    f5 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 5, tensor", tensor_function_space)

    a4 = (
        ufl.inner(f5 * (1 + f1 * f2) * ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.inner(f4, ufl.grad(u)) * v * ufl.dx
        + f3 * u * v * ufl.dx
    )
    assert ufl4rom.utils.name(a4) == "b120cea9b80f224c34689e66b177bfd36c2dfbe7"


def test_name_scalar_5() -> None:
    """We change the integration domain to be the boundary. This is a variation of form 1."""
    cell = ufl.triangle
    dim = 2
    scalar_element = ufl.finiteelement.FiniteElement(
        "Lagrange", cell, 1, (), ufl.pullback.identity_pullback, ufl.sobolevspace.H1)
    vector_element = ufl.finiteelement.FiniteElement(
        "Lagrange", cell, 1, (dim, ), ufl.pullback.identity_pullback, ufl.sobolevspace.H1)

    domain = ufl.Mesh(vector_element)
    scalar_function_space = ufl.FunctionSpace(domain, scalar_element)

    u = ufl.TrialFunction(scalar_function_space)
    v = ufl.TestFunction(scalar_function_space)
    f1 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 1", scalar_function_space)
    f2 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 2", scalar_function_space)
    f3 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 3", scalar_function_space)

    a5 = f3 * f2 * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.ds + f2 * u.dx(0) * v * ufl.ds + f1 * u * v * ufl.ds
    assert ufl4rom.utils.name(a5) == "bcf86769ae751ff2a511ef84f7854ed6674e193a"


def test_name_scalar_6() -> None:
    """We add a term depending on the mesh size."""
    cell = ufl.triangle
    dim = 2
    scalar_element = ufl.finiteelement.FiniteElement(
        "Lagrange", cell, 1, (), ufl.pullback.identity_pullback, ufl.sobolevspace.H1)
    vector_element = ufl.finiteelement.FiniteElement(
        "Lagrange", cell, 1, (dim, ), ufl.pullback.identity_pullback, ufl.sobolevspace.H1)

    domain = ufl.Mesh(vector_element)
    scalar_function_space = ufl.FunctionSpace(domain, scalar_element)

    u = ufl.TrialFunction(scalar_function_space)
    v = ufl.TestFunction(scalar_function_space)
    f1 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 1", scalar_function_space)
    h = ufl.CellDiameter(domain)

    a6 = f1 * h * u * v * ufl.dx
    assert ufl4rom.utils.name(a6) == "993b2225a49cf67301e05df09921eac64f41974f"


def test_name_scalar_7() -> None:
    """We change the coefficients to be non-parametrized."""
    cell = ufl.triangle
    dim = 2
    scalar_element = ufl.finiteelement.FiniteElement(
        "Lagrange", cell, 1, (), ufl.pullback.identity_pullback, ufl.sobolevspace.H1)
    vector_element = ufl.finiteelement.FiniteElement(
        "Lagrange", cell, 1, (dim, ), ufl.pullback.identity_pullback, ufl.sobolevspace.H1)

    domain = ufl.Mesh(vector_element)
    scalar_function_space = ufl.FunctionSpace(domain, scalar_element)

    u = ufl.TrialFunction(scalar_function_space)
    v = ufl.TestFunction(scalar_function_space)
    f1 = ufl4rom.utils.NamedCoefficient("non-parametrized coefficient 1", scalar_function_space)
    f2 = ufl4rom.utils.NamedCoefficient("non-parametrized coefficient 2", scalar_function_space)
    f3 = ufl4rom.utils.NamedCoefficient("non-parametrized coefficient 3", scalar_function_space)

    a7 = f3 * f2 * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + f2 * u.dx(0) * v * ufl.dx + f1 * u * v * ufl.dx
    assert ufl4rom.utils.name(a7) == "8d4ea7b2ddd2d75c06a2a6d34d8d07016e61afb5"


def test_name_scalar_8() -> None:
    """A part of the diffusion coefficient is parametrized, while advection-reaction are not parametrized."""
    cell = ufl.triangle
    dim = 2
    scalar_element = ufl.finiteelement.FiniteElement(
        "Lagrange", cell, 1, (), ufl.pullback.identity_pullback, ufl.sobolevspace.H1)
    vector_element = ufl.finiteelement.FiniteElement(
        "Lagrange", cell, 1, (dim, ), ufl.pullback.identity_pullback, ufl.sobolevspace.H1)

    domain = ufl.Mesh(vector_element)
    scalar_function_space = ufl.FunctionSpace(domain, scalar_element)

    u = ufl.TrialFunction(scalar_function_space)
    v = ufl.TestFunction(scalar_function_space)
    f1 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 1", scalar_function_space)
    f2 = ufl4rom.utils.NamedCoefficient("non-parametrized coefficient 2", scalar_function_space)
    f3 = ufl4rom.utils.NamedCoefficient("non-parametrized coefficient 3", scalar_function_space)

    a8 = f1 * f3 * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + f2 * u.dx(0) * v * ufl.dx + f3 * u * v * ufl.dx
    assert ufl4rom.utils.name(a8) == "a52c28fdfa7dee5529cca81fb87a9b8631368d1a"


def test_name_scalar_9() -> None:
    """A part of the diffusion coefficient is parametrized, while advection-reaction terms are not parametrized.

    The terms in this test are written in a different way when compared to form 8.
    """
    cell = ufl.triangle
    dim = 2
    scalar_element = ufl.finiteelement.FiniteElement(
        "Lagrange", cell, 1, (), ufl.pullback.identity_pullback, ufl.sobolevspace.H1)
    vector_element = ufl.finiteelement.FiniteElement(
        "Lagrange", cell, 1, (dim, ), ufl.pullback.identity_pullback, ufl.sobolevspace.H1)

    domain = ufl.Mesh(vector_element)
    scalar_function_space = ufl.FunctionSpace(domain, scalar_element)

    u = ufl.TrialFunction(scalar_function_space)
    v = ufl.TestFunction(scalar_function_space)
    f1 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 1", scalar_function_space)
    f2 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 2", scalar_function_space)
    f3 = ufl4rom.utils.NamedCoefficient("non-parametrized coefficient 3", scalar_function_space)
    f4 = ufl4rom.utils.NamedCoefficient("non-parametrized coefficient 4", scalar_function_space)

    a9 = (
        f2 * f4 / f1 * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + f3 * u.dx(0) * v * ufl.dx
        + f4 * u * v * ufl.dx
    )
    assert ufl4rom.utils.name(a9) == "2b5d676cf1559f4b3c368445e3abace5b0b236b9"


def test_name_scalar_10() -> None:
    """Similarly to form 6, we add a term depending on the mesh size multiplied by a non-parametrized coefficient."""
    cell = ufl.triangle
    dim = 2
    scalar_element = ufl.finiteelement.FiniteElement(
        "Lagrange", cell, 1, (), ufl.pullback.identity_pullback, ufl.sobolevspace.H1)
    vector_element = ufl.finiteelement.FiniteElement(
        "Lagrange", cell, 1, (dim, ), ufl.pullback.identity_pullback, ufl.sobolevspace.H1)

    domain = ufl.Mesh(vector_element)
    scalar_function_space = ufl.FunctionSpace(domain, scalar_element)

    u = ufl.TrialFunction(scalar_function_space)
    v = ufl.TestFunction(scalar_function_space)
    f1 = ufl4rom.utils.NamedCoefficient("non-parametrized coefficient 1", scalar_function_space)
    h = ufl.CellDiameter(domain)

    a10 = f1 * h * u * v * ufl.dx
    assert ufl4rom.utils.name(a10) == "ed3508b6031f35d4a7861ab2de79de1562519e0e"


def test_name_scalar_11() -> None:
    """Add a term depending on the mesh size multiplied by the product of parametrized and a non-parametrized coeffs.

    This is simalar to test 6.
    """
    cell = ufl.triangle
    dim = 2
    scalar_element = ufl.finiteelement.FiniteElement(
        "Lagrange", cell, 1, (), ufl.pullback.identity_pullback, ufl.sobolevspace.H1)
    vector_element = ufl.finiteelement.FiniteElement(
        "Lagrange", cell, 1, (dim, ), ufl.pullback.identity_pullback, ufl.sobolevspace.H1)

    domain = ufl.Mesh(vector_element)
    scalar_function_space = ufl.FunctionSpace(domain, scalar_element)

    u = ufl.TrialFunction(scalar_function_space)
    v = ufl.TestFunction(scalar_function_space)
    f1 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 1", scalar_function_space)
    f2 = ufl4rom.utils.NamedCoefficient("non-parametrized coefficient 2", scalar_function_space)
    h = ufl.CellDiameter(domain)

    a11 = f2 * f1 * h * u * v * ufl.dx
    assert ufl4rom.utils.name(a11) == "e55b2e59210033b0b0aca238774012a2c0aae65d"


def test_name_scalar_12() -> None:
    """We change form 11 with a slightly different coefficient."""
    cell = ufl.triangle
    dim = 2
    scalar_element = ufl.finiteelement.FiniteElement(
        "Lagrange", cell, 1, (), ufl.pullback.identity_pullback, ufl.sobolevspace.H1)
    vector_element = ufl.finiteelement.FiniteElement(
        "Lagrange", cell, 1, (dim, ), ufl.pullback.identity_pullback, ufl.sobolevspace.H1)

    domain = ufl.Mesh(vector_element)
    scalar_function_space = ufl.FunctionSpace(domain, scalar_element)

    u = ufl.TrialFunction(scalar_function_space)
    v = ufl.TestFunction(scalar_function_space)
    f1 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 1", scalar_function_space)
    f2 = ufl4rom.utils.NamedCoefficient("non-parametrized coefficient 2", scalar_function_space)
    h = ufl.CellDiameter(domain)

    a12 = f1 * h / f2 * u * v * ufl.dx
    assert ufl4rom.utils.name(a12) == "8ef519458d1fc45550a16d4d9408984b21755a65"


def test_name_scalar_13() -> None:
    """We now introduce constants in the expression."""
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

    u = ufl.TrialFunction(scalar_function_space)
    v = ufl.TestFunction(scalar_function_space)
    f1 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 1, scalar", scalar_function_space)
    f2 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 2, vector", vector_function_space)
    f3 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 3, tensor", tensor_function_space)
    c1 = ufl4rom.utils.NamedConstant("constant 1, scalar", domain, shape=())
    c2 = ufl4rom.utils.NamedConstant("constant 2, tensor", domain, shape=(dim, dim))

    a13 = (
        ufl.inner(c2 * f3 * c1 * ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.inner(c1 * f2, ufl.grad(u)) * v * ufl.dx
        + c1 * f1 * u * v * ufl.dx
    )
    assert ufl4rom.utils.name(a13) == "05e0dc97e56ecfafc4d036cf18d56e66792897b7"


def test_name_scalar_14() -> None:
    """Test a form similar to form 1, but where each term is multiplied by a scalar function.

    The scalar function represents the solution of a parametrized problem.
    """
    cell = ufl.triangle
    dim = 2
    scalar_element = ufl.finiteelement.FiniteElement(
        "Lagrange", cell, 1, (), ufl.pullback.identity_pullback, ufl.sobolevspace.H1)
    vector_element = ufl.finiteelement.FiniteElement(
        "Lagrange", cell, 1, (dim, ), ufl.pullback.identity_pullback, ufl.sobolevspace.H1)

    domain = ufl.Mesh(vector_element)
    scalar_function_space = ufl.FunctionSpace(domain, scalar_element)

    u = ufl.TrialFunction(scalar_function_space)
    v = ufl.TestFunction(scalar_function_space)
    f1 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 1", scalar_function_space)
    f2 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 2", scalar_function_space)
    f3 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 3", scalar_function_space)
    s1 = ufl4rom.utils.NamedCoefficient("solution of parametrized problem 1", scalar_function_space)

    a14 = (
        f3 * s1 * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + s1 / f2 * u.dx(0) * v * ufl.dx
        + f1 * s1 * u * v * ufl.dx
    )
    assert ufl4rom.utils.name(a14) == "08301071bbc84ede7b0460ea1c06e2b3c25522bb"


def test_name_scalar_15() -> None:
    """Test a form similar to form 4, but where each term is replaced by a solution of a parametrized problem.

    Scalar, vector or tensor shaped solutions are considered.
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

    u = ufl.TrialFunction(scalar_function_space)
    v = ufl.TestFunction(scalar_function_space)
    s1 = ufl4rom.utils.NamedCoefficient("solution of parametrized problem 1, scalar", scalar_function_space)
    s2 = ufl4rom.utils.NamedCoefficient("solution of parametrized problem 2, vector", vector_function_space)
    s3 = ufl4rom.utils.NamedCoefficient("solution of parametrized problem 3, tensor", tensor_function_space)

    a15 = (
        ufl.inner(s3 * ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.inner(s2, ufl.grad(u)) * v * ufl.dx
        + s1 * u * v * ufl.dx
    )
    assert ufl4rom.utils.name(a15) == "7b06667717ffda599f2eff4154eff6fbd32513d5"


def test_name_scalar_16() -> None:
    """We multiply non-parametrized coefficients (as in form 7) and solutions."""
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

    u = ufl.TrialFunction(scalar_function_space)
    v = ufl.TestFunction(scalar_function_space)
    f1 = ufl4rom.utils.NamedCoefficient("non-parametrized coefficient 1, scalar", scalar_function_space)
    f2 = ufl4rom.utils.NamedCoefficient("non-parametrized coefficient 2, scalar", scalar_function_space)
    s1 = ufl4rom.utils.NamedCoefficient("solution of parametrized problem 1, scalar", scalar_function_space)
    s2 = ufl4rom.utils.NamedCoefficient("solution of parametrized problem 2, vector", vector_function_space)
    s3 = ufl4rom.utils.NamedCoefficient("solution of parametrized problem 3, tensor", tensor_function_space)

    a16 = (
        ufl.inner(f2 * s3 * f1 * ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.inner(s2, ufl.grad(u)) * v * ufl.dx
        + s1 * u * v * ufl.dx
    )
    assert ufl4rom.utils.name(a16) == "4c4f4d14900cec34584468991e5dfdff782ce159"


def test_name_scalar_17() -> None:
    """Test a form similar to form 15, but where each term is multiplied to a component of a solution."""
    cell = ufl.triangle
    dim = 2
    scalar_element = ufl.finiteelement.FiniteElement(
        "Lagrange", cell, 1, (), ufl.pullback.identity_pullback, ufl.sobolevspace.H1)
    vector_element = ufl.finiteelement.FiniteElement(
        "Lagrange", cell, 1, (dim, ), ufl.pullback.identity_pullback, ufl.sobolevspace.H1)

    domain = ufl.Mesh(vector_element)
    scalar_function_space = ufl.FunctionSpace(domain, scalar_element)
    vector_function_space = ufl.FunctionSpace(domain, vector_element)

    u = ufl.TrialFunction(scalar_function_space)
    v = ufl.TestFunction(scalar_function_space)
    s1 = ufl4rom.utils.NamedCoefficient("solution of parametrized problem 1, vector", vector_function_space)

    a17 = s1[0] * u * v * ufl.dx + s1[1] * u.dx(0) * v * ufl.dx
    assert ufl4rom.utils.name(a17) == "5e3f5406f5f0ecc79bef3517721b520a276cf761"


def test_name_scalar_18() -> None:
    """Test a form similar to form 14, but where each term is multiplied by a scalar function.

    In contrast to form 14, the function  does not represent the solution of a parametrized problem.
    """
    cell = ufl.triangle
    dim = 2
    scalar_element = ufl.finiteelement.FiniteElement(
        "Lagrange", cell, 1, (), ufl.pullback.identity_pullback, ufl.sobolevspace.H1)
    vector_element = ufl.finiteelement.FiniteElement(
        "Lagrange", cell, 1, (dim, ), ufl.pullback.identity_pullback, ufl.sobolevspace.H1)

    domain = ufl.Mesh(vector_element)
    scalar_function_space = ufl.FunctionSpace(domain, scalar_element)

    u = ufl.TrialFunction(scalar_function_space)
    v = ufl.TestFunction(scalar_function_space)
    f1 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 1", scalar_function_space)
    f2 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 2", scalar_function_space)
    f3 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 3", scalar_function_space)
    k1 = ufl4rom.utils.NamedCoefficient("auxiliary known function 1", scalar_function_space)

    a18 = (
        f3 * k1 * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + f2 * k1 * u.dx(0) * v * ufl.dx
        + f1 * k1 * u * v * ufl.dx
    )
    assert ufl4rom.utils.name(a18) == "09c8f1a52eaf4bd52cd5345933ae10bdcda6d599"


def test_name_scalar_19() -> None:
    """Test a form similar to form 16, but where each term is multiplied by a scalar function.

    In contrast to form 16, the function does not represent the solution of a parametrized problem.
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

    u = ufl.TrialFunction(scalar_function_space)
    v = ufl.TestFunction(scalar_function_space)
    f1 = ufl4rom.utils.NamedCoefficient("non-parametrized coefficient 1, scalar", scalar_function_space)
    f2 = ufl4rom.utils.NamedCoefficient("non-parametrized coefficient 2, scalar", scalar_function_space)
    k1 = ufl4rom.utils.NamedCoefficient("auxiliary known function 1, scalar", scalar_function_space)
    k2 = ufl4rom.utils.NamedCoefficient("auxiliary known function 2, vector", vector_function_space)
    k3 = ufl4rom.utils.NamedCoefficient("auxiliary known function 3, tensor", tensor_function_space)

    a19 = (
        ufl.inner(f2 * k3 * f1 * ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.inner(k2, ufl.grad(u)) * v * ufl.dx
        + k1 * u * v * ufl.dx
    )
    assert ufl4rom.utils.name(a19) == "8e13841a2956f772bfbd27775d5411c039f95028"


def test_name_scalar_20() -> None:
    """Test a form similar to form 17, but where each term is multiplied to a component of a function.

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

    u = ufl.TrialFunction(scalar_function_space)
    v = ufl.TestFunction(scalar_function_space)
    k1 = ufl4rom.utils.NamedCoefficient("auxiliary known function 1, vector", vector_function_space)

    a20 = k1[0] * u * v * ufl.dx + k1[1] * u.dx(0) * v * ufl.dx
    assert ufl4rom.utils.name(a20) == "2a396fd6eb486b38d8fead49e8e078508034fc81"


def test_name_scalar_21() -> None:
    """Test a form similar to form 17, but where each term is multiplied to the gradient of a solution."""
    cell = ufl.triangle
    dim = 2
    scalar_element = ufl.finiteelement.FiniteElement(
        "Lagrange", cell, 1, (), ufl.pullback.identity_pullback, ufl.sobolevspace.H1)
    vector_element = ufl.finiteelement.FiniteElement(
        "Lagrange", cell, 1, (dim, ), ufl.pullback.identity_pullback, ufl.sobolevspace.H1)

    domain = ufl.Mesh(vector_element)
    scalar_function_space = ufl.FunctionSpace(domain, scalar_element)
    vector_function_space = ufl.FunctionSpace(domain, vector_element)

    u = ufl.TrialFunction(scalar_function_space)
    v = ufl.TestFunction(scalar_function_space)
    s1 = ufl4rom.utils.NamedCoefficient("solution of parametrized problem 1, vector", vector_function_space)

    a21 = ufl.inner(ufl.grad(s1[0]), ufl.grad(u)) * v * ufl.dx + s1[1].dx(0) * u.dx(0) * v * ufl.dx
    assert ufl4rom.utils.name(a21) == "1be5d577013feea78851ada47acf52f1a69a4e73"


def test_name_scalar_22() -> None:
    """Test a form similar to form 21, but where each term is multiplied to the gradient of a function.

    In contrast to form 21, the function is not the solution of a parametrized problem.
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

    u = ufl.TrialFunction(scalar_function_space)
    v = ufl.TestFunction(scalar_function_space)
    k1 = ufl4rom.utils.NamedCoefficient("auxiliary known function 1, vector", vector_function_space)

    a22 = ufl.inner(ufl.grad(k1[0]), ufl.grad(u)) * v * ufl.dx + k1[1].dx(0) * u.dx(0) * v * ufl.dx
    assert ufl4rom.utils.name(a22) == "cbed45afa27e9347250dd3c92eb8666482d46342"


def test_name_scalar_23() -> None:
    """Test a form with a division between two expressions, which can be equivalently written as one coefficient."""
    cell = ufl.triangle
    dim = 2
    scalar_element = ufl.finiteelement.FiniteElement(
        "Lagrange", cell, 1, (), ufl.pullback.identity_pullback, ufl.sobolevspace.H1)
    vector_element = ufl.finiteelement.FiniteElement(
        "Lagrange", cell, 1, (dim, ), ufl.pullback.identity_pullback, ufl.sobolevspace.H1)

    domain = ufl.Mesh(vector_element)
    scalar_function_space = ufl.FunctionSpace(domain, scalar_element)

    u = ufl.TrialFunction(scalar_function_space)
    v = ufl.TestFunction(scalar_function_space)
    f1 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 1", scalar_function_space)
    f2 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 2", scalar_function_space)

    a23 = f2 / f1 * u * v * ufl.dx
    assert ufl4rom.utils.name(a23) == "6edcc91d2e79224479722f406a7e19fa585e8779"


def test_name_scalar_24() -> None:
    """Test a form with division between two expressions, which cannot be written as one coefficient.

    This is in contrast to form 24.
    """
    cell = ufl.triangle
    dim = 2
    scalar_element = ufl.finiteelement.FiniteElement(
        "Lagrange", cell, 1, (), ufl.pullback.identity_pullback, ufl.sobolevspace.H1)
    vector_element = ufl.finiteelement.FiniteElement(
        "Lagrange", cell, 1, (dim, ), ufl.pullback.identity_pullback, ufl.sobolevspace.H1)

    domain = ufl.Mesh(vector_element)
    scalar_function_space = ufl.FunctionSpace(domain, scalar_element)

    u = ufl.TrialFunction(scalar_function_space)
    v = ufl.TestFunction(scalar_function_space)
    f1 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 1", scalar_function_space)
    f2 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 2", scalar_function_space)

    a24 = f2 * u * v / f1 * ufl.dx
    assert ufl4rom.utils.name(a24) == "d6892e8c6b2a2e49c0b784a1374f3c2f4a9704c5"


def test_name_scalar_failure_coefficient() -> None:
    """Test a variation of form 1 that will fail due to not having used named coefficients."""
    cell = ufl.triangle
    dim = 2
    scalar_element = ufl.finiteelement.FiniteElement(
        "Lagrange", cell, 1, (), ufl.pullback.identity_pullback, ufl.sobolevspace.H1)
    vector_element = ufl.finiteelement.FiniteElement(
        "Lagrange", cell, 1, (dim, ), ufl.pullback.identity_pullback, ufl.sobolevspace.H1)

    domain = ufl.Mesh(vector_element)
    scalar_function_space = ufl.FunctionSpace(domain, scalar_element)

    u = ufl.TrialFunction(scalar_function_space)
    v = ufl.TestFunction(scalar_function_space)
    f1 = ufl.Coefficient(scalar_function_space)
    f2 = ufl.Coefficient(scalar_function_space)
    f3 = ufl.Coefficient(scalar_function_space)

    a1 = f3 * f2 * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + f2 * u.dx(0) * v * ufl.dx + f1 * u * v * ufl.dx
    with pytest.raises(RuntimeError) as excinfo:
        ufl4rom.utils.name(a1)
    assert (
        str(excinfo.value) == "The case of plain UFL coefficients is not handled, because its name cannot be changed")


def test_name_scalar_failure_constant() -> None:
    """Test a variation of form 13 that will fail due to not have used named constants."""
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

    u = ufl.TrialFunction(scalar_function_space)
    v = ufl.TestFunction(scalar_function_space)
    f1 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 1, scalar", scalar_function_space)
    f2 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 2, vector", vector_function_space)
    f3 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 3, tensor", tensor_function_space)
    c1 = ufl.Constant(domain, shape=())
    c2 = ufl.Constant(domain, shape=(dim, dim))

    a13 = (
        ufl.inner(c2 * f3 * c1 * ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.inner(c1 * f2, ufl.grad(u)) * v * ufl.dx
        + c1 * f1 * u * v * ufl.dx
    )
    with pytest.raises(RuntimeError) as excinfo:
        ufl4rom.utils.name(a13)
    assert (
        str(excinfo.value) == "The case of plain UFL constants is not handled, because its value cannot be extracted")


def test_name_scalar_debug() -> None:
    """Test again form 1 with the additional debug option."""
    cell = ufl.triangle
    dim = 2
    scalar_element = ufl.finiteelement.FiniteElement(
        "Lagrange", cell, 1, (), ufl.pullback.identity_pullback, ufl.sobolevspace.H1)
    vector_element = ufl.finiteelement.FiniteElement(
        "Lagrange", cell, 1, (dim, ), ufl.pullback.identity_pullback, ufl.sobolevspace.H1)

    domain = ufl.Mesh(vector_element)
    scalar_function_space = ufl.FunctionSpace(domain, scalar_element)

    u = ufl.TrialFunction(scalar_function_space)
    v = ufl.TestFunction(scalar_function_space)
    f1 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 1", scalar_function_space)
    f2 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 2", scalar_function_space)
    f3 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 3", scalar_function_space)

    a1 = f3 * f2 * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + f2 * u.dx(0) * v * ufl.dx + f1 * u * v * ufl.dx
    assert ufl4rom.utils.name(a1, debug=True) == "29a2c37955642711c70c20aa631185f5cd47f041"


def test_name_scalar_expression() -> None:
    """Test the first addend of form 1 to cover name computation of an UFL expression (rather than a form)."""
    cell = ufl.triangle
    dim = 2
    scalar_element = ufl.finiteelement.FiniteElement(
        "Lagrange", cell, 1, (), ufl.pullback.identity_pullback, ufl.sobolevspace.H1)
    vector_element = ufl.finiteelement.FiniteElement(
        "Lagrange", cell, 1, (dim, ), ufl.pullback.identity_pullback, ufl.sobolevspace.H1)

    domain = ufl.Mesh(vector_element)
    scalar_function_space = ufl.FunctionSpace(domain, scalar_element)

    u = ufl.TrialFunction(scalar_function_space)
    v = ufl.TestFunction(scalar_function_space)
    f2 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 2", scalar_function_space)
    f3 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 3", scalar_function_space)

    e1 = f3 * f2 * ufl.inner(ufl.grad(u), ufl.grad(v))
    assert ufl4rom.utils.name(e1) == "b3262702253f9e8164dd26faf13240412f774d45"
