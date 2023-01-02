# Copyright (C) 2021-2023 by the ufl4rom authors
#
# This file is part of ufl4rom.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Unit tests for ufl4rom.utils.name module with forms on a scalar finite element."""


import pytest
import ufl

import ufl4rom.utils


def test_name_scalar_1() -> None:
    """Test a basic advection-diffusion-reaction parametrized form, with all parametrized coefficients."""
    cell = ufl.triangle
    element = ufl.FiniteElement("Lagrange", cell, 1)

    u = ufl.TrialFunction(element)
    v = ufl.TestFunction(element)
    f1 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 1", element)
    f2 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 2", element)
    f3 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 3", element)

    a1 = f3 * f2 * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + f2 * u.dx(0) * v * ufl.dx + f1 * u * v * ufl.dx
    assert ufl4rom.utils.name(a1) == "8bfbb685ff8a1bc1671e5e20e67d7622dc8b7f50"


def test_name_scalar_2() -> None:
    """We move the diffusion coefficient inside the ufl.inner product, splitting it into the two arguments of\
    the ufl.inner product."""
    cell = ufl.triangle
    element = ufl.FiniteElement("Lagrange", cell, 1)

    u = ufl.TrialFunction(element)
    v = ufl.TestFunction(element)
    f1 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 1", element)
    f2 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 2", element)
    f3 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 3", element)

    a2 = ufl.inner(f3 * ufl.grad(u), f2 * ufl.grad(v)) * ufl.dx + f2 * u.dx(0) * v * ufl.dx + f1 * u * v * ufl.dx
    assert ufl4rom.utils.name(a2) == "1c1e6e6bd61e5097981d078bb0b7cc1366d5c7c9"


def test_name_scalar_3() -> None:
    """Test a form containing a sum of coefficients."""
    cell = ufl.triangle
    element = ufl.FiniteElement("Lagrange", cell, 1)

    u = ufl.TrialFunction(element)
    v = ufl.TestFunction(element)
    f1 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 1", element)
    f2 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 2", element)
    f3 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 3", element)

    a3 = (
        f3 * f2 * (1 + f1 * f2) * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + f2 * (1 + f2 * f3) * u.dx(0) * v * ufl.dx
        + f3 * (1 + f1 * f2) * u * v * ufl.dx
    )
    assert ufl4rom.utils.name(a3) == "d79b5c747c1f2007562f260fc4715ba7000e9777"


def test_name_scalar_4() -> None:
    """We use now a diffusivity tensor and a vector convective field, to test if matrix and vector coefficients are\
    correctly detected."""
    cell = ufl.triangle
    scalar_element = ufl.FiniteElement("Lagrange", cell, 1)
    vector_element = ufl.VectorElement("Lagrange", cell, 1)
    tensor_element = ufl.TensorElement("Lagrange", cell, 1)

    u = ufl.TrialFunction(scalar_element)
    v = ufl.TestFunction(scalar_element)
    f1 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 1, scalar", scalar_element)
    f2 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 2, scalar", scalar_element)
    f3 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 3, scalar", scalar_element)
    f4 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 4, vector", vector_element)
    f5 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 5, tensor", tensor_element)

    a4 = (
        ufl.inner(f5 * (1 + f1 * f2) * ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.inner(f4, ufl.grad(u)) * v * ufl.dx
        + f3 * u * v * ufl.dx
    )
    assert ufl4rom.utils.name(a4) == "4953c3f5e649b2c18f0286f82bd61c78ef771abb"


def test_name_scalar_5() -> None:
    """We change the integration domain to be the boundary. This is a variation of form 1."""
    cell = ufl.triangle
    element = ufl.FiniteElement("Lagrange", cell, 1)

    u = ufl.TrialFunction(element)
    v = ufl.TestFunction(element)
    f1 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 1", element)
    f2 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 2", element)
    f3 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 3", element)

    a5 = f3 * f2 * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.ds + f2 * u.dx(0) * v * ufl.ds + f1 * u * v * ufl.ds
    assert ufl4rom.utils.name(a5) == "713e7fe051de0d7af44cff3bbdb0b1788b11d922"


def test_name_scalar_6() -> None:
    """We add a term depending on the mesh size."""
    cell = ufl.triangle
    element = ufl.FiniteElement("Lagrange", cell, 1)

    u = ufl.TrialFunction(element)
    v = ufl.TestFunction(element)
    f1 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 1", element)
    h = ufl.CellDiameter(cell)

    a6 = f1 * h * u * v * ufl.dx
    assert ufl4rom.utils.name(a6) == "a8a2315d1962ec1ba14054ec2117fa19df01dafd"


def test_name_scalar_7() -> None:
    """We change the coefficients to be non-parametrized."""
    cell = ufl.triangle
    element = ufl.FiniteElement("Lagrange", cell, 1)

    u = ufl.TrialFunction(element)
    v = ufl.TestFunction(element)
    f1 = ufl4rom.utils.NamedCoefficient("non-parametrized coefficient 1", element)
    f2 = ufl4rom.utils.NamedCoefficient("non-parametrized coefficient 2", element)
    f3 = ufl4rom.utils.NamedCoefficient("non-parametrized coefficient 3", element)

    a7 = f3 * f2 * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + f2 * u.dx(0) * v * ufl.dx + f1 * u * v * ufl.dx
    assert ufl4rom.utils.name(a7) == "50b215b50db01484e15f86c9e1f1629ea92eba7d"


def test_name_scalar_8() -> None:
    """A part of the diffusion coefficient is parametrized, while advection-reaction are not parametrized."""
    cell = ufl.triangle
    element = ufl.FiniteElement("Lagrange", cell, 1)

    u = ufl.TrialFunction(element)
    v = ufl.TestFunction(element)
    f1 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 1", element)
    f2 = ufl4rom.utils.NamedCoefficient("non-parametrized coefficient 2", element)
    f3 = ufl4rom.utils.NamedCoefficient("non-parametrized coefficient 3", element)

    a8 = f1 * f3 * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + f2 * u.dx(0) * v * ufl.dx + f3 * u * v * ufl.dx
    assert ufl4rom.utils.name(a8) == "78cf1a3559fd154bccff9fd61fccb8a93d20659a"


def test_name_scalar_9() -> None:
    """A part of the diffusion coefficient is parametrized, while advection-reaction terms are not parametrized,\
    where the terms are written in a different way when compared to form 8."""
    cell = ufl.triangle
    element = ufl.FiniteElement("Lagrange", cell, 1)

    u = ufl.TrialFunction(element)
    v = ufl.TestFunction(element)
    f1 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 1", element)
    f2 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 2", element)
    f3 = ufl4rom.utils.NamedCoefficient("non-parametrized coefficient 3", element)
    f4 = ufl4rom.utils.NamedCoefficient("non-parametrized coefficient 4", element)

    a9 = (
        f2 * f4 / f1 * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + f3 * u.dx(0) * v * ufl.dx
        + f4 * u * v * ufl.dx
    )
    assert ufl4rom.utils.name(a9) == "92edf1d4f5b8d9403df748809f78c10282fff50b"


def test_name_scalar_10() -> None:
    """Similarly to form 6, we add a term depending on the mesh size multiplied by a non-parametrized coefficient."""
    cell = ufl.triangle
    element = ufl.FiniteElement("Lagrange", cell, 1)

    u = ufl.TrialFunction(element)
    v = ufl.TestFunction(element)
    f1 = ufl4rom.utils.NamedCoefficient("non-parametrized coefficient 1", element)
    h = ufl.CellDiameter(cell)

    a10 = f1 * h * u * v * ufl.dx
    assert ufl4rom.utils.name(a10) == "a740b1feae9229127e87482cf9edd96c908f77ee"


def test_name_scalar_11() -> None:
    """Similarly to form 6, we add a term depending on the mesh size multiplied by the product of parametrized\
    and a non-parametrized coefficients."""
    cell = ufl.triangle
    element = ufl.FiniteElement("Lagrange", cell, 1)

    u = ufl.TrialFunction(element)
    v = ufl.TestFunction(element)
    f1 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 1", element)
    f2 = ufl4rom.utils.NamedCoefficient("non-parametrized coefficient 2", element)
    h = ufl.CellDiameter(cell)

    a11 = f2 * f1 * h * u * v * ufl.dx
    assert ufl4rom.utils.name(a11) == "9957762a9c2a5ee26d264d9d6ecce49b61a8e94d"


def test_name_scalar_12() -> None:
    """We change form 11 with a slightly different coefficient."""
    cell = ufl.triangle
    element = ufl.FiniteElement("Lagrange", cell, 1)

    u = ufl.TrialFunction(element)
    v = ufl.TestFunction(element)
    f1 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 1", element)
    f2 = ufl4rom.utils.NamedCoefficient("non-parametrized coefficient 2", element)
    h = ufl.CellDiameter(cell)

    a12 = f1 * h / f2 * u * v * ufl.dx
    assert ufl4rom.utils.name(a12) == "63b4925ea76e1349e76affbd7a873c0f27927c59"


def test_name_scalar_13() -> None:
    """We now introduce constants in the expression."""
    cell = ufl.triangle
    dim = cell.topological_dimension()
    scalar_element = ufl.FiniteElement("Lagrange", cell, 1)
    vector_element = ufl.VectorElement("Lagrange", cell, 1)
    tensor_element = ufl.TensorElement("Lagrange", cell, 1)

    u = ufl.TrialFunction(scalar_element)
    v = ufl.TestFunction(scalar_element)
    f1 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 1, scalar", scalar_element)
    f2 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 2, vector", vector_element)
    f3 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 3, tensor", tensor_element)
    c1 = ufl4rom.utils.NamedConstant("constant 1, scalar", cell, shape=())
    c2 = ufl4rom.utils.NamedConstant("constant 2, tensor", cell, shape=(dim, dim))

    a13 = (
        ufl.inner(c2 * f3 * c1 * ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.inner(c1 * f2, ufl.grad(u)) * v * ufl.dx
        + c1 * f1 * u * v * ufl.dx
    )
    assert ufl4rom.utils.name(a13) == "c60c4f77d47990fa8d42de6719f60f54f71b9161"


def test_name_scalar_14() -> None:
    """Test a form similar to form 1, but where each term is multiplied by a scalar function, which represents\
    the solution of a parametrized problem."""
    cell = ufl.triangle
    element = ufl.FiniteElement("Lagrange", cell, 1)

    u = ufl.TrialFunction(element)
    v = ufl.TestFunction(element)
    f1 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 1", element)
    f2 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 2", element)
    f3 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 3", element)
    s1 = ufl4rom.utils.NamedCoefficient("solution of parametrized problem 1", element)

    a14 = (
        f3 * s1 * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + s1 / f2 * u.dx(0) * v * ufl.dx
        + f1 * s1 * u * v * ufl.dx
    )
    assert ufl4rom.utils.name(a14) == "3c7383020464a2ce0db5752572a30ad5c2d51595"


def test_name_scalar_15() -> None:
    """Test a form similar to form 4, but where each term is replaced by a solution of a parametrized problem,\
    either scalar, vector or tensor shaped."""
    cell = ufl.triangle
    scalar_element = ufl.FiniteElement("Lagrange", cell, 1)
    vector_element = ufl.VectorElement("Lagrange", cell, 1)
    tensor_element = ufl.TensorElement("Lagrange", cell, 1)

    u = ufl.TrialFunction(scalar_element)
    v = ufl.TestFunction(scalar_element)
    s1 = ufl4rom.utils.NamedCoefficient("solution of parametrized problem 1, scalar", scalar_element)
    s2 = ufl4rom.utils.NamedCoefficient("solution of parametrized problem 2, vector", vector_element)
    s3 = ufl4rom.utils.NamedCoefficient("solution of parametrized problem 3, tensor", tensor_element)

    a15 = (
        ufl.inner(s3 * ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.inner(s2, ufl.grad(u)) * v * ufl.dx
        + s1 * u * v * ufl.dx
    )
    assert ufl4rom.utils.name(a15) == "9ab5c129fc836d0abc109cdf50e0b29d89db894b"


def test_name_scalar_16() -> None:
    """We multiply non-parametrized coefficients (as in form 7) and solutions."""
    cell = ufl.triangle
    scalar_element = ufl.FiniteElement("Lagrange", cell, 1)
    vector_element = ufl.VectorElement("Lagrange", cell, 1)
    tensor_element = ufl.TensorElement("Lagrange", cell, 1)

    u = ufl.TrialFunction(scalar_element)
    v = ufl.TestFunction(scalar_element)
    f1 = ufl4rom.utils.NamedCoefficient("non-parametrized coefficient 1, scalar", scalar_element)
    f2 = ufl4rom.utils.NamedCoefficient("non-parametrized coefficient 2, scalar", scalar_element)
    s1 = ufl4rom.utils.NamedCoefficient("solution of parametrized problem 1, scalar", scalar_element)
    s2 = ufl4rom.utils.NamedCoefficient("solution of parametrized problem 2, vector", vector_element)
    s3 = ufl4rom.utils.NamedCoefficient("solution of parametrized problem 3, tensor", tensor_element)

    a16 = (
        ufl.inner(f2 * s3 * f1 * ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.inner(s2, ufl.grad(u)) * v * ufl.dx
        + s1 * u * v * ufl.dx
    )
    assert ufl4rom.utils.name(a16) == "53c26d852f7f1b0ee26c282b9021412a71eccec1"


def test_name_scalar_17() -> None:
    """Test a form similar to form 15, but where each term is multiplied to a component of a solution of a\
    parametrized problem."""
    cell = ufl.triangle
    scalar_element = ufl.FiniteElement("Lagrange", cell, 1)
    vector_element = ufl.VectorElement("Lagrange", cell, 1)

    u = ufl.TrialFunction(scalar_element)
    v = ufl.TestFunction(scalar_element)
    s1 = ufl4rom.utils.NamedCoefficient("solution of parametrized problem 1, vector", vector_element)

    a17 = s1[0] * u * v * ufl.dx + s1[1] * u.dx(0) * v * ufl.dx
    assert ufl4rom.utils.name(a17) == "b5a6f31a47e8caee742b2bf7cca6d1c420074d17"


def test_name_scalar_18() -> None:
    """Test a form similar to form 14, but where each term is multiplied by a scalar function, which does not\
    represent the solution of a parametrized problem."""
    cell = ufl.triangle
    element = ufl.FiniteElement("Lagrange", cell, 1)

    u = ufl.TrialFunction(element)
    v = ufl.TestFunction(element)
    f1 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 1", element)
    f2 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 2", element)
    f3 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 3", element)
    k1 = ufl4rom.utils.NamedCoefficient("auxiliary known function 1", element)

    a18 = (
        f3 * k1 * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + f2 * k1 * u.dx(0) * v * ufl.dx
        + f1 * k1 * u * v * ufl.dx
    )
    assert ufl4rom.utils.name(a18) == "004cb35effb1f37e3027694f38aae079a779ffcd"


def test_name_scalar_19() -> None:
    """Test a form similar to form 16, but where each term is multiplied by a scalar function, which does not\
    represent the solution of a parametrized problem."""
    cell = ufl.triangle
    scalar_element = ufl.FiniteElement("Lagrange", cell, 1)
    vector_element = ufl.VectorElement("Lagrange", cell, 1)
    tensor_element = ufl.TensorElement("Lagrange", cell, 1)

    u = ufl.TrialFunction(scalar_element)
    v = ufl.TestFunction(scalar_element)
    f1 = ufl4rom.utils.NamedCoefficient("non-parametrized coefficient 1, scalar", scalar_element)
    f2 = ufl4rom.utils.NamedCoefficient("non-parametrized coefficient 2, scalar", scalar_element)
    k1 = ufl4rom.utils.NamedCoefficient("auxiliary known function 1, scalar", scalar_element)
    k2 = ufl4rom.utils.NamedCoefficient("auxiliary known function 2, vector", vector_element)
    k3 = ufl4rom.utils.NamedCoefficient("auxiliary known function 3, tensor", tensor_element)

    a19 = (
        ufl.inner(f2 * k3 * f1 * ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.inner(k2, ufl.grad(u)) * v * ufl.dx
        + k1 * u * v * ufl.dx
    )
    assert ufl4rom.utils.name(a19) == "e10e71362c48af2d291fd2469b57117deee4b933"


def test_name_scalar_20() -> None:
    """Test a form similar to form 17, but where each term is multiplied to a component of a function which does not\
    represent the solution of a parametrized problem."""
    cell = ufl.triangle
    scalar_element = ufl.FiniteElement("Lagrange", cell, 1)
    vector_element = ufl.VectorElement("Lagrange", cell, 1)

    u = ufl.TrialFunction(scalar_element)
    v = ufl.TestFunction(scalar_element)
    k1 = ufl4rom.utils.NamedCoefficient("auxiliary known function 1, vector", vector_element)

    a20 = k1[0] * u * v * ufl.dx + k1[1] * u.dx(0) * v * ufl.dx
    assert ufl4rom.utils.name(a20) == "e9c389b1a83b8c2811af885889233dcba23393fd"


def test_name_scalar_21() -> None:
    """Test a form similar to form 17, but where each term is multiplied to the gradient or partial derivative of\
    a component of a solution of a parametrized problem."""
    cell = ufl.triangle
    scalar_element = ufl.FiniteElement("Lagrange", cell, 1)
    vector_element = ufl.VectorElement("Lagrange", cell, 1)

    u = ufl.TrialFunction(scalar_element)
    v = ufl.TestFunction(scalar_element)
    s1 = ufl4rom.utils.NamedCoefficient("solution of parametrized problem 1, vector", vector_element)

    a21 = ufl.inner(ufl.grad(s1[0]), ufl.grad(u)) * v * ufl.dx + s1[1].dx(0) * u.dx(0) * v * ufl.dx
    assert ufl4rom.utils.name(a21) == "e0ce8a7b27fb5cfb3d9559698b5d0fc818024e6f"


def test_name_scalar_22() -> None:
    """Test a form similar to form 21, but where each term is multiplied to the gradient or partial derivative of\
    a component of a Function which is not the solution of a parametrized problem."""
    cell = ufl.triangle
    scalar_element = ufl.FiniteElement("Lagrange", cell, 1)
    vector_element = ufl.VectorElement("Lagrange", cell, 1)

    u = ufl.TrialFunction(scalar_element)
    v = ufl.TestFunction(scalar_element)
    k1 = ufl4rom.utils.NamedCoefficient("auxiliary known function 1, vector", vector_element)

    a22 = ufl.inner(ufl.grad(k1[0]), ufl.grad(u)) * v * ufl.dx + k1[1].dx(0) * u.dx(0) * v * ufl.dx
    assert ufl4rom.utils.name(a22) == "15074edef16fcb8a3752c596466585994fc7cbc4"


def test_name_scalar_23() -> None:
    """Test a form with a division between two expressions, which could have been equivalently written as\
    one coefficient."""
    cell = ufl.triangle
    element = ufl.FiniteElement("Lagrange", cell, 1)

    u = ufl.TrialFunction(element)
    v = ufl.TestFunction(element)
    f1 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 1", element)
    f2 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 2", element)

    a23 = f2 / f1 * u * v * ufl.dx
    assert ufl4rom.utils.name(a23) == "9e44d006fccd4c7cacb818da5f10b0e08fa0a211"


def test_name_scalar_24() -> None:
    """Test a form with division between two expressions, which (in contrast to form 24) cannot be written as\
    one coefficient."""
    cell = ufl.triangle
    element = ufl.FiniteElement("Lagrange", cell, 1)

    u = ufl.TrialFunction(element)
    v = ufl.TestFunction(element)
    f1 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 1", element)
    f2 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 2", element)

    a24 = f2 * u * v / f1 * ufl.dx
    assert ufl4rom.utils.name(a24) == "dbd2a2f8ebf20792bd41e99b748592d8587fe918"


def test_name_scalar_failure_coefficient() -> None:
    """Test a variation of form 1 that will fail due to not having used named coefficients."""
    cell = ufl.triangle
    element = ufl.FiniteElement("Lagrange", cell, 1)

    u = ufl.TrialFunction(element)
    v = ufl.TestFunction(element)
    f1 = ufl.Coefficient(element)
    f2 = ufl.Coefficient(element)
    f3 = ufl.Coefficient(element)

    a1 = f3 * f2 * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + f2 * u.dx(0) * v * ufl.dx + f1 * u * v * ufl.dx
    with pytest.raises(RuntimeError) as excinfo:
        ufl4rom.utils.name(a1)
    assert (
        str(excinfo.value) == "The case of plain UFL coefficients is not handled, because its name cannot be changed")


def test_name_scalar_failure_constant() -> None:
    """Test a variation of form 13 that will fail due to not have used named constants."""
    cell = ufl.triangle
    dim = cell.topological_dimension()
    scalar_element = ufl.FiniteElement("Lagrange", cell, 1)
    vector_element = ufl.VectorElement("Lagrange", cell, 1)
    tensor_element = ufl.TensorElement("Lagrange", cell, 1)

    u = ufl.TrialFunction(scalar_element)
    v = ufl.TestFunction(scalar_element)
    f1 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 1, scalar", scalar_element)
    f2 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 2, vector", vector_element)
    f3 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 3, tensor", tensor_element)
    c1 = ufl.Constant(cell, shape=())
    c2 = ufl.Constant(cell, shape=(dim, dim))

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
    element = ufl.FiniteElement("Lagrange", cell, 1)

    u = ufl.TrialFunction(element)
    v = ufl.TestFunction(element)
    f1 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 1", element)
    f2 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 2", element)
    f3 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 3", element)

    a1 = f3 * f2 * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + f2 * u.dx(0) * v * ufl.dx + f1 * u * v * ufl.dx
    assert ufl4rom.utils.name(a1, debug=True) == "8bfbb685ff8a1bc1671e5e20e67d7622dc8b7f50"


def test_name_scalar_expression() -> None:
    """Test the first addend of form 1 to cover name computation of an UFL expression (rather than a form)."""
    cell = ufl.triangle
    element = ufl.FiniteElement("Lagrange", cell, 1)

    u = ufl.TrialFunction(element)
    v = ufl.TestFunction(element)
    f2 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 2", element)
    f3 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 3", element)

    e1 = f3 * f2 * ufl.inner(ufl.grad(u), ufl.grad(v))
    assert ufl4rom.utils.name(e1) == "41907fc024afa06d5e8817bdf6e3435db3517f3a"
