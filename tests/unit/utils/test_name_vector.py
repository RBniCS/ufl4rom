# Copyright (C) 2021-2023 by the ufl4rom authors
#
# This file is part of ufl4rom.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Unit tests for ufl4rom.utils.name module with forms on a scalar finite element."""

import ufl

import ufl4rom.utils


def test_name_vector_1() -> None:
    """Test a basic vector advection-diffusion-reaction parametrized form, with all parametrized coefficients."""
    cell = ufl.triangle
    scalar_element = ufl.FiniteElement("Lagrange", cell, 1)
    vector_element = ufl.VectorElement("Lagrange", cell, 1)
    tensor_element = ufl.TensorElement("Lagrange", cell, 1)

    u = ufl.TrialFunction(vector_element)
    v = ufl.TestFunction(vector_element)
    f1 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 1, scalar", scalar_element)
    f2 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 2, vector", vector_element)
    f3 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 3, tensor", tensor_element)

    a1 = (
        ufl.inner(f3 * ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.inner(ufl.grad(u) * f2, v) * ufl.dx
        + f1 * ufl.inner(u, v) * ufl.dx
    )
    assert ufl4rom.utils.name(a1) == "581c5dd1855b4dd8d422d957f4ffe904ee63f639"


def test_name_vector_2() -> None:
    """In this case the diffusivity tensor is given by the product of two expressions."""
    cell = ufl.triangle
    scalar_element = ufl.FiniteElement("Lagrange", cell, 1)
    vector_element = ufl.VectorElement("Lagrange", cell, 1)
    tensor_element = ufl.TensorElement("Lagrange", cell, 1)

    u = ufl.TrialFunction(vector_element)
    v = ufl.TestFunction(vector_element)
    f1 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 1, scalar", scalar_element)
    f2 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 2, vector", vector_element)
    f3 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 3, tensor", tensor_element)
    f4 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 4, tensor", tensor_element)

    a2 = (
        ufl.inner(f3 * f4 * ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.inner(ufl.grad(u) * f2, v) * ufl.dx
        + f1 * ufl.inner(u, v) * ufl.dx
    )
    assert ufl4rom.utils.name(a2) == "4038a6ad1ec8f501ca71ec7948da880ad00fe3ec"


def test_name_vector_3() -> None:
    """We try now with a more complex expression for each coefficient."""
    cell = ufl.triangle
    scalar_element = ufl.FiniteElement("Lagrange", cell, 1)
    vector_element = ufl.VectorElement("Lagrange", cell, 1)
    tensor_element = ufl.TensorElement("Lagrange", cell, 1)

    u = ufl.TrialFunction(vector_element)
    v = ufl.TestFunction(vector_element)
    f1 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 1, scalar", scalar_element)
    f2 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 2, vector", vector_element)
    f3 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 3, tensor", tensor_element)
    f4 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 4, tensor", tensor_element)

    a3 = (
        ufl.inner(ufl.det(f3) * (f4 + f3 * f3) * f1, ufl.grad(v)) * ufl.dx + ufl.inner(ufl.grad(u) * f2, v) * ufl.dx
        + f1 * ufl.inner(u, v) * ufl.dx
    )
    assert ufl4rom.utils.name(a3) == "a3aec1176f4cdc46e4ed748cfbcec56f7079fc33"


def test_name_vector_4() -> None:
    """We add a term depending on the mesh size."""
    cell = ufl.triangle
    scalar_element = ufl.FiniteElement("Lagrange", cell, 1)
    vector_element = ufl.VectorElement("Lagrange", cell, 1)
    tensor_element = ufl.TensorElement("Lagrange", cell, 1)

    u = ufl.TrialFunction(vector_element)
    v = ufl.TestFunction(vector_element)
    f1 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 1, scalar", scalar_element)
    f2 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 2, vector", vector_element)
    f3 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 3, tensor", tensor_element)
    h = ufl.CellDiameter(cell)

    a4 = (
        ufl.inner(f3 * h * ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.inner(ufl.grad(u) * f2 * h, v) * ufl.dx
        + f1 * h * ufl.inner(u, v) * ufl.dx
    )
    assert ufl4rom.utils.name(a4) == "0255588ca95930446239cc88c1a4713beb5f0d05"


def test_name_vector_5() -> None:
    """Starting from form 4, use parenthesis to change the UFL tree."""
    cell = ufl.triangle
    scalar_element = ufl.FiniteElement("Lagrange", cell, 1)
    vector_element = ufl.VectorElement("Lagrange", cell, 1)
    tensor_element = ufl.TensorElement("Lagrange", cell, 1)

    u = ufl.TrialFunction(vector_element)
    v = ufl.TestFunction(vector_element)
    f1 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 1, scalar", scalar_element)
    f2 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 2, vector", vector_element)
    f3 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 3, tensor", tensor_element)
    h = ufl.CellDiameter(cell)

    a5 = (
        ufl.inner((f3 * h) * ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.inner(ufl.grad(u) * (f2 * h), v) * ufl.dx
        + (f1 * h) * ufl.inner(u, v) * ufl.dx
    )
    assert ufl4rom.utils.name(a5) == "3891b3bcead7aa0b0b3d0254f9ac8256db1ef556"


def test_name_vector_6() -> None:
    """We change the coefficients to be non-parametrized."""
    cell = ufl.triangle
    scalar_element = ufl.FiniteElement("Lagrange", cell, 1)
    vector_element = ufl.VectorElement("Lagrange", cell, 1)
    tensor_element = ufl.TensorElement("Lagrange", cell, 1)

    u = ufl.TrialFunction(vector_element)
    v = ufl.TestFunction(vector_element)
    f1 = ufl4rom.utils.NamedCoefficient("non-parametrized coefficient 1, scalar", scalar_element)
    f2 = ufl4rom.utils.NamedCoefficient("non-parametrized coefficient 2, vector", vector_element)
    f3 = ufl4rom.utils.NamedCoefficient("non-parametrized coefficient 3, tensor", tensor_element)

    a6 = (
        ufl.inner(f3 * ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.inner(ufl.grad(u) * f2, v) * ufl.dx
        + f1 * ufl.inner(u, v) * ufl.dx
    )
    assert ufl4rom.utils.name(a6) == "2bc1c041bb6503b697852314ce72409271b89e76"


def test_name_vector_7() -> None:
    """A part of the diffusion coefficient is parametrized, while advection-reaction are not parametrized."""
    cell = ufl.triangle
    scalar_element = ufl.FiniteElement("Lagrange", cell, 1)
    vector_element = ufl.VectorElement("Lagrange", cell, 1)
    tensor_element = ufl.TensorElement("Lagrange", cell, 1)

    u = ufl.TrialFunction(vector_element)
    v = ufl.TestFunction(vector_element)
    f1 = ufl4rom.utils.NamedCoefficient("non-parametrized coefficient 1, scalar", scalar_element)
    f2 = ufl4rom.utils.NamedCoefficient("non-parametrized coefficient 2, vector", vector_element)
    f3 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 3, tensor", tensor_element)
    f4 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 4, tensor", tensor_element)
    f5 = ufl4rom.utils.NamedCoefficient("non-parametrized coefficient 5, tensor", tensor_element)

    a7 = (
        ufl.inner(f5 * (f3 * f4) * ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.inner(ufl.grad(u) * f2, v) * ufl.dx
        + f1 * ufl.inner(u, v) * ufl.dx
    )
    assert ufl4rom.utils.name(a7) == "d30b741b33f1efb02c035dbfb21c39753b5e3660"


def test_name_vector_8() -> None:
    """Test a case similar to form 7, but hwere the order of the matrix multiplication is different."""
    cell = ufl.triangle
    scalar_element = ufl.FiniteElement("Lagrange", cell, 1)
    vector_element = ufl.VectorElement("Lagrange", cell, 1)
    tensor_element = ufl.TensorElement("Lagrange", cell, 1)

    u = ufl.TrialFunction(vector_element)
    v = ufl.TestFunction(vector_element)
    f1 = ufl4rom.utils.NamedCoefficient("non-parametrized coefficient 1, scalar", scalar_element)
    f2 = ufl4rom.utils.NamedCoefficient("non-parametrized coefficient 2, vector", vector_element)
    f3 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 3, tensor", tensor_element)
    f4 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 4, tensor", tensor_element)
    f5 = ufl4rom.utils.NamedCoefficient("non-parametrized coefficient 5, tensor", tensor_element)

    a8 = (
        ufl.inner(f3 * f5 * f4 * ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.inner(ufl.grad(u) * f2, v) * ufl.dx
        + f1 * ufl.inner(u, v) * ufl.dx
    )
    assert ufl4rom.utils.name(a8) == "4eba6d49dd78f635529bfaffc714a40cf214badf"


def test_name_vector_9() -> None:
    """Test a form similar to form 7, with a coefficient replaced by a constant."""
    cell = ufl.triangle
    dim = cell.topological_dimension()
    scalar_element = ufl.FiniteElement("Lagrange", cell, 1)
    vector_element = ufl.VectorElement("Lagrange", cell, 1)
    tensor_element = ufl.TensorElement("Lagrange", cell, 1)

    u = ufl.TrialFunction(vector_element)
    v = ufl.TestFunction(vector_element)
    f1 = ufl4rom.utils.NamedCoefficient("non-parametrized coefficient 1, scalar", scalar_element)
    f2 = ufl4rom.utils.NamedCoefficient("non-parametrized coefficient 2, vector", vector_element)
    f3 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 3, tensor", tensor_element)
    f4 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 4, tensor", tensor_element)
    c1 = ufl4rom.utils.NamedConstant("constant 1, tensor", cell, shape=(dim, dim))

    a9 = (
        ufl.inner(c1 * (f3 * f4) * ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.inner(ufl.grad(u) * f2, v) * ufl.dx
        + f1 * ufl.inner(u, v) * ufl.dx
    )
    assert ufl4rom.utils.name(a9) == "3b3c09fa7d8ef0df3f67fdad5e7d5bbc8d8a93dd"


def test_name_vector_10() -> None:
    """Test a form similar to form 8, but the order of the matrix multiplication is different."""
    cell = ufl.triangle
    dim = cell.topological_dimension()
    scalar_element = ufl.FiniteElement("Lagrange", cell, 1)
    vector_element = ufl.VectorElement("Lagrange", cell, 1)
    tensor_element = ufl.TensorElement("Lagrange", cell, 1)

    u = ufl.TrialFunction(vector_element)
    v = ufl.TestFunction(vector_element)
    f1 = ufl4rom.utils.NamedCoefficient("non-parametrized coefficient 1, scalar", scalar_element)
    f2 = ufl4rom.utils.NamedCoefficient("non-parametrized coefficient 2, vector", vector_element)
    f3 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 3, tensor", tensor_element)
    f4 = ufl4rom.utils.NamedCoefficient("parametrized coefficient 4, tensor", tensor_element)
    c1 = ufl4rom.utils.NamedConstant("constant 1, tensor", cell, shape=(dim, dim))

    a10 = (
        ufl.inner(f3 * c1 * f4 * ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.inner(ufl.grad(u) * f2, v) * ufl.dx
        + f1 * ufl.inner(u, v) * ufl.dx
    )
    assert ufl4rom.utils.name(a10) == "0fecd0e3a8eb31e99c24f80c9334fe2c65478354"


def test_name_vector_11() -> None:
    """Test form similar to form 1, but where each term is multiplied by a Function, which represents the solution of\
    a parametrized problem."""
    cell = ufl.triangle
    scalar_element = ufl.FiniteElement("Lagrange", cell, 1)
    vector_element = ufl.VectorElement("Lagrange", cell, 1)
    tensor_element = ufl.TensorElement("Lagrange", cell, 1)

    u = ufl.TrialFunction(vector_element)
    v = ufl.TestFunction(vector_element)
    s1 = ufl4rom.utils.NamedCoefficient("solution of parametrized problem 1, scalar", scalar_element)
    s2 = ufl4rom.utils.NamedCoefficient("solution of parametrized problem 2, vector", vector_element)
    s3 = ufl4rom.utils.NamedCoefficient("solution of parametrized problem 3, tensor", tensor_element)

    a11 = (
        ufl.inner(s3 * ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.inner(ufl.grad(u) * s2, v) * ufl.dx
        + s1 * ufl.inner(u, v) * ufl.dx
    )
    assert ufl4rom.utils.name(a11) == "e0cb8a6c0033bdd714db00f4c5a4edb29efd332e"


def test_name_vector_12() -> None:
    """Test a form similar to form 11, but where each term is multiplied by a component of a solution of a parametrized\
    problem, resulting in an Indexed coefficient."""
    cell = ufl.triangle
    element = ufl.VectorElement("Lagrange", cell, 1)

    u = ufl.TrialFunction(element)
    v = ufl.TestFunction(element)
    s1 = ufl4rom.utils.NamedCoefficient("solution of parametrized problem 1", element)

    a12 = s1[0] * ufl.inner(u, v) * ufl.dx
    assert ufl4rom.utils.name(a12) == "6a10201510dba42fc699f698ba6a8ec0a91162da"


def test_name_vector_13() -> None:
    """Test a form similar to form 11, but where each term is multiplied by a Function, which does not represent the\
    solution of a parametrized problem."""
    cell = ufl.triangle
    scalar_element = ufl.FiniteElement("Lagrange", cell, 1)
    vector_element = ufl.VectorElement("Lagrange", cell, 1)
    tensor_element = ufl.TensorElement("Lagrange", cell, 1)

    u = ufl.TrialFunction(vector_element)
    v = ufl.TestFunction(vector_element)
    k1 = ufl4rom.utils.NamedCoefficient("auxiliary known function 1, scalar", scalar_element)
    k2 = ufl4rom.utils.NamedCoefficient("auxiliary known function 2, vector", vector_element)
    k3 = ufl4rom.utils.NamedCoefficient("auxiliary known function 3, tensor", tensor_element)

    a13 = (
        ufl.inner(k3 * ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.inner(ufl.grad(u) * k2, v) * ufl.dx
        + k1 * ufl.inner(u, v) * ufl.dx
    )
    assert ufl4rom.utils.name(a13) == "34162988d1d9e4e52b317a40f2f0050bfd0ba149"


def test_name_vector_14() -> None:
    """Test a form similar to form 12, but where each term is multiplied by a component of a Function which does not\
    represent the solution of a parametrized problem."""
    cell = ufl.triangle
    element = ufl.VectorElement("Lagrange", cell, 1)

    u = ufl.TrialFunction(element)
    v = ufl.TestFunction(element)
    k1 = ufl4rom.utils.NamedCoefficient("auxiliary known function 1", element)

    a14 = k1[0] * ufl.inner(u, v) * ufl.dx
    assert ufl4rom.utils.name(a14) == "a58d1b0421353ed49eb621a0f9e707977c866726"


def test_name_vector_15() -> None:
    """Test a form similar to form 11, but where each term is multiplied by the gradientor partial derivative of\
    a Function, which represents the solution of a parametrized problem."""
    cell = ufl.triangle
    scalar_element = ufl.FiniteElement("Lagrange", cell, 1)
    vector_element = ufl.VectorElement("Lagrange", cell, 1)

    u = ufl.TrialFunction(vector_element)
    v = ufl.TestFunction(vector_element)
    s1 = ufl4rom.utils.NamedCoefficient("solution of parametrized problem 1, scalar", scalar_element)
    s2 = ufl4rom.utils.NamedCoefficient("solution of parametrized problem 2, vector", vector_element)

    a15 = (
        ufl.inner(ufl.grad(s2) * ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.inner(ufl.grad(u) * ufl.grad(s1), v) * ufl.dx
        + s1.dx(0) * ufl.inner(u, v) * ufl.dx
    )
    assert ufl4rom.utils.name(a15) == "b461d753beef25ad68b7631d97587900026a8866"


def test_name_vector_16() -> None:
    """Test a form similar to form 12, but where each term is multiplied by the gradient or partial derivative of\
    a component of a solution of a parametrized problem."""
    cell = ufl.triangle
    element = ufl.VectorElement("Lagrange", cell, 1)

    u = ufl.TrialFunction(element)
    v = ufl.TestFunction(element)
    s1 = ufl4rom.utils.NamedCoefficient("solution of parametrized problem 1", element)

    a16 = ufl.inner(ufl.grad(s1[0]), u) * v[0] * ufl.dx + s1[0].dx(0) * ufl.inner(u, v) * ufl.dx
    assert ufl4rom.utils.name(a16) == "47725758e324b297285af1c3952936d1e6ae83fe"


def test_name_vector_17() -> None:
    """Test a form is similar to form 13, but where each term is multiplied by a the gradient or partial derivative of\
    a Function, which does not represent the solution of a parametrized problem."""
    cell = ufl.triangle
    scalar_element = ufl.FiniteElement("Lagrange", cell, 1)
    vector_element = ufl.VectorElement("Lagrange", cell, 1)

    u = ufl.TrialFunction(vector_element)
    v = ufl.TestFunction(vector_element)
    k1 = ufl4rom.utils.NamedCoefficient("auxiliary known function 1, scalar", scalar_element)
    k2 = ufl4rom.utils.NamedCoefficient("auxiliary known function 2, vector", vector_element)

    a17 = (
        ufl.inner(ufl.grad(k2) * ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.inner(ufl.grad(u) * ufl.grad(k1), v) * ufl.dx
        + k1.dx(0) * ufl.inner(u, v) * ufl.dx
    )
    assert ufl4rom.utils.name(a17) == "5e4166e704713f95dbd86eca68246f5ec11b9e9a"


def test_name_vector_18() -> None:
    """Test a form similar to form 14, but where each term is multiplied by the gradient or partial derivative of\
    a component of a Function which does not represent the solution of a parametrized problem."""
    cell = ufl.triangle
    element = ufl.VectorElement("Lagrange", cell, 1)

    u = ufl.TrialFunction(element)
    v = ufl.TestFunction(element)
    k1 = ufl4rom.utils.NamedCoefficient("auxiliary known function 1", element)

    a18 = ufl.inner(ufl.grad(k1[0]), u) * v[0] * ufl.dx + k1[0].dx(0) * ufl.inner(u, v) * ufl.dx
    assert ufl4rom.utils.name(a18) == "025d727f36916897825c120604e944f9193701ee"
