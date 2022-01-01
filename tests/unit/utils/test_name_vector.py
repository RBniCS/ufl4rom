# Copyright (C) 2021-2022 by the ufl4rom authors
#
# This file is part of ufl4rom.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from logging import DEBUG, getLogger
from ufl import (CellDiameter, det, dx, FiniteElement, grad, inner, TensorElement, TestFunction, TrialFunction,
                 triangle, VectorElement)
from ufl4rom.test import enable_logging
from ufl4rom.utils import name, NamedCoefficient, NamedConstant

test_logger = getLogger("tests/unit/utils/test_name_vector.py")
enable_logger = enable_logging({test_logger: DEBUG})


@enable_logger
def test_name_vector_1():
    cell = triangle
    scalar_element = FiniteElement("Lagrange", cell, 1)
    vector_element = VectorElement("Lagrange", cell, 1)
    tensor_element = TensorElement("Lagrange", cell, 1)

    u = TrialFunction(vector_element)
    v = TestFunction(vector_element)
    f1 = NamedCoefficient("parametrized coefficient 1, scalar", scalar_element)
    f2 = NamedCoefficient("parametrized coefficient 2, vector", vector_element)
    f3 = NamedCoefficient("parametrized coefficient 3, tensor", tensor_element)

    a1 = inner(f3 * grad(u), grad(v)) * dx + inner(grad(u) * f2, v) * dx + f1 * inner(u, v) * dx
    test_logger.log(DEBUG, "*** ###              FORM 1             ### ***")
    test_logger.log(DEBUG, "This is a basic vector advection-diffusion-reaction parametrized form,"
                    + " with all parametrized coefficients")

    assert name(a1) == "581c5dd1855b4dd8d422d957f4ffe904ee63f639"


@enable_logger
def test_name_vector_2():
    cell = triangle
    scalar_element = FiniteElement("Lagrange", cell, 1)
    vector_element = VectorElement("Lagrange", cell, 1)
    tensor_element = TensorElement("Lagrange", cell, 1)

    u = TrialFunction(vector_element)
    v = TestFunction(vector_element)
    f1 = NamedCoefficient("parametrized coefficient 1, scalar", scalar_element)
    f2 = NamedCoefficient("parametrized coefficient 2, vector", vector_element)
    f3 = NamedCoefficient("parametrized coefficient 3, tensor", tensor_element)
    f4 = NamedCoefficient("parametrized coefficient 4, tensor", tensor_element)

    a2 = inner(f3 * f4 * grad(u), grad(v)) * dx + inner(grad(u) * f2, v) * dx + f1 * inner(u, v) * dx
    test_logger.log(DEBUG, "*** ###              FORM 2             ### ***")
    test_logger.log(DEBUG, "In this case the diffusivity tensor is given by the product of two expressions")

    assert name(a2) == "4038a6ad1ec8f501ca71ec7948da880ad00fe3ec"


@enable_logger
def test_name_vector_3():
    cell = triangle
    scalar_element = FiniteElement("Lagrange", cell, 1)
    vector_element = VectorElement("Lagrange", cell, 1)
    tensor_element = TensorElement("Lagrange", cell, 1)

    u = TrialFunction(vector_element)
    v = TestFunction(vector_element)
    f1 = NamedCoefficient("parametrized coefficient 1, scalar", scalar_element)
    f2 = NamedCoefficient("parametrized coefficient 2, vector", vector_element)
    f3 = NamedCoefficient("parametrized coefficient 3, tensor", tensor_element)
    f4 = NamedCoefficient("parametrized coefficient 4, tensor", tensor_element)

    a3 = inner(det(f3) * (f4 + f3 * f3) * f1, grad(v)) * dx + inner(grad(u) * f2, v) * dx + f1 * inner(u, v) * dx
    test_logger.log(DEBUG, "*** ###              FORM 3             ### ***")
    test_logger.log(DEBUG, "We try now with a more complex expression for each coefficient")

    assert name(a3) == "a3aec1176f4cdc46e4ed748cfbcec56f7079fc33"


@enable_logger
def test_name_vector_4():
    cell = triangle
    scalar_element = FiniteElement("Lagrange", cell, 1)
    vector_element = VectorElement("Lagrange", cell, 1)
    tensor_element = TensorElement("Lagrange", cell, 1)

    u = TrialFunction(vector_element)
    v = TestFunction(vector_element)
    f1 = NamedCoefficient("parametrized coefficient 1, scalar", scalar_element)
    f2 = NamedCoefficient("parametrized coefficient 2, vector", vector_element)
    f3 = NamedCoefficient("parametrized coefficient 3, tensor", tensor_element)
    h = CellDiameter(cell)

    a4 = inner(f3 * h * grad(u), grad(v)) * dx + inner(grad(u) * f2 * h, v) * dx + f1 * h * inner(u, v) * dx
    test_logger.log(DEBUG, "*** ###              FORM 4             ### ***")
    test_logger.log(DEBUG, "We add a term depending on the mesh size.")

    assert name(a4) == "0255588ca95930446239cc88c1a4713beb5f0d05"


@enable_logger
def test_name_vector_5():
    cell = triangle
    scalar_element = FiniteElement("Lagrange", cell, 1)
    vector_element = VectorElement("Lagrange", cell, 1)
    tensor_element = TensorElement("Lagrange", cell, 1)

    u = TrialFunction(vector_element)
    v = TestFunction(vector_element)
    f1 = NamedCoefficient("parametrized coefficient 1, scalar", scalar_element)
    f2 = NamedCoefficient("parametrized coefficient 2, vector", vector_element)
    f3 = NamedCoefficient("parametrized coefficient 3, tensor", tensor_element)
    h = CellDiameter(cell)

    a5 = inner((f3 * h) * grad(u), grad(v)) * dx + inner(grad(u) * (f2 * h), v) * dx + (f1 * h) * inner(u, v) * dx
    test_logger.log(DEBUG, "*** ###              FORM 5             ### ***")
    test_logger.log(DEBUG, "Starting from form 4, use parenthesis to change the UFL tree")

    assert name(a5) == "3891b3bcead7aa0b0b3d0254f9ac8256db1ef556"


@enable_logger
def test_name_vector_6():
    cell = triangle
    scalar_element = FiniteElement("Lagrange", cell, 1)
    vector_element = VectorElement("Lagrange", cell, 1)
    tensor_element = TensorElement("Lagrange", cell, 1)

    u = TrialFunction(vector_element)
    v = TestFunction(vector_element)
    f1 = NamedCoefficient("non-parametrized coefficient 1, scalar", scalar_element)
    f2 = NamedCoefficient("non-parametrized coefficient 2, vector", vector_element)
    f3 = NamedCoefficient("non-parametrized coefficient 3, tensor", tensor_element)

    a6 = inner(f3 * grad(u), grad(v)) * dx + inner(grad(u) * f2, v) * dx + f1 * inner(u, v) * dx
    test_logger.log(DEBUG, "*** ###              FORM 6             ### ***")
    test_logger.log(DEBUG, "We change the coefficients to be non-parametrized.")

    assert name(a6) == "2bc1c041bb6503b697852314ce72409271b89e76"


@enable_logger
def test_name_vector_7():
    cell = triangle
    scalar_element = FiniteElement("Lagrange", cell, 1)
    vector_element = VectorElement("Lagrange", cell, 1)
    tensor_element = TensorElement("Lagrange", cell, 1)

    u = TrialFunction(vector_element)
    v = TestFunction(vector_element)
    f1 = NamedCoefficient("non-parametrized coefficient 1, scalar", scalar_element)
    f2 = NamedCoefficient("non-parametrized coefficient 2, vector", vector_element)
    f3 = NamedCoefficient("parametrized coefficient 3, tensor", tensor_element)
    f4 = NamedCoefficient("parametrized coefficient 4, tensor", tensor_element)
    f5 = NamedCoefficient("non-parametrized coefficient 5, tensor", tensor_element)

    a7 = inner(f5 * (f3 * f4) * grad(u), grad(v)) * dx + inner(grad(u) * f2, v) * dx + f1 * inner(u, v) * dx
    test_logger.log(DEBUG, "*** ###              FORM 7             ### ***")
    test_logger.log(DEBUG, "A part of the diffusion coefficient is parametrized, while advection-reaction are"
                    + " not parametrized.")

    assert name(a7) == "d30b741b33f1efb02c035dbfb21c39753b5e3660"


@enable_logger
def test_name_vector_8():
    cell = triangle
    scalar_element = FiniteElement("Lagrange", cell, 1)
    vector_element = VectorElement("Lagrange", cell, 1)
    tensor_element = TensorElement("Lagrange", cell, 1)

    u = TrialFunction(vector_element)
    v = TestFunction(vector_element)
    f1 = NamedCoefficient("non-parametrized coefficient 1, scalar", scalar_element)
    f2 = NamedCoefficient("non-parametrized coefficient 2, vector", vector_element)
    f3 = NamedCoefficient("parametrized coefficient 3, tensor", tensor_element)
    f4 = NamedCoefficient("parametrized coefficient 4, tensor", tensor_element)
    f5 = NamedCoefficient("non-parametrized coefficient 5, tensor", tensor_element)

    a8 = inner(f3 * f5 * f4 * grad(u), grad(v)) * dx + inner(grad(u) * f2, v) * dx + f1 * inner(u, v) * dx
    test_logger.log(DEBUG, "*** ###              FORM 8             ### ***")
    test_logger.log(DEBUG, "This case is similar to form 7, but the order of the matrix multiplication is different.")

    assert name(a8) == "4eba6d49dd78f635529bfaffc714a40cf214badf"


@enable_logger
def test_name_vector_9():
    cell = triangle
    dim = cell.topological_dimension()
    scalar_element = FiniteElement("Lagrange", cell, 1)
    vector_element = VectorElement("Lagrange", cell, 1)
    tensor_element = TensorElement("Lagrange", cell, 1)

    u = TrialFunction(vector_element)
    v = TestFunction(vector_element)
    f1 = NamedCoefficient("non-parametrized coefficient 1, scalar", scalar_element)
    f2 = NamedCoefficient("non-parametrized coefficient 2, vector", vector_element)
    f3 = NamedCoefficient("parametrized coefficient 3, tensor", tensor_element)
    f4 = NamedCoefficient("parametrized coefficient 4, tensor", tensor_element)
    c1 = NamedConstant("constant 1, tensor", cell, shape=(dim, dim))

    a9 = inner(c1 * (f3 * f4) * grad(u), grad(v)) * dx + inner(grad(u) * f2, v) * dx + f1 * inner(u, v) * dx
    test_logger.log(DEBUG, "*** ###              FORM 9             ### ***")
    test_logger.log(DEBUG, "This is similar to form 7, with a coefficient replaced by a constant")

    assert name(a9) == "3b3c09fa7d8ef0df3f67fdad5e7d5bbc8d8a93dd"


@enable_logger
def test_name_vector_10():
    cell = triangle
    dim = cell.topological_dimension()
    scalar_element = FiniteElement("Lagrange", cell, 1)
    vector_element = VectorElement("Lagrange", cell, 1)
    tensor_element = TensorElement("Lagrange", cell, 1)

    u = TrialFunction(vector_element)
    v = TestFunction(vector_element)
    f1 = NamedCoefficient("non-parametrized coefficient 1, scalar", scalar_element)
    f2 = NamedCoefficient("non-parametrized coefficient 2, vector", vector_element)
    f3 = NamedCoefficient("parametrized coefficient 3, tensor", tensor_element)
    f4 = NamedCoefficient("parametrized coefficient 4, tensor", tensor_element)
    c1 = NamedConstant("constant 1, tensor", cell, shape=(dim, dim))

    a10 = inner(f3 * c1 * f4 * grad(u), grad(v)) * dx + inner(grad(u) * f2, v) * dx + f1 * inner(u, v) * dx
    test_logger.log(DEBUG, "*** ###              FORM 10             ### ***")
    test_logger.log(DEBUG, "This is similar to form 8, but the order of the matrix multiplication is different")

    assert name(a10) == "0dd7da61f92c0048bccecf50652d9b70070401b7"


@enable_logger
def test_name_vector_11():
    cell = triangle
    scalar_element = FiniteElement("Lagrange", cell, 1)
    vector_element = VectorElement("Lagrange", cell, 1)
    tensor_element = TensorElement("Lagrange", cell, 1)

    u = TrialFunction(vector_element)
    v = TestFunction(vector_element)
    s1 = NamedCoefficient("solution of parametrized problem 1, scalar", scalar_element)
    s2 = NamedCoefficient("solution of parametrized problem 2, vector", vector_element)
    s3 = NamedCoefficient("solution of parametrized problem 3, tensor", tensor_element)

    a11 = inner(s3 * grad(u), grad(v)) * dx + inner(grad(u) * s2, v) * dx + s1 * inner(u, v) * dx
    test_logger.log(DEBUG, "*** ###              FORM 11             ### ***")
    test_logger.log(DEBUG, "This form is similar to form 1, but each term is multiplied by a Function,"
                    + " which represents the solution of a parametrized problem")

    assert name(a11) == "e0cb8a6c0033bdd714db00f4c5a4edb29efd332e"


@enable_logger
def test_name_vector_12():
    cell = triangle
    element = VectorElement("Lagrange", cell, 1)

    u = TrialFunction(element)
    v = TestFunction(element)
    s1 = NamedCoefficient("solution of parametrized problem 1", element)

    a12 = s1[0] * inner(u, v) * dx
    test_logger.log(DEBUG, "*** ###              FORM 12             ### ***")
    test_logger.log(DEBUG, "This form is similar to form 11, but each term is multiplied by a component of"
                    + " a solution of a parametrized problem, resulting in an Indexed coefficient")

    assert name(a12) == "6a10201510dba42fc699f698ba6a8ec0a91162da"


@enable_logger
def test_name_vector_13():
    cell = triangle
    scalar_element = FiniteElement("Lagrange", cell, 1)
    vector_element = VectorElement("Lagrange", cell, 1)
    tensor_element = TensorElement("Lagrange", cell, 1)

    u = TrialFunction(vector_element)
    v = TestFunction(vector_element)
    k1 = NamedCoefficient("auxiliary known function 1, scalar", scalar_element)
    k2 = NamedCoefficient("auxiliary known function 2, vector", vector_element)
    k3 = NamedCoefficient("auxiliary known function 3, tensor", tensor_element)

    a13 = inner(k3 * grad(u), grad(v)) * dx + inner(grad(u) * k2, v) * dx + k1 * inner(u, v) * dx
    test_logger.log(DEBUG, "*** ###              FORM 13             ### ***")
    test_logger.log(DEBUG, "This form is similar to form 11, but each term is multiplied by a Function,"
                    + " which does not represent the solution of a parametrized problem.")

    assert name(a13) == "34162988d1d9e4e52b317a40f2f0050bfd0ba149"


@enable_logger
def test_name_vector_14():
    cell = triangle
    element = VectorElement("Lagrange", cell, 1)

    u = TrialFunction(element)
    v = TestFunction(element)
    k1 = NamedCoefficient("auxiliary known function 1", element)

    a14 = k1[0] * inner(u, v) * dx
    test_logger.log(DEBUG, "*** ###              FORM 14             ### ***")
    test_logger.log(DEBUG, "This form is similar to form 12, but each term is multiplied by a component of"
                    + " a Function which does not represent the solution of a parametrized problem.")

    assert name(a14) == "a58d1b0421353ed49eb621a0f9e707977c866726"


@enable_logger
def test_name_vector_15():
    cell = triangle
    scalar_element = FiniteElement("Lagrange", cell, 1)
    vector_element = VectorElement("Lagrange", cell, 1)

    u = TrialFunction(vector_element)
    v = TestFunction(vector_element)
    s1 = NamedCoefficient("solution of parametrized problem 1, scalar", scalar_element)
    s2 = NamedCoefficient("solution of parametrized problem 2, vector", vector_element)

    a15 = (inner(grad(s2) * grad(u), grad(v)) * dx + inner(grad(u) * grad(s1), v) * dx
           + s1.dx(0) * inner(u, v) * dx)
    test_logger.log(DEBUG, "*** ###              FORM 15             ### ***")
    test_logger.log(DEBUG, "This form is similar to form 11, but each term is multiplied by the gradient/"
                    + "partial derivative of a Function, which represents the solution of a parametrized problem")

    assert name(a15) == "b461d753beef25ad68b7631d97587900026a8866"


@enable_logger
def test_name_vector_16():
    cell = triangle
    element = VectorElement("Lagrange", cell, 1)

    u = TrialFunction(element)
    v = TestFunction(element)
    s1 = NamedCoefficient("solution of parametrized problem 1", element)

    a16 = inner(grad(s1[0]), u) * v[0] * dx + s1[0].dx(0) * inner(u, v) * dx
    test_logger.log(DEBUG, "*** ###              FORM 16             ### ***")
    test_logger.log(DEBUG, "This form is similar to form 12, but each term is multiplied by the gradient/"
                    + "partial derivative of a component of a solution of a parametrized problem")

    assert name(a16) == "47725758e324b297285af1c3952936d1e6ae83fe"


@enable_logger
def test_name_vector_17():
    cell = triangle
    scalar_element = FiniteElement("Lagrange", cell, 1)
    vector_element = VectorElement("Lagrange", cell, 1)

    u = TrialFunction(vector_element)
    v = TestFunction(vector_element)
    k1 = NamedCoefficient("auxiliary known function 1, scalar", scalar_element)
    k2 = NamedCoefficient("auxiliary known function 2, vector", vector_element)

    a17 = (inner(grad(k2) * grad(u), grad(v)) * dx + inner(grad(u) * grad(k1), v) * dx
           + k1.dx(0) * inner(u, v) * dx)
    test_logger.log(DEBUG, "*** ###              FORM 17             ### ***")
    test_logger.log(DEBUG, "This form is similar to form 13, but each term is multiplied by a the gradient/"
                    + "partial derivative of Function, which does not represent the solution of a parametrized"
                    + " problem")

    assert name(a17) == "5e4166e704713f95dbd86eca68246f5ec11b9e9a"


@enable_logger
def test_name_vector_18():
    cell = triangle
    element = VectorElement("Lagrange", cell, 1)

    u = TrialFunction(element)
    v = TestFunction(element)
    k1 = NamedCoefficient("auxiliary known function 1", element)

    a18 = (inner(grad(k1[0]), u) * v[0] * dx + k1[0].dx(0) * inner(u, v) * dx)
    test_logger.log(DEBUG, "*** ###              FORM 18             ### ***")
    test_logger.log(DEBUG, "This form is similar to form 14, but each term is multiplied by the gradient/"
                    + "partial derivative of a component of a Function which does not represent the solution"
                    + " of a parametrized problem.")

    assert name(a18) == "025d727f36916897825c120604e944f9193701ee"
