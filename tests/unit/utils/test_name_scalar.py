# Copyright (C) 2021-2022 by the ufl4rom authors
#
# This file is part of ufl4rom.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import pytest
from logging import DEBUG, getLogger
from ufl import (CellDiameter, Coefficient, Constant, ds, dx, FiniteElement, grad, inner, TensorElement,
                 TestFunction, TrialFunction, triangle, VectorElement)
from ufl4rom.test import enable_logging
from ufl4rom.utils import name, NamedCoefficient, NamedConstant

test_logger = getLogger("tests/unit/utils/test_name_scalar.py")
enable_logger = enable_logging({test_logger: DEBUG})


@enable_logger
def test_name_scalar_1():
    cell = triangle
    element = FiniteElement("Lagrange", cell, 1)

    u = TrialFunction(element)
    v = TestFunction(element)
    f1 = NamedCoefficient("parametrized coefficient 1", element)
    f2 = NamedCoefficient("parametrized coefficient 2", element)
    f3 = NamedCoefficient("parametrized coefficient 3", element)

    a1 = f3 * f2 * inner(grad(u), grad(v)) * dx + f2 * u.dx(0) * v * dx + f1 * u * v * dx
    test_logger.log(DEBUG, "*** ###              FORM 1             ### ***")
    test_logger.log(DEBUG, "This is a basic advection-diffusion-reaction parametrized form, with all"
                    + " parametrized coefficients")

    assert name(a1) == "8bfbb685ff8a1bc1671e5e20e67d7622dc8b7f50"


@enable_logger
def test_name_scalar_2():
    cell = triangle
    element = FiniteElement("Lagrange", cell, 1)

    u = TrialFunction(element)
    v = TestFunction(element)
    f1 = NamedCoefficient("parametrized coefficient 1", element)
    f2 = NamedCoefficient("parametrized coefficient 2", element)
    f3 = NamedCoefficient("parametrized coefficient 3", element)

    a2 = inner(f3 * grad(u), f2 * grad(v)) * dx + f2 * u.dx(0) * v * dx + f1 * u * v * dx
    test_logger.log(DEBUG, "*** ###              FORM 2             ### ***")
    test_logger.log(DEBUG, "We move the diffusion coefficient inside the inner product, splitting it into"
                    + " the two arguments of the inner product")

    assert name(a2) == "1c1e6e6bd61e5097981d078bb0b7cc1366d5c7c9"


@enable_logger
def test_name_scalar_3():
    cell = triangle
    element = FiniteElement("Lagrange", cell, 1)

    u = TrialFunction(element)
    v = TestFunction(element)
    f1 = NamedCoefficient("parametrized coefficient 1", element)
    f2 = NamedCoefficient("parametrized coefficient 2", element)
    f3 = NamedCoefficient("parametrized coefficient 3", element)

    a3 = (f3 * f2 * (1 + f1 * f2) * inner(grad(u), grad(v)) * dx
          + f2 * (1 + f2 * f3) * u.dx(0) * v * dx
          + f3 * (1 + f1 * f2) * u * v * dx)
    test_logger.log(DEBUG, "*** ###              FORM 3             ### ***")
    test_logger.log(DEBUG, "This form tests the expansion of a sum of coefficients")

    assert name(a3) == "d79b5c747c1f2007562f260fc4715ba7000e9777"


@enable_logger
def test_name_scalar_4():
    cell = triangle
    scalar_element = FiniteElement("Lagrange", cell, 1)
    vector_element = VectorElement("Lagrange", cell, 1)
    tensor_element = TensorElement("Lagrange", cell, 1)

    u = TrialFunction(scalar_element)
    v = TestFunction(scalar_element)
    f1 = NamedCoefficient("parametrized coefficient 1, scalar", scalar_element)
    f2 = NamedCoefficient("parametrized coefficient 2, scalar", scalar_element)
    f3 = NamedCoefficient("parametrized coefficient 3, scalar", scalar_element)
    f4 = NamedCoefficient("parametrized coefficient 4, vector", vector_element)
    f5 = NamedCoefficient("parametrized coefficient 5, tensor", tensor_element)

    a4 = (inner(f5 * (1 + f1 * f2) * grad(u), grad(v)) * dx + inner(f4, grad(u)) * v * dx
          + f3 * u * v * dx)
    test_logger.log(DEBUG, "*** ###              FORM 4             ### ***")
    test_logger.log(DEBUG, "We use now a diffusivity tensor and a vector convective field, to test if"
                    + " matrix and vector coefficients are correctly detected.")

    assert name(a4) == "4953c3f5e649b2c18f0286f82bd61c78ef771abb"


@enable_logger
def test_name_scalar_5():
    cell = triangle
    element = FiniteElement("Lagrange", cell, 1)

    u = TrialFunction(element)
    v = TestFunction(element)
    f1 = NamedCoefficient("parametrized coefficient 1", element)
    f2 = NamedCoefficient("parametrized coefficient 2", element)
    f3 = NamedCoefficient("parametrized coefficient 3", element)

    a5 = f3 * f2 * inner(grad(u), grad(v)) * ds + f2 * u.dx(0) * v * ds + f1 * u * v * ds
    test_logger.log(DEBUG, "*** ###              FORM 5             ### ***")
    test_logger.log(DEBUG, "We change the integration domain to be the boundary. This is a variation of form 1")

    assert name(a5) == "713e7fe051de0d7af44cff3bbdb0b1788b11d922"


@enable_logger
def test_name_scalar_6():
    cell = triangle
    element = FiniteElement("Lagrange", cell, 1)

    u = TrialFunction(element)
    v = TestFunction(element)
    f1 = NamedCoefficient("parametrized coefficient 1", element)
    h = CellDiameter(cell)

    a6 = f1 * h * u * v * dx
    test_logger.log(DEBUG, "*** ###              FORM 6             ### ***")
    test_logger.log(DEBUG, "We add a term depending on the mesh size.")

    assert name(a6) == "a8a2315d1962ec1ba14054ec2117fa19df01dafd"


@enable_logger
def test_name_scalar_7():
    cell = triangle
    element = FiniteElement("Lagrange", cell, 1)

    u = TrialFunction(element)
    v = TestFunction(element)
    f1 = NamedCoefficient("non-parametrized coefficient 1", element)
    f2 = NamedCoefficient("non-parametrized coefficient 2", element)
    f3 = NamedCoefficient("non-parametrized coefficient 3", element)

    a7 = f3 * f2 * inner(grad(u), grad(v)) * dx + f2 * u.dx(0) * v * dx + f1 * u * v * dx
    test_logger.log(DEBUG, "*** ###              FORM 7             ### ***")
    test_logger.log(DEBUG, "We change the coefficients to be non-parametrized.")

    assert name(a7) == "50b215b50db01484e15f86c9e1f1629ea92eba7d"


@enable_logger
def test_name_scalar_8():
    cell = triangle
    element = FiniteElement("Lagrange", cell, 1)

    u = TrialFunction(element)
    v = TestFunction(element)
    f1 = NamedCoefficient("parametrized coefficient 1", element)
    f2 = NamedCoefficient("non-parametrized coefficient 2", element)
    f3 = NamedCoefficient("non-parametrized coefficient 3", element)

    a8 = f1 * f3 * inner(grad(u), grad(v)) * dx + f2 * u.dx(0) * v * dx + f3 * u * v * dx
    test_logger.log(DEBUG, "*** ###              FORM 8             ### ***")
    test_logger.log(DEBUG, "A part of the diffusion coefficient is parametrized, while advection-reaction are"
                    + " not parametrized.")

    assert name(a8) == "78cf1a3559fd154bccff9fd61fccb8a93d20659a"


@enable_logger
def test_name_scalar_9():
    cell = triangle
    element = FiniteElement("Lagrange", cell, 1)

    u = TrialFunction(element)
    v = TestFunction(element)
    f1 = NamedCoefficient("parametrized coefficient 1", element)
    f2 = NamedCoefficient("parametrized coefficient 2", element)
    f3 = NamedCoefficient("non-parametrized coefficient 3", element)
    f4 = NamedCoefficient("non-parametrized coefficient 4", element)

    a9 = f2 * f4 / f1 * inner(grad(u), grad(v)) * dx + f3 * u.dx(0) * v * dx + f4 * u * v * dx
    test_logger.log(DEBUG, "*** ###              FORM 9             ### ***")
    test_logger.log(DEBUG, "A part of the diffusion coefficient is parametrized, while advection-reaction terms"
                    + " are not parametrized,where the terms are written in a different way when compared to form 8.")

    assert name(a9) == "92edf1d4f5b8d9403df748809f78c10282fff50b"


@enable_logger
def test_name_scalar_10():
    cell = triangle
    element = FiniteElement("Lagrange", cell, 1)

    u = TrialFunction(element)
    v = TestFunction(element)
    f1 = NamedCoefficient("non-parametrized coefficient 1", element)
    h = CellDiameter(cell)

    a10 = f1 * h * u * v * dx
    test_logger.log(DEBUG, "*** ###              FORM 10             ### ***")
    test_logger.log(DEBUG, "Similarly to form 6, we add a term depending on the mesh size multiplied by a"
                    + " non-parametrized coefficient.")

    assert name(a10) == "a740b1feae9229127e87482cf9edd96c908f77ee"


@enable_logger
def test_name_scalar_11():
    cell = triangle
    element = FiniteElement("Lagrange", cell, 1)

    u = TrialFunction(element)
    v = TestFunction(element)
    f1 = NamedCoefficient("parametrized coefficient 1", element)
    f2 = NamedCoefficient("non-parametrized coefficient 2", element)
    h = CellDiameter(cell)

    a11 = f2 * f1 * h * u * v * dx
    test_logger.log(DEBUG, "*** ###              FORM 11             ### ***")
    test_logger.log(DEBUG, "Similarly to form 6, we add a term depending on the mesh size multiplied by"
                    + " the product of parametrized and a non-parametrized coefficients.")

    assert name(a11) == "9957762a9c2a5ee26d264d9d6ecce49b61a8e94d"


@enable_logger
def test_name_scalar_12():
    cell = triangle
    element = FiniteElement("Lagrange", cell, 1)

    u = TrialFunction(element)
    v = TestFunction(element)
    f1 = NamedCoefficient("parametrized coefficient 1", element)
    f2 = NamedCoefficient("non-parametrized coefficient 2", element)
    h = CellDiameter(cell)

    a12 = f1 * h / f2 * u * v * dx
    test_logger.log(DEBUG, "*** ###              FORM 12             ### ***")
    test_logger.log(DEBUG, "We change form 11 with a slightly different coefficient.")

    assert name(a12) == "63b4925ea76e1349e76affbd7a873c0f27927c59"


@enable_logger
def test_name_scalar_13():
    cell = triangle
    dim = cell.topological_dimension()
    scalar_element = FiniteElement("Lagrange", cell, 1)
    vector_element = VectorElement("Lagrange", cell, 1)
    tensor_element = TensorElement("Lagrange", cell, 1)

    u = TrialFunction(scalar_element)
    v = TestFunction(scalar_element)
    f1 = NamedCoefficient("parametrized coefficient 1, scalar", scalar_element)
    f2 = NamedCoefficient("parametrized coefficient 2, vector", vector_element)
    f3 = NamedCoefficient("parametrized coefficient 3, tensor", tensor_element)
    c1 = NamedConstant("constant 1, scalar", cell, shape=())
    c2 = NamedConstant("constant 2, tensor", cell, shape=(dim, dim))

    a13 = (inner(c2 * f3 * c1 * grad(u), grad(v)) * dx + inner(c1 * f2, grad(u)) * v * dx
           + c1 * f1 * u * v * dx)
    test_logger.log(DEBUG, "*** ###              FORM 13             ### ***")
    test_logger.log(DEBUG, "We now introduce constants in the expression.")

    assert name(a13) == "c60c4f77d47990fa8d42de6719f60f54f71b9161"


@enable_logger
def test_name_scalar_14():
    cell = triangle
    element = FiniteElement("Lagrange", cell, 1)

    u = TrialFunction(element)
    v = TestFunction(element)
    f1 = NamedCoefficient("parametrized coefficient 1", element)
    f2 = NamedCoefficient("parametrized coefficient 2", element)
    f3 = NamedCoefficient("parametrized coefficient 3", element)
    s1 = NamedCoefficient("solution of parametrized problem 1", element)

    a14 = (f3 * s1 * inner(grad(u), grad(v)) * dx + s1 / f2 * u.dx(0) * v * dx
           + f1 * s1 * u * v * dx)
    test_logger.log(DEBUG, "*** ###              FORM 14             ### ***")
    test_logger.log(DEBUG, "This form is similar to form 1, but each term is multiplied by a scalar function,"
                    + " which represents the solution of a parametrized problem")

    assert name(a14) == "3c7383020464a2ce0db5752572a30ad5c2d51595"


@enable_logger
def test_name_scalar_15():
    cell = triangle
    scalar_element = FiniteElement("Lagrange", cell, 1)
    vector_element = VectorElement("Lagrange", cell, 1)
    tensor_element = TensorElement("Lagrange", cell, 1)

    u = TrialFunction(scalar_element)
    v = TestFunction(scalar_element)
    s1 = NamedCoefficient("solution of parametrized problem 1, scalar", scalar_element)
    s2 = NamedCoefficient("solution of parametrized problem 2, vector", vector_element)
    s3 = NamedCoefficient("solution of parametrized problem 3, tensor", tensor_element)

    a15 = inner(s3 * grad(u), grad(v)) * dx + inner(s2, grad(u)) * v * dx + s1 * u * v * dx
    test_logger.log(DEBUG, "*** ###              FORM 15             ### ***")
    test_logger.log(DEBUG, "This form is similar to form 4, but each term is replaced by a solution of"
                    + " a parametrized problem, either scalar, vector or tensor shaped")

    assert name(a15) == "9ab5c129fc836d0abc109cdf50e0b29d89db894b"


@enable_logger
def test_name_scalar_16():
    cell = triangle
    scalar_element = FiniteElement("Lagrange", cell, 1)
    vector_element = VectorElement("Lagrange", cell, 1)
    tensor_element = TensorElement("Lagrange", cell, 1)

    u = TrialFunction(scalar_element)
    v = TestFunction(scalar_element)
    f1 = NamedCoefficient("non-parametrized coefficient 1, scalar", scalar_element)
    f2 = NamedCoefficient("non-parametrized coefficient 2, scalar", scalar_element)
    s1 = NamedCoefficient("solution of parametrized problem 1, scalar", scalar_element)
    s2 = NamedCoefficient("solution of parametrized problem 2, vector", vector_element)
    s3 = NamedCoefficient("solution of parametrized problem 3, tensor", tensor_element)

    a16 = inner(f2 * s3 * f1 * grad(u), grad(v)) * dx + inner(s2, grad(u)) * v * dx + s1 * u * v * dx
    test_logger.log(DEBUG, "*** ###              FORM 16             ### ***")
    test_logger.log(DEBUG, "We multiply non-parametrized coefficients (as in form 7) and solutions.")

    assert name(a16) == "53c26d852f7f1b0ee26c282b9021412a71eccec1"


@enable_logger
def test_name_scalar_17():
    cell = triangle
    scalar_element = FiniteElement("Lagrange", cell, 1)
    vector_element = VectorElement("Lagrange", cell, 1)

    u = TrialFunction(scalar_element)
    v = TestFunction(scalar_element)
    s1 = NamedCoefficient("solution of parametrized problem 1, vector", vector_element)

    a17 = s1[0] * u * v * dx + s1[1] * u.dx(0) * v * dx
    test_logger.log(DEBUG, "*** ###              FORM 17             ### ***")
    test_logger.log(DEBUG, "This form is similar to form 15, but each term is multiplied to a component of"
                    + " a solution of a parametrized problem")

    assert name(a17) == "b5a6f31a47e8caee742b2bf7cca6d1c420074d17"


@enable_logger
def test_name_scalar_18():
    cell = triangle
    element = FiniteElement("Lagrange", cell, 1)

    u = TrialFunction(element)
    v = TestFunction(element)
    f1 = NamedCoefficient("parametrized coefficient 1", element)
    f2 = NamedCoefficient("parametrized coefficient 2", element)
    f3 = NamedCoefficient("parametrized coefficient 3", element)
    k1 = NamedCoefficient("auxiliary known function 1", element)

    a18 = (f3 * k1 * inner(grad(u), grad(v)) * dx + f2 * k1 * u.dx(0) * v * dx
           + f1 * k1 * u * v * dx)
    test_logger.log(DEBUG, "*** ###              FORM 18             ### ***")
    test_logger.log(DEBUG, "This form is similar to form 14, but each term is multiplied by a scalar function,"
                    + " which does not represent the solution of a parametrized problem")

    assert name(a18) == "004cb35effb1f37e3027694f38aae079a779ffcd"


@enable_logger
def test_name_scalar_19():
    cell = triangle
    scalar_element = FiniteElement("Lagrange", cell, 1)
    vector_element = VectorElement("Lagrange", cell, 1)
    tensor_element = TensorElement("Lagrange", cell, 1)

    u = TrialFunction(scalar_element)
    v = TestFunction(scalar_element)
    f1 = NamedCoefficient("non-parametrized coefficient 1, scalar", scalar_element)
    f2 = NamedCoefficient("non-parametrized coefficient 2, scalar", scalar_element)
    k1 = NamedCoefficient("auxiliary known function 1, scalar", scalar_element)
    k2 = NamedCoefficient("auxiliary known function 2, vector", vector_element)
    k3 = NamedCoefficient("auxiliary known function 3, tensor", tensor_element)

    a19 = inner(f2 * k3 * f1 * grad(u), grad(v)) * dx + inner(k2, grad(u)) * v * dx + k1 * u * v * dx
    test_logger.log(DEBUG, "*** ###              FORM 19             ### ***")
    test_logger.log(DEBUG, "This form is similar to form 16, but each term is multiplied by a scalar function,"
                    + " which does not represent the solution of a parametrized problem.")

    assert name(a19) == "e10e71362c48af2d291fd2469b57117deee4b933"


@enable_logger
def test_name_scalar_20():
    cell = triangle
    scalar_element = FiniteElement("Lagrange", cell, 1)
    vector_element = VectorElement("Lagrange", cell, 1)

    u = TrialFunction(scalar_element)
    v = TestFunction(scalar_element)
    k1 = NamedCoefficient("auxiliary known function 1, vector", vector_element)

    a20 = k1[0] * u * v * dx + k1[1] * u.dx(0) * v * dx
    test_logger.log(DEBUG, "*** ###              FORM 20             ### ***")
    test_logger.log(DEBUG, "This form is similar to form 17, but each term is multiplied to a component of"
                    + " a function which does not represent the solution of a parametrized problem.")

    assert name(a20) == "e9c389b1a83b8c2811af885889233dcba23393fd"


@enable_logger
def test_name_scalar_21():
    cell = triangle
    scalar_element = FiniteElement("Lagrange", cell, 1)
    vector_element = VectorElement("Lagrange", cell, 1)

    u = TrialFunction(scalar_element)
    v = TestFunction(scalar_element)
    s1 = NamedCoefficient("solution of parametrized problem 1, vector", vector_element)

    a21 = inner(grad(s1[0]), grad(u)) * v * dx + s1[1].dx(0) * u.dx(0) * v * dx
    test_logger.log(DEBUG, "*** ###              FORM 21             ### ***")
    test_logger.log(DEBUG, "This form is similar to form 17, but each term is multiplied to the gradient/"
                    + "partial derivative of a component of a solution of a parametrized problem")

    assert name(a21) == "e0ce8a7b27fb5cfb3d9559698b5d0fc818024e6f"


@enable_logger
def test_name_scalar_22():
    cell = triangle
    scalar_element = FiniteElement("Lagrange", cell, 1)
    vector_element = VectorElement("Lagrange", cell, 1)

    u = TrialFunction(scalar_element)
    v = TestFunction(scalar_element)
    k1 = NamedCoefficient("auxiliary known function 1, vector", vector_element)

    a22 = inner(grad(k1[0]), grad(u)) * v * dx + k1[1].dx(0) * u.dx(0) * v * dx
    test_logger.log(DEBUG, "*** ###              FORM 22             ### ***")
    test_logger.log(DEBUG, "This form is similar to form 21, but each term is multiplied to the gradient/"
                    + "partial derivative of a component of a Function which is not the solution of"
                    + " a parametrized problem.")

    assert name(a22) == "15074edef16fcb8a3752c596466585994fc7cbc4"


@enable_logger
def test_name_scalar_23():
    cell = triangle
    element = FiniteElement("Lagrange", cell, 1)

    u = TrialFunction(element)
    v = TestFunction(element)
    f1 = NamedCoefficient("parametrized coefficient 1", element)
    f2 = NamedCoefficient("parametrized coefficient 2", element)

    a23 = f2 / f1 * u * v * dx
    test_logger.log(DEBUG, "*** ###              FORM 23             ### ***")
    test_logger.log(DEBUG, "This form tests a division between two expressions, which could have been equivalently"
                    + " written as one coefficient")

    assert name(a23) == "9e44d006fccd4c7cacb818da5f10b0e08fa0a211"


@enable_logger
def test_name_scalar_24():
    cell = triangle
    element = FiniteElement("Lagrange", cell, 1)

    u = TrialFunction(element)
    v = TestFunction(element)
    f1 = NamedCoefficient("parametrized coefficient 1", element)
    f2 = NamedCoefficient("parametrized coefficient 2", element)

    a24 = f2 * u * v / f1 * dx
    test_logger.log(DEBUG, "*** ###              FORM 24             ### ***")
    test_logger.log(DEBUG, "This form tests a division between two expressions, which (in contrast to form 24)"
                    + " cannot be written as one coefficient")

    assert name(a24) == "dbd2a2f8ebf20792bd41e99b748592d8587fe918"


@enable_logger
def test_name_scalar_failure_coefficient():
    cell = triangle
    element = FiniteElement("Lagrange", cell, 1)

    u = TrialFunction(element)
    v = TestFunction(element)
    f1 = Coefficient(element)
    f2 = Coefficient(element)
    f3 = Coefficient(element)

    a1 = f3 * f2 * inner(grad(u), grad(v)) * dx + f2 * u.dx(0) * v * dx + f1 * u * v * dx
    test_logger.log(DEBUG, "*** ###              FORM 1 (failure coefficient)            ### ***")
    test_logger.log(DEBUG, "This variation of form 1 will fail due to not having used named coefficients.")

    with pytest.raises(RuntimeError) as excinfo:
        name(a1)
    assert (
        str(excinfo.value) == "The case of plain UFL coefficients is not handled, because its name cannot be changed")


@enable_logger
def test_name_scalar_failure_constant():
    cell = triangle
    dim = cell.topological_dimension()
    scalar_element = FiniteElement("Lagrange", cell, 1)
    vector_element = VectorElement("Lagrange", cell, 1)
    tensor_element = TensorElement("Lagrange", cell, 1)

    u = TrialFunction(scalar_element)
    v = TestFunction(scalar_element)
    f1 = NamedCoefficient("parametrized coefficient 1, scalar", scalar_element)
    f2 = NamedCoefficient("parametrized coefficient 2, vector", vector_element)
    f3 = NamedCoefficient("parametrized coefficient 3, tensor", tensor_element)
    c1 = Constant(cell, shape=())
    c2 = Constant(cell, shape=(dim, dim))

    a13 = (inner(c2 * f3 * c1 * grad(u), grad(v)) * dx + inner(c1 * f2, grad(u)) * v * dx
           + c1 * f1 * u * v * dx)
    test_logger.log(DEBUG, "*** ###              FORM 13             ### ***")
    test_logger.log(DEBUG, "This variation of form 13 will fail due to not have used named constants.")

    with pytest.raises(RuntimeError) as excinfo:
        name(a13)
    assert (
        str(excinfo.value) == "The case of plain UFL constants is not handled, because its value cannot be extracted")
