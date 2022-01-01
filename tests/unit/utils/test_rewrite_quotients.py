# Copyright (C) 2021-2022 by the ufl4rom authors
#
# This file is part of ufl4rom.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from ufl import (
    Coefficient, dx, FiniteElement, inner, imag, real, TensorElement, TestFunction, TrialFunction,
    triangle, VectorElement)
from ufl.algorithms.renumbering import renumber_indices
from ufl4rom.utils import rewrite_quotients


def test_rewrite_quotients_real_no_quotient():
    cell = triangle
    element = FiniteElement("Lagrange", cell, 1)

    u = TrialFunction(element)
    v = TestFunction(element)
    f = Coefficient(element)

    form_before_expansion = f * u * v * dx
    expected_form_after_expansion = f * u * v * dx

    assert rewrite_quotients(form_before_expansion) == expected_form_after_expansion


def test_rewrite_quotients_real_one_quotient():
    cell = triangle
    element = FiniteElement("Lagrange", cell, 1)

    u = TrialFunction(element)
    v = TestFunction(element)
    f1 = Coefficient(element)
    f2 = Coefficient(element)

    form_before_expansion = f1 / f2 * u * v * dx
    expected_form_after_expansion = f1 * (1 / f2) * u * v * dx

    assert rewrite_quotients(form_before_expansion) == expected_form_after_expansion


def test_rewrite_quotients_vector_real_scalar_numerator():
    cell = triangle
    scalar_element = FiniteElement("Lagrange", cell, 1)
    vector_element = VectorElement("Lagrange", cell, 1)

    u = TrialFunction(vector_element)
    v = TestFunction(vector_element)
    f1 = Coefficient(scalar_element)
    f2 = Coefficient(scalar_element)

    form_before_expansion = renumber_indices(inner(f1 / f2 * u, v) * dx)
    expected_form_after_expansion = renumber_indices(inner(f1 * (1 / f2) * u, v) * dx)

    assert rewrite_quotients(form_before_expansion) == expected_form_after_expansion


def test_rewrite_quotients_vector_real_tensor_numerator():
    cell = triangle
    scalar_element = FiniteElement("Lagrange", cell, 1)
    vector_element = VectorElement("Lagrange", cell, 1)
    tensor_element = TensorElement("Lagrange", cell, 1)

    u = TrialFunction(vector_element)
    v = TestFunction(vector_element)
    f1 = Coefficient(tensor_element)
    f2 = Coefficient(scalar_element)

    form_before_expansion = renumber_indices(inner(f1 / f2 * u, v) * dx)
    expected_form_after_expansion = renumber_indices(inner(f1 * (1 / f2) * u, v) * dx)

    assert rewrite_quotients(form_before_expansion) == expected_form_after_expansion


def test_rewrite_quotients_complex_no_quotient():
    cell = triangle
    element = FiniteElement("Lagrange", cell, 1)

    u = TrialFunction(element)
    v = TestFunction(element)
    f = Coefficient(element)

    form_before_expansion = f * imag(u) * real(v) * dx
    expected_form_after_expansion = f * imag(u) * real(v) * dx

    assert rewrite_quotients(form_before_expansion) == expected_form_after_expansion


def test_rewrite_quotients_complex_one_quotient():
    cell = triangle
    element = FiniteElement("Lagrange", cell, 1)

    u = TrialFunction(element)
    v = TestFunction(element)
    f1 = Coefficient(element)
    f2 = Coefficient(element)

    form_before_expansion = imag(f1 / f2) * u * v * dx
    expected_form_after_expansion = imag(f1 * (1 / f2)) * u * v * dx

    assert rewrite_quotients(form_before_expansion) == expected_form_after_expansion
