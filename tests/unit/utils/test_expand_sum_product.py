# Copyright (C) 2021 by the ufl4rom authors
#
# This file is part of ufl4rom.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from ufl import (
    Coefficient, ds, dx, FiniteElement, grad, inner, imag, MixedElement, real, TensorElement,
    TestFunction, TrialFunction, triangle, VectorElement)
from ufl.algorithms.renumbering import renumber_indices
from ufl4rom.utils import expand_sum_product


def test_expand_sum_product_real_no_sum():
    cell = triangle
    element = FiniteElement("Lagrange", cell, 1)

    u = TrialFunction(element)
    v = TestFunction(element)
    f = Coefficient(element)

    form_before_expansion = f * u * v * dx
    expected_form_after_expansion = f * u * v * dx

    assert expand_sum_product(form_before_expansion) == expected_form_after_expansion


def test_expand_sum_product_real_sum():
    cell = triangle
    element = FiniteElement("Lagrange", cell, 1)

    u = TrialFunction(element)
    v = TestFunction(element)
    f1 = Coefficient(element)
    f2 = Coefficient(element)

    form_before_expansion = (f1 + f2) * u * v * dx
    expected_form_after_expansion = f1 * u * v * dx + f2 * u * v * dx

    assert expand_sum_product(form_before_expansion) == expected_form_after_expansion


def test_expand_sum_product_real_sum_grad():
    cell = triangle
    scalar_element = FiniteElement("Lagrange", cell, 1)
    vector_element = VectorElement("Lagrange", cell, 1)

    u = TrialFunction(vector_element)
    v = TestFunction(vector_element)
    f = Coefficient(scalar_element)

    form_before_expansion = f * inner(grad(u[0] + u[1]), grad(v[0] + v[1])) * dx
    expected_form_after_expansion = (
        f * inner(grad(u[0]), grad(v[0])) * dx
        + f * inner(grad(u[1]), grad(v[0])) * dx
        + f * inner(grad(u[0]), grad(v[1])) * dx
        + f * inner(grad(u[1]), grad(v[1])) * dx)

    assert expand_sum_product(form_before_expansion) == expected_form_after_expansion


def test_expand_sum_product_real_sum_measures():
    cell = triangle
    element = FiniteElement("Lagrange", cell, 1)

    u = TrialFunction(element)
    v = TestFunction(element)
    f = Coefficient(element)

    form_before_expansion = f * u * v * (dx + ds)
    expected_form_after_expansion = f * u * v * dx + f * u * v * ds

    assert expand_sum_product(form_before_expansion) == expected_form_after_expansion


def test_expand_sum_product_vector_real_scalar_coefficients():
    cell = triangle
    scalar_element = FiniteElement("Lagrange", cell, 1)
    vector_element = VectorElement("Lagrange", cell, 1)

    u = TrialFunction(vector_element)
    v = TestFunction(vector_element)
    f1 = Coefficient(scalar_element)
    f2 = Coefficient(scalar_element)

    form_before_expansion = (
        renumber_indices(inner((f1 + f2) * u, v) * dx))
    expected_form_after_expansion = (
        renumber_indices(inner(f1 * u, v) * dx)
        + renumber_indices(inner(f2 * u, v) * dx))

    assert expand_sum_product(form_before_expansion) == expected_form_after_expansion


def test_expand_sum_product_vector_real_tensor_coefficients():
    cell = triangle
    vector_element = VectorElement("Lagrange", cell, 1)
    tensor_element = TensorElement("Lagrange", cell, 1)

    u = TrialFunction(vector_element)
    v = TestFunction(vector_element)
    f1 = Coefficient(tensor_element)
    f2 = Coefficient(tensor_element)

    form_before_expansion = (
        renumber_indices(inner((f1 + f2) * u, v) * dx))
    expected_form_after_expansion = (
        renumber_indices(inner(f1 * u, v) * dx)
        + renumber_indices(inner(f2 * u, v) * dx))

    assert expand_sum_product(form_before_expansion) == expected_form_after_expansion


def test_expand_sum_product_vector_real_tensor_coefficients_grad():
    cell = triangle
    scalar_element = FiniteElement("Lagrange", cell, 1)
    tensor_element = TensorElement("Lagrange", cell, 1)

    u = TrialFunction(scalar_element)
    v = TestFunction(scalar_element)
    f1 = Coefficient(tensor_element)
    f2 = Coefficient(tensor_element)

    form_before_expansion = (
        renumber_indices(inner((f1 + f2) * grad(u), grad(v)) * dx))
    expected_form_after_expansion = (
        renumber_indices(inner(f1 * grad(u), grad(v)) * dx)
        + renumber_indices(inner(f2 * grad(u), grad(v)) * dx))

    assert expand_sum_product(form_before_expansion) == expected_form_after_expansion


def test_expand_sum_product_vector_real_tensor_coefficients_sum_grad():
    cell = triangle
    scalar_element = VectorElement("Lagrange", cell, 1)
    tensor_element = TensorElement("Lagrange", cell, 1)

    u = TrialFunction(scalar_element)
    v = TestFunction(scalar_element)
    f1 = Coefficient(tensor_element)
    f2 = Coefficient(tensor_element)

    form_before_expansion = (
        renumber_indices(inner((f1 + f2) * grad(u[0] + u[1]), grad(v[0] - v[1])) * dx))
    expected_form_after_expansion = (
        renumber_indices(inner(f2 * grad(u[1]), grad(v[0])) * dx)
        + renumber_indices(inner(f2 * grad(u[1]), grad(- v[1])) * dx)
        + renumber_indices(inner(f2 * grad(u[0]), grad(v[0])) * dx)
        + renumber_indices(inner(f2 * grad(u[0]), grad(- v[1])) * dx)
        + renumber_indices(inner(f1 * grad(u[0]), grad(v[0])) * dx)
        + renumber_indices(inner(f1 * grad(u[0]), grad(- v[1])) * dx)
        + renumber_indices(inner(f1 * grad(u[1]), grad(v[0])) * dx)
        + renumber_indices(inner(f1 * grad(u[1]), grad(- v[1])) * dx))

    assert expand_sum_product(form_before_expansion) == expected_form_after_expansion


def test_expand_sum_product_mixed_real_scalar_coefficients():
    cell = triangle
    scalar_element = FiniteElement("Lagrange", cell, 1)
    vector_element = VectorElement("Lagrange", cell, 1)
    mixed_element = MixedElement(scalar_element, vector_element)

    u = TrialFunction(mixed_element)
    v = TestFunction(mixed_element)
    f1 = Coefficient(scalar_element)
    f2 = Coefficient(scalar_element)

    form_before_expansion = (
        renumber_indices(inner((f1 + f2) * u, v) * dx))
    expected_form_after_expansion = (
        renumber_indices(inner(f1 * u, v) * dx)
        + renumber_indices(inner(f2 * u, v) * dx))

    assert expand_sum_product(form_before_expansion) == expected_form_after_expansion


def test_expand_sum_product_mixed_component_real_scalar_coefficients():
    cell = triangle
    scalar_element = FiniteElement("Lagrange", cell, 1)
    vector_element = VectorElement("Lagrange", cell, 1)
    mixed_element = MixedElement(scalar_element, vector_element)

    u = TrialFunction(mixed_element)
    v = TestFunction(mixed_element)
    f1 = Coefficient(scalar_element)
    f2 = Coefficient(scalar_element)

    form_before_expansion = (
        renumber_indices(inner((f1 + f2) * u[1], v[1]) * dx))
    expected_form_after_expansion = (
        renumber_indices(inner(f1 * u[1], v[1]) * dx)
        + renumber_indices(inner(f2 * u[1], v[1]) * dx))

    assert expand_sum_product(form_before_expansion) == expected_form_after_expansion


def test_expand_sum_product_complex_no_sum():
    cell = triangle
    element = FiniteElement("Lagrange", cell, 1)

    u = TrialFunction(element)
    v = TestFunction(element)
    f = Coefficient(element)

    form_before_expansion = f * imag(u) * real(v) * dx
    expected_form_after_expansion = f * imag(u) * real(v) * dx

    assert expand_sum_product(form_before_expansion) == expected_form_after_expansion


def test_expand_sum_product_complex_sum():
    cell = triangle
    scalar_element = FiniteElement("Lagrange", cell, 1)
    vector_element = VectorElement("Lagrange", cell, 1)

    u = TrialFunction(vector_element)
    v = TestFunction(vector_element)
    f = Coefficient(scalar_element)

    form_before_expansion = f * imag(u[0] + u[1]) * real(v[0] + v[1]) * dx
    expected_form_after_expansion = (
        f * imag(u[0]) * real(v[0]) * dx
        + f * imag(u[0]) * real(v[1]) * dx
        + f * imag(u[1]) * real(v[0]) * dx
        + f * imag(u[1]) * real(v[1]) * dx)

    assert expand_sum_product(form_before_expansion) == expected_form_after_expansion
