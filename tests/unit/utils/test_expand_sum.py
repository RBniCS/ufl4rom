# Copyright (C) 2021-2022 by the ufl4rom authors
#
# This file is part of ufl4rom.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Unit tests for ufl4rom.utils.expand_sum module."""

import typing

import pytest
import ufl
import ufl.algorithms.renumbering

import ufl4rom.utils


def test_expand_sum_real_no_sum() -> None:
    """Test ufl4rom.utils.expand_sum when the form actually contains no sum at all."""
    cell = ufl.triangle
    element = ufl.FiniteElement("Lagrange", cell, 1)

    u = ufl.TrialFunction(element)
    v = ufl.TestFunction(element)
    f = ufl.Coefficient(element)

    form_before_expansion = f * u * v * ufl.dx
    expected_form_after_expansion = f * u * v * ufl.dx

    assert ufl4rom.utils.expand_sum(form_before_expansion) == expected_form_after_expansion


def test_expand_sum_real_sum() -> None:
    """Test ufl4rom.utils.expand_sum when the form contains the sum of two real coefficients."""
    cell = ufl.triangle
    element = ufl.FiniteElement("Lagrange", cell, 1)

    u = ufl.TrialFunction(element)
    v = ufl.TestFunction(element)
    f1 = ufl.Coefficient(element)
    f2 = ufl.Coefficient(element)

    form_before_expansion = (f1 + f2) * u * v * ufl.dx
    expected_form_after_expansion = f1 * u * v * ufl.dx + f2 * u * v * ufl.dx

    assert ufl4rom.utils.expand_sum(form_before_expansion) == expected_form_after_expansion


@pytest.mark.parametrize("op", [ufl.grad, ufl.div, ufl.curl, ufl.nabla_grad, ufl.nabla_div])
def test_expand_sum_real_sum_differential_operators(op: typing.Callable) -> None:
    """Test ufl4rom.utils.expand_sum when the form contains a differential operator of the arguments."""
    cell = ufl.triangle
    scalar_element = ufl.FiniteElement("Lagrange", cell, 1)
    vector_element = ufl.VectorElement("Lagrange", cell, 1)

    u = ufl.TrialFunction(vector_element)
    v = ufl.TestFunction(vector_element)
    f = ufl.Coefficient(scalar_element)

    form_before_expansion = f * ufl.inner(op(u[0] + u[1]), op(v[0] + v[1])) * ufl.dx
    expected_form_after_expansion = (
        f * ufl.inner(op(u[0]), op(v[0])) * ufl.dx
        + f * ufl.inner(op(u[1]), op(v[0])) * ufl.dx
        + f * ufl.inner(op(u[0]), op(v[1])) * ufl.dx
        + f * ufl.inner(op(u[1]), op(v[1])) * ufl.dx)

    assert ufl4rom.utils.expand_sum(form_before_expansion) == expected_form_after_expansion


def test_expand_sum_real_sum_measures() -> None:
    """Test ufl4rom.utils.expand_sum when the form contains the sum of measures."""
    cell = ufl.triangle
    element = ufl.FiniteElement("Lagrange", cell, 1)

    u = ufl.TrialFunction(element)
    v = ufl.TestFunction(element)
    f = ufl.Coefficient(element)

    form_before_expansion = f * u * v * (ufl.dx + ufl.ds)
    expected_form_after_expansion = f * u * v * ufl.dx + f * u * v * ufl.ds

    assert ufl4rom.utils.expand_sum(form_before_expansion) == expected_form_after_expansion


def test_expand_sum_vector_real_scalar_coefficients() -> None:
    """Test ufl4rom.utils.expand_sum when the form contains the sum of two vector real-valued coefficients."""
    cell = ufl.triangle
    scalar_element = ufl.FiniteElement("Lagrange", cell, 1)
    vector_element = ufl.VectorElement("Lagrange", cell, 1)

    u = ufl.TrialFunction(vector_element)
    v = ufl.TestFunction(vector_element)
    f1 = ufl.Coefficient(scalar_element)
    f2 = ufl.Coefficient(scalar_element)

    form_before_expansion = (
        ufl.algorithms.renumbering.renumber_indices(ufl.inner((f1 + f2) * u, v) * ufl.dx))
    expected_form_after_expansion = (
        ufl.algorithms.renumbering.renumber_indices(ufl.inner(f1 * u, v) * ufl.dx)
        + ufl.algorithms.renumbering.renumber_indices(ufl.inner(f2 * u, v) * ufl.dx))

    assert ufl4rom.utils.expand_sum(form_before_expansion) == expected_form_after_expansion


def test_expand_sum_vector_real_tensor_coefficients() -> None:
    """Test ufl4rom.utils.expand_sum when the form contains the sum of two tensor real-valued coefficients."""
    cell = ufl.triangle
    vector_element = ufl.VectorElement("Lagrange", cell, 1)
    tensor_element = ufl.TensorElement("Lagrange", cell, 1)

    u = ufl.TrialFunction(vector_element)
    v = ufl.TestFunction(vector_element)
    f1 = ufl.Coefficient(tensor_element)
    f2 = ufl.Coefficient(tensor_element)

    form_before_expansion = (
        ufl.algorithms.renumbering.renumber_indices(ufl.inner((f1 + f2) * u, v) * ufl.dx))
    expected_form_after_expansion = (
        ufl.algorithms.renumbering.renumber_indices(ufl.inner(f1 * u, v) * ufl.dx)
        + ufl.algorithms.renumbering.renumber_indices(ufl.inner(f2 * u, v) * ufl.dx))

    assert ufl4rom.utils.expand_sum(form_before_expansion) == expected_form_after_expansion


def test_expand_sum_vector_real_tensor_coefficients_grad() -> None:
    """Test ufl4rom.utils.expand_sum when the form contains the sum of two tensor real-valued coefficients\
    and arguments defined on a vector finite element space."""
    cell = ufl.triangle
    scalar_element = ufl.FiniteElement("Lagrange", cell, 1)
    tensor_element = ufl.TensorElement("Lagrange", cell, 1)

    u = ufl.TrialFunction(scalar_element)
    v = ufl.TestFunction(scalar_element)
    f1 = ufl.Coefficient(tensor_element)
    f2 = ufl.Coefficient(tensor_element)

    form_before_expansion = (
        ufl.algorithms.renumbering.renumber_indices(ufl.inner((f1 + f2) * ufl.grad(u), ufl.grad(v)) * ufl.dx))
    expected_form_after_expansion = (
        ufl.algorithms.renumbering.renumber_indices(ufl.inner(f1 * ufl.grad(u), ufl.grad(v)) * ufl.dx)
        + ufl.algorithms.renumbering.renumber_indices(ufl.inner(f2 * ufl.grad(u), ufl.grad(v)) * ufl.dx))

    assert ufl4rom.utils.expand_sum(form_before_expansion) == expected_form_after_expansion


def test_expand_sum_vector_real_tensor_coefficients_sum_grad() -> None:
    """Test ufl4rom.utils.expand_sum when the form contains the sum of two tensor real-valued coefficients\
    and the gradient of the sum of arguments components."""
    cell = ufl.triangle
    scalar_element = ufl.VectorElement("Lagrange", cell, 1)
    tensor_element = ufl.TensorElement("Lagrange", cell, 1)

    u = ufl.TrialFunction(scalar_element)
    v = ufl.TestFunction(scalar_element)
    f1 = ufl.Coefficient(tensor_element)
    f2 = ufl.Coefficient(tensor_element)

    form_before_expansion = (
        ufl.algorithms.renumbering.renumber_indices(
            ufl.inner((f1 + f2) * ufl.grad(u[0] + u[1]), ufl.grad(v[0] - v[1])) * ufl.dx))
    expected_form_after_expansion = (
        ufl.algorithms.renumbering.renumber_indices(ufl.inner(f2 * ufl.grad(u[1]), ufl.grad(v[0])) * ufl.dx)
        + ufl.algorithms.renumbering.renumber_indices(ufl.inner(f2 * ufl.grad(u[1]), ufl.grad(- v[1])) * ufl.dx)
        + ufl.algorithms.renumbering.renumber_indices(ufl.inner(f2 * ufl.grad(u[0]), ufl.grad(v[0])) * ufl.dx)
        + ufl.algorithms.renumbering.renumber_indices(ufl.inner(f2 * ufl.grad(u[0]), ufl.grad(- v[1])) * ufl.dx)
        + ufl.algorithms.renumbering.renumber_indices(ufl.inner(f1 * ufl.grad(u[0]), ufl.grad(v[0])) * ufl.dx)
        + ufl.algorithms.renumbering.renumber_indices(ufl.inner(f1 * ufl.grad(u[0]), ufl.grad(- v[1])) * ufl.dx)
        + ufl.algorithms.renumbering.renumber_indices(ufl.inner(f1 * ufl.grad(u[1]), ufl.grad(v[0])) * ufl.dx)
        + ufl.algorithms.renumbering.renumber_indices(ufl.inner(f1 * ufl.grad(u[1]), ufl.grad(- v[1])) * ufl.dx))

    assert ufl4rom.utils.expand_sum(form_before_expansion) == expected_form_after_expansion


def test_expand_sum_mixed_real_scalar_coefficients() -> None:
    """Test ufl4rom.utils.expand_sum with arguments defined on a mixed element."""
    cell = ufl.triangle
    scalar_element = ufl.FiniteElement("Lagrange", cell, 1)
    vector_element = ufl.VectorElement("Lagrange", cell, 1)
    mixed_element = ufl.MixedElement(scalar_element, vector_element)

    u = ufl.TrialFunction(mixed_element)
    v = ufl.TestFunction(mixed_element)
    f1 = ufl.Coefficient(scalar_element)
    f2 = ufl.Coefficient(scalar_element)

    form_before_expansion = (
        ufl.algorithms.renumbering.renumber_indices(ufl.inner((f1 + f2) * u, v) * ufl.dx))
    expected_form_after_expansion = (
        ufl.algorithms.renumbering.renumber_indices(ufl.inner(f1 * u, v) * ufl.dx)
        + ufl.algorithms.renumbering.renumber_indices(ufl.inner(f2 * u, v) * ufl.dx))

    assert ufl4rom.utils.expand_sum(form_before_expansion) == expected_form_after_expansion


def test_expand_sum_mixed_component_real_scalar_coefficients() -> None:
    """Test ufl4rom.utils.expand_sum with components of arguments defined on a mixed element."""
    cell = ufl.triangle
    scalar_element = ufl.FiniteElement("Lagrange", cell, 1)
    vector_element = ufl.VectorElement("Lagrange", cell, 1)
    mixed_element = ufl.MixedElement(scalar_element, vector_element)

    u = ufl.TrialFunction(mixed_element)
    v = ufl.TestFunction(mixed_element)
    f1 = ufl.Coefficient(scalar_element)
    f2 = ufl.Coefficient(scalar_element)

    form_before_expansion = (
        ufl.algorithms.renumbering.renumber_indices(ufl.inner((f1 + f2) * u[1], v[1]) * ufl.dx))
    expected_form_after_expansion = (
        ufl.algorithms.renumbering.renumber_indices(ufl.inner(f1 * u[1], v[1]) * ufl.dx)
        + ufl.algorithms.renumbering.renumber_indices(ufl.inner(f2 * u[1], v[1]) * ufl.dx))

    assert ufl4rom.utils.expand_sum(form_before_expansion) == expected_form_after_expansion


def test_expand_sum_complex_no_sum() -> None:
    """Test ufl4rom.utils.expand_sum with when the form actually contains no sum at all and complex-valued arguments."""
    cell = ufl.triangle
    element = ufl.FiniteElement("Lagrange", cell, 1)

    u = ufl.TrialFunction(element)
    v = ufl.TestFunction(element)
    f = ufl.Coefficient(element)

    form_before_expansion = f * ufl.imag(u) * ufl.real(v) * ufl.dx
    expected_form_after_expansion = f * ufl.imag(u) * ufl.real(v) * ufl.dx

    assert ufl4rom.utils.expand_sum(form_before_expansion) == expected_form_after_expansion


def test_expand_sum_complex_sum() -> None:
    """Test ufl4rom.utils.expand_sum when the form contains the gradient of the sum of two complex-valude\
    arguments components."""
    cell = ufl.triangle
    scalar_element = ufl.FiniteElement("Lagrange", cell, 1)
    vector_element = ufl.VectorElement("Lagrange", cell, 1)

    u = ufl.TrialFunction(vector_element)
    v = ufl.TestFunction(vector_element)
    f = ufl.Coefficient(scalar_element)

    form_before_expansion = f * ufl.imag(u[0] + u[1]) * ufl.real(v[0] + v[1]) * ufl.dx
    expected_form_after_expansion = (
        f * ufl.imag(u[0]) * ufl.real(v[0]) * ufl.dx
        + f * ufl.imag(u[0]) * ufl.real(v[1]) * ufl.dx
        + f * ufl.imag(u[1]) * ufl.real(v[0]) * ufl.dx
        + f * ufl.imag(u[1]) * ufl.real(v[1]) * ufl.dx)

    assert ufl4rom.utils.expand_sum(form_before_expansion) == expected_form_after_expansion
