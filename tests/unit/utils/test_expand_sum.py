# Copyright (C) 2021-2024 by the ufl4rom authors
#
# This file is part of ufl4rom.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Unit tests for ufl4rom.utils.expand_sum module."""

import typing

import pytest
import ufl
import ufl.algorithms.renumbering
import ufl.finiteelement
import ufl.pullback
import ufl.sobolevspace

import ufl4rom.utils


def test_expand_sum_real_no_sum() -> None:
    """Test ufl4rom.utils.expand_sum when the form actually contains no sum at all."""
    cell = ufl.triangle
    dim = cell.geometric_dimension()
    scalar_element = ufl.finiteelement.FiniteElement(
        "Lagrange", cell, 1, (), ufl.pullback.identity_pullback, ufl.sobolevspace.H1)
    vector_element = ufl.finiteelement.FiniteElement(
        "Lagrange", cell, 1, (dim, ), ufl.pullback.identity_pullback, ufl.sobolevspace.H1)

    domain = ufl.Mesh(vector_element)
    scalar_function_space = ufl.FunctionSpace(domain, scalar_element)

    u = ufl.TrialFunction(scalar_function_space)
    v = ufl.TestFunction(scalar_function_space)
    f = ufl.Coefficient(scalar_function_space)

    form_before_expansion = f * u * v * ufl.dx
    expected_form_after_expansion = f * u * v * ufl.dx

    assert ufl4rom.utils.expand_sum(form_before_expansion) == expected_form_after_expansion


def test_expand_sum_real_sum() -> None:
    """Test ufl4rom.utils.expand_sum when the form contains the sum of two real coefficients."""
    cell = ufl.triangle
    dim = cell.geometric_dimension()
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

    form_before_expansion = (f1 + f2) * u * v * ufl.dx
    expected_form_after_expansion = f1 * u * v * ufl.dx + f2 * u * v * ufl.dx

    assert ufl4rom.utils.expand_sum(form_before_expansion) == expected_form_after_expansion


@pytest.mark.parametrize("op", [ufl.grad, ufl.div, ufl.curl, ufl.nabla_grad, ufl.nabla_div])
def test_expand_sum_real_sum_differential_operators(  # type: ignore[no-any-unimported]
    op: typing.Callable[[ufl.core.expr.Expr], ufl.core.expr.Expr]
) -> None:
    """Test ufl4rom.utils.expand_sum when the form contains a differential operator of the arguments."""
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
    f = ufl.Coefficient(scalar_function_space)

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
    dim = cell.geometric_dimension()
    scalar_element = ufl.finiteelement.FiniteElement(
        "Lagrange", cell, 1, (), ufl.pullback.identity_pullback, ufl.sobolevspace.H1)
    vector_element = ufl.finiteelement.FiniteElement(
        "Lagrange", cell, 1, (dim, ), ufl.pullback.identity_pullback, ufl.sobolevspace.H1)

    domain = ufl.Mesh(vector_element)
    scalar_function_space = ufl.FunctionSpace(domain, scalar_element)

    u = ufl.TrialFunction(scalar_function_space)
    v = ufl.TestFunction(scalar_function_space)
    f = ufl.Coefficient(scalar_function_space)

    form_before_expansion = f * u * v * (ufl.dx + ufl.ds)
    expected_form_after_expansion = f * u * v * ufl.dx + f * u * v * ufl.ds

    assert ufl4rom.utils.expand_sum(form_before_expansion) == expected_form_after_expansion


@pytest.mark.parametrize("product", [ufl.inner, ufl.dot])
def test_expand_sum_vector_real_scalar_coefficients(  # type: ignore[no-any-unimported]
    product: typing.Callable[[ufl.core.expr.Expr, ufl.core.expr.Expr], ufl.core.expr.Expr]
) -> None:
    """Test ufl4rom.utils.expand_sum when the form contains the sum of two vector real-valued coefficients."""
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
    f1 = ufl.Coefficient(scalar_function_space)
    f2 = ufl.Coefficient(scalar_function_space)

    form_before_expansion = (
        ufl.algorithms.renumbering.renumber_indices(product((f1 + f2) * u, v) * ufl.dx))
    expected_form_after_expansion = (
        ufl.algorithms.renumbering.renumber_indices(product(f1 * u, v) * ufl.dx)
        + ufl.algorithms.renumbering.renumber_indices(product(f2 * u, v) * ufl.dx))

    assert ufl4rom.utils.expand_sum(form_before_expansion) == expected_form_after_expansion


def test_expand_sum_vector_real_tensor_coefficients() -> None:
    """Test ufl4rom.utils.expand_sum when the form contains the sum of two tensor real-valued coefficients."""
    cell = ufl.triangle
    dim = cell.geometric_dimension()
    vector_element = ufl.finiteelement.FiniteElement(
        "Lagrange", cell, 1, (dim, ), ufl.pullback.identity_pullback, ufl.sobolevspace.H1)
    tensor_element = ufl.finiteelement.FiniteElement(
        "Lagrange", cell, 1, (dim, dim), ufl.pullback.identity_pullback, ufl.sobolevspace.H1)

    domain = ufl.Mesh(vector_element)
    vector_function_space = ufl.FunctionSpace(domain, vector_element)
    tensor_function_space = ufl.FunctionSpace(domain, tensor_element)

    u = ufl.TrialFunction(vector_function_space)
    v = ufl.TestFunction(vector_function_space)
    f1 = ufl.Coefficient(tensor_function_space)
    f2 = ufl.Coefficient(tensor_function_space)

    form_before_expansion = (
        ufl.algorithms.renumbering.renumber_indices(ufl.inner((f1 + f2) * u, v) * ufl.dx))
    expected_form_after_expansion = (
        ufl.algorithms.renumbering.renumber_indices(ufl.inner(f1 * u, v) * ufl.dx)
        + ufl.algorithms.renumbering.renumber_indices(ufl.inner(f2 * u, v) * ufl.dx))

    assert ufl4rom.utils.expand_sum(form_before_expansion) == expected_form_after_expansion


def test_expand_sum_vector_real_tensor_coefficients_grad() -> None:
    """Test ufl4rom.utils.expand_sum when the form contains the sum of two tensor real-valued coefficients.

    The difference w.r.t. the previous test is that arguments are now defined on a vector finite element space.
    """
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
    tensor_function_space = ufl.FunctionSpace(domain, tensor_element)

    u = ufl.TrialFunction(scalar_function_space)
    v = ufl.TestFunction(scalar_function_space)
    f1 = ufl.Coefficient(tensor_function_space)
    f2 = ufl.Coefficient(tensor_function_space)

    form_before_expansion = (
        ufl.algorithms.renumbering.renumber_indices(ufl.inner((f1 + f2) * ufl.grad(u), ufl.grad(v)) * ufl.dx))
    expected_form_after_expansion = (
        ufl.algorithms.renumbering.renumber_indices(ufl.inner(f1 * ufl.grad(u), ufl.grad(v)) * ufl.dx)
        + ufl.algorithms.renumbering.renumber_indices(ufl.inner(f2 * ufl.grad(u), ufl.grad(v)) * ufl.dx))

    assert ufl4rom.utils.expand_sum(form_before_expansion) == expected_form_after_expansion


def test_expand_sum_vector_real_tensor_coefficients_sum_grad() -> None:
    """Test ufl4rom.utils.expand_sum when the form contains the sum of two tensor real-valued coefficients.

    The difference w.r.t. the previous test is that the gradient now contains the sum of arguments components.
    """
    cell = ufl.triangle
    dim = cell.geometric_dimension()
    vector_element = ufl.finiteelement.FiniteElement(
        "Lagrange", cell, 1, (dim, ), ufl.pullback.identity_pullback, ufl.sobolevspace.H1)
    tensor_element = ufl.finiteelement.FiniteElement(
        "Lagrange", cell, 1, (dim, dim), ufl.pullback.identity_pullback, ufl.sobolevspace.H1)

    domain = ufl.Mesh(vector_element)
    vector_function_space = ufl.FunctionSpace(domain, vector_element)
    tensor_function_space = ufl.FunctionSpace(domain, tensor_element)

    u = ufl.TrialFunction(vector_function_space)
    v = ufl.TestFunction(vector_function_space)
    f1 = ufl.Coefficient(tensor_function_space)
    f2 = ufl.Coefficient(tensor_function_space)

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
    dim = cell.geometric_dimension()
    scalar_element = ufl.finiteelement.FiniteElement(
        "Lagrange", cell, 1, (), ufl.pullback.identity_pullback, ufl.sobolevspace.H1)
    vector_element = ufl.finiteelement.FiniteElement(
        "Lagrange", cell, 1, (dim, ), ufl.pullback.identity_pullback, ufl.sobolevspace.H1)
    mixed_element = ufl.finiteelement.MixedElement([scalar_element, vector_element])

    domain = ufl.Mesh(vector_element)
    scalar_function_space = ufl.FunctionSpace(domain, scalar_element)
    mixed_function_space = ufl.FunctionSpace(domain, mixed_element)

    u = ufl.TrialFunction(mixed_function_space)
    v = ufl.TestFunction(mixed_function_space)
    f1 = ufl.Coefficient(scalar_function_space)
    f2 = ufl.Coefficient(scalar_function_space)

    form_before_expansion = (
        ufl.algorithms.renumbering.renumber_indices(ufl.inner((f1 + f2) * u, v) * ufl.dx))
    expected_form_after_expansion = (
        ufl.algorithms.renumbering.renumber_indices(ufl.inner(f1 * u, v) * ufl.dx)
        + ufl.algorithms.renumbering.renumber_indices(ufl.inner(f2 * u, v) * ufl.dx))

    assert ufl4rom.utils.expand_sum(form_before_expansion) == expected_form_after_expansion


def test_expand_sum_mixed_component_real_scalar_coefficients() -> None:
    """Test ufl4rom.utils.expand_sum with components of arguments defined on a mixed element."""
    cell = ufl.triangle
    dim = cell.geometric_dimension()
    scalar_element = ufl.finiteelement.FiniteElement(
        "Lagrange", cell, 1, (), ufl.pullback.identity_pullback, ufl.sobolevspace.H1)
    vector_element = ufl.finiteelement.FiniteElement(
        "Lagrange", cell, 1, (dim, ), ufl.pullback.identity_pullback, ufl.sobolevspace.H1)
    mixed_element = ufl.finiteelement.MixedElement([scalar_element, vector_element])

    domain = ufl.Mesh(vector_element)
    scalar_function_space = ufl.FunctionSpace(domain, scalar_element)
    mixed_function_space = ufl.FunctionSpace(domain, mixed_element)

    u = ufl.TrialFunction(mixed_function_space)
    v = ufl.TestFunction(mixed_function_space)
    f1 = ufl.Coefficient(scalar_function_space)
    f2 = ufl.Coefficient(scalar_function_space)

    form_before_expansion = (
        ufl.algorithms.renumbering.renumber_indices(ufl.inner((f1 + f2) * u[1], v[1]) * ufl.dx))
    expected_form_after_expansion = (
        ufl.algorithms.renumbering.renumber_indices(ufl.inner(f1 * u[1], v[1]) * ufl.dx)
        + ufl.algorithms.renumbering.renumber_indices(ufl.inner(f2 * u[1], v[1]) * ufl.dx))

    assert ufl4rom.utils.expand_sum(form_before_expansion) == expected_form_after_expansion


def test_expand_sum_complex_no_sum() -> None:
    """Test ufl4rom.utils.expand_sum with when the form actually contains no sum at all and complex-valued arguments."""
    cell = ufl.triangle
    dim = cell.geometric_dimension()
    scalar_element = ufl.finiteelement.FiniteElement(
        "Lagrange", cell, 1, (), ufl.pullback.identity_pullback, ufl.sobolevspace.H1)
    vector_element = ufl.finiteelement.FiniteElement(
        "Lagrange", cell, 1, (dim, ), ufl.pullback.identity_pullback, ufl.sobolevspace.H1)

    domain = ufl.Mesh(vector_element)
    scalar_function_space = ufl.FunctionSpace(domain, scalar_element)

    u = ufl.TrialFunction(scalar_function_space)
    v = ufl.TestFunction(scalar_function_space)
    f = ufl.Coefficient(scalar_function_space)

    form_before_expansion = f * ufl.imag(u) * ufl.real(v) * ufl.dx
    expected_form_after_expansion = f * ufl.imag(u) * ufl.real(v) * ufl.dx

    assert ufl4rom.utils.expand_sum(form_before_expansion) == expected_form_after_expansion


def test_expand_sum_complex_sum() -> None:
    """Test ufl4rom.utils.expand_sum when the form contains the gradient of the sum of complex-valued components."""
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
    f = ufl.Coefficient(scalar_function_space)

    form_before_expansion = f * ufl.imag(u[0] + u[1]) * ufl.real(v[0] + v[1]) * ufl.dx
    expected_form_after_expansion = (
        f * ufl.imag(u[0]) * ufl.real(v[0]) * ufl.dx
        + f * ufl.imag(u[0]) * ufl.real(v[1]) * ufl.dx
        + f * ufl.imag(u[1]) * ufl.real(v[0]) * ufl.dx
        + f * ufl.imag(u[1]) * ufl.real(v[1]) * ufl.dx)

    assert ufl4rom.utils.expand_sum(form_before_expansion) == expected_form_after_expansion
