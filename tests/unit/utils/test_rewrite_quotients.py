# Copyright (C) 2021-2024 by the ufl4rom authors
#
# This file is part of ufl4rom.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Unit tests for ufl4rom.utils.rewrite_quotients module."""

import ufl
import ufl.algorithms.renumbering
import ufl.finiteelement
import ufl.pullback
import ufl.sobolevspace

import ufl4rom.utils


def test_rewrite_quotients_real_no_quotient() -> None:
    """Test ufl4rom.utils.rewrite_quotients when the form actually contains no quotients at all."""
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
    f = ufl.Coefficient(scalar_function_space)

    form_before_expansion = f * u * v * ufl.dx
    expected_form_after_expansion = f * u * v * ufl.dx

    assert ufl4rom.utils.rewrite_quotients(form_before_expansion) == expected_form_after_expansion


def test_rewrite_quotients_real_one_quotient() -> None:
    """Test ufl4rom.utils.rewrite_quotients with a quotient between two coefficients."""
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

    form_before_expansion = f1 / f2 * u * v * ufl.dx
    expected_form_after_expansion = f1 * (1 / f2) * u * v * ufl.dx

    assert ufl4rom.utils.rewrite_quotients(form_before_expansion) == expected_form_after_expansion


def test_rewrite_quotients_vector_real_scalar_numerator() -> None:
    """Test ufl4rom.utils.rewrite_quotients with a quotient in an expression with a vector-valued coefficients."""
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
    f1 = ufl.Coefficient(scalar_function_space)
    f2 = ufl.Coefficient(scalar_function_space)

    form_before_expansion = ufl.algorithms.renumbering.renumber_indices(ufl.inner(f1 / f2 * u, v) * ufl.dx)
    expected_form_after_expansion = ufl.algorithms.renumbering.renumber_indices(
        ufl.inner(f1 * (1 / f2) * u, v) * ufl.dx)

    assert ufl4rom.utils.rewrite_quotients(form_before_expansion) == expected_form_after_expansion


def test_rewrite_quotients_vector_real_tensor_numerator() -> None:
    """Test ufl4rom.utils.rewrite_quotients with a quotient between a tensor coefficient and a scalar coefficient."""
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
    f1 = ufl.Coefficient(tensor_function_space)
    f2 = ufl.Coefficient(scalar_function_space)

    form_before_expansion = ufl.algorithms.renumbering.renumber_indices(ufl.inner(f1 / f2 * u, v) * ufl.dx)
    expected_form_after_expansion = ufl.algorithms.renumbering.renumber_indices(
        ufl.inner(f1 * (1 / f2) * u, v) * ufl.dx)

    assert ufl4rom.utils.rewrite_quotients(form_before_expansion) == expected_form_after_expansion


def test_rewrite_quotients_complex_no_quotient() -> None:
    """Test ufl4rom.utils.rewrite_quotients when the form contains no quotients and complex-valued arguments."""
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
    f = ufl.Coefficient(scalar_function_space)

    form_before_expansion = f * ufl.imag(u) * ufl.real(v) * ufl.dx
    expected_form_after_expansion = f * ufl.imag(u) * ufl.real(v) * ufl.dx

    assert ufl4rom.utils.rewrite_quotients(form_before_expansion) == expected_form_after_expansion


def test_rewrite_quotients_complex_one_quotient() -> None:
    """Test ufl4rom.utils.expand_sum when the form contains the sum of two complex coefficients."""
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

    form_before_expansion = ufl.imag(f1 / f2) * u * v * ufl.dx
    expected_form_after_expansion = ufl.imag(f1 * (1 / f2)) * u * v * ufl.dx

    assert ufl4rom.utils.rewrite_quotients(form_before_expansion) == expected_form_after_expansion
