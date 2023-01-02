# Copyright (C) 2021-2023 by the ufl4rom authors
#
# This file is part of ufl4rom.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Unit tests for ufl4rom.utils.rewrite_quotients module."""

import ufl
import ufl.algorithms.renumbering

import ufl4rom.utils


def test_rewrite_quotients_real_no_quotient() -> None:
    """Test ufl4rom.utils.rewrite_quotients when the form actually contains no quotients at all."""
    cell = ufl.triangle
    element = ufl.FiniteElement("Lagrange", cell, 1)

    u = ufl.TrialFunction(element)
    v = ufl.TestFunction(element)
    f = ufl.Coefficient(element)

    form_before_expansion = f * u * v * ufl.dx
    expected_form_after_expansion = f * u * v * ufl.dx

    assert ufl4rom.utils.rewrite_quotients(form_before_expansion) == expected_form_after_expansion


def test_rewrite_quotients_real_one_quotient() -> None:
    """Test ufl4rom.utils.rewrite_quotients with a quotient between two coefficients."""
    cell = ufl.triangle
    element = ufl.FiniteElement("Lagrange", cell, 1)

    u = ufl.TrialFunction(element)
    v = ufl.TestFunction(element)
    f1 = ufl.Coefficient(element)
    f2 = ufl.Coefficient(element)

    form_before_expansion = f1 / f2 * u * v * ufl.dx
    expected_form_after_expansion = f1 * (1 / f2) * u * v * ufl.dx

    assert ufl4rom.utils.rewrite_quotients(form_before_expansion) == expected_form_after_expansion


def test_rewrite_quotients_vector_real_scalar_numerator() -> None:
    """Test ufl4rom.utils.rewrite_quotients with a quotient between two coefficients and arguments in a vector-valued\
    function space."""
    cell = ufl.triangle
    scalar_element = ufl.FiniteElement("Lagrange", cell, 1)
    vector_element = ufl.VectorElement("Lagrange", cell, 1)

    u = ufl.TrialFunction(vector_element)
    v = ufl.TestFunction(vector_element)
    f1 = ufl.Coefficient(scalar_element)
    f2 = ufl.Coefficient(scalar_element)

    form_before_expansion = ufl.algorithms.renumbering.renumber_indices(ufl.inner(f1 / f2 * u, v) * ufl.dx)
    expected_form_after_expansion = ufl.algorithms.renumbering.renumber_indices(
        ufl.inner(f1 * (1 / f2) * u, v) * ufl.dx)

    assert ufl4rom.utils.rewrite_quotients(form_before_expansion) == expected_form_after_expansion


def test_rewrite_quotients_vector_real_tensor_numerator() -> None:
    """Test ufl4rom.utils.rewrite_quotients with a quotient between a tensor coefficient and a scalar coefficient."""
    cell = ufl.triangle
    scalar_element = ufl.FiniteElement("Lagrange", cell, 1)
    vector_element = ufl.VectorElement("Lagrange", cell, 1)
    tensor_element = ufl.TensorElement("Lagrange", cell, 1)

    u = ufl.TrialFunction(vector_element)
    v = ufl.TestFunction(vector_element)
    f1 = ufl.Coefficient(tensor_element)
    f2 = ufl.Coefficient(scalar_element)

    form_before_expansion = ufl.algorithms.renumbering.renumber_indices(ufl.inner(f1 / f2 * u, v) * ufl.dx)
    expected_form_after_expansion = ufl.algorithms.renumbering.renumber_indices(
        ufl.inner(f1 * (1 / f2) * u, v) * ufl.dx)

    assert ufl4rom.utils.rewrite_quotients(form_before_expansion) == expected_form_after_expansion


def test_rewrite_quotients_complex_no_quotient() -> None:
    """Test ufl4rom.utils.rewrite_quotients when the form actually contains no quotients at all and complex-valued\
    arguments."""
    cell = ufl.triangle
    element = ufl.FiniteElement("Lagrange", cell, 1)

    u = ufl.TrialFunction(element)
    v = ufl.TestFunction(element)
    f = ufl.Coefficient(element)

    form_before_expansion = f * ufl.imag(u) * ufl.real(v) * ufl.dx
    expected_form_after_expansion = f * ufl.imag(u) * ufl.real(v) * ufl.dx

    assert ufl4rom.utils.rewrite_quotients(form_before_expansion) == expected_form_after_expansion


def test_rewrite_quotients_complex_one_quotient() -> None:
    """Test ufl4rom.utils.expand_sum when the form contains the sum of two complex coefficients."""
    cell = ufl.triangle
    element = ufl.FiniteElement("Lagrange", cell, 1)

    u = ufl.TrialFunction(element)
    v = ufl.TestFunction(element)
    f1 = ufl.Coefficient(element)
    f2 = ufl.Coefficient(element)

    form_before_expansion = ufl.imag(f1 / f2) * u * v * ufl.dx
    expected_form_after_expansion = ufl.imag(f1 * (1 / f2)) * u * v * ufl.dx

    assert ufl4rom.utils.rewrite_quotients(form_before_expansion) == expected_form_after_expansion
