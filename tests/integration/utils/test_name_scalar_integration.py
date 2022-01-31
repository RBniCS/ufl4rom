# Copyright (C) 2021-2022 by the ufl4rom authors
#
# This file is part of ufl4rom.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Integration tests for ufl4rom.utils.name module."""

import pytest

import ufl4rom.utils


def test_name_scalar_1_dolfin() -> None:
    """Test a basic advection-diffusion-reaction parametrized form, with all parametrized dolfin coefficients."""
    dolfin = pytest.importorskip("dolfin")
    mesh = dolfin.UnitSquareMesh(2, 2)
    V = dolfin.FunctionSpace(mesh, "Lagrange", 1)

    u = dolfin.TrialFunction(V)
    v = dolfin.TestFunction(V)
    f1 = dolfin.Function(V, name="parametrized coefficient 1")
    f2 = dolfin.Function(V, name="parametrized coefficient 2")
    f3 = dolfin.Function(V, name="parametrized coefficient 3")

    dx = dolfin.dx
    grad = dolfin.grad
    inner = dolfin.inner

    a1 = f3 * f2 * inner(grad(u), grad(v)) * dx + f2 * u.dx(0) * v * dx + f1 * u * v * dx
    assert ufl4rom.utils.name(a1) == "971696bc0259455b4145dc94396c3a2d6d2594dc"


def test_name_scalar_1_dolfinx() -> None:
    """Test a basic advection-diffusion-reaction parametrized form, with all parametrized dolfinx coefficients."""
    ufl = pytest.importorskip("ufl")
    dolfinx = pytest.importorskip("dolfinx")
    pytest.importorskip("dolfinx.fem")
    pytest.importorskip("dolfinx.mesh")
    mpi4py = pytest.importorskip("mpi4py")
    pytest.importorskip("mpi4py.MPI")
    mesh = dolfinx.mesh.create_unit_square(mpi4py.MPI.COMM_WORLD, 2, 2)
    V = dolfinx.fem.FunctionSpace(mesh, ("Lagrange", 1))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    f1 = dolfinx.fem.Function(V, name="parametrized coefficient 1")
    f2 = dolfinx.fem.Function(V, name="parametrized coefficient 2")
    f3 = dolfinx.fem.Function(V, name="parametrized coefficient 3")

    dx = ufl.dx
    grad = ufl.grad
    inner = ufl.inner

    a1 = f3 * f2 * inner(grad(u), grad(v)) * dx + f2 * u.dx(0) * v * dx + f1 * u * v * dx
    assert ufl4rom.utils.name(a1) == "971696bc0259455b4145dc94396c3a2d6d2594dc"


def test_name_scalar_1_firedrake() -> None:
    """Test a basic advection-diffusion-reaction parametrized form, with all parametrized firedrake coefficients."""
    firedrake = pytest.importorskip("firedrake")
    mesh = firedrake.UnitSquareMesh(2, 2)
    V = firedrake.FunctionSpace(mesh, "Lagrange", 1)

    u = firedrake.TrialFunction(V)
    v = firedrake.TestFunction(V)
    f1 = firedrake.Function(V, name="parametrized coefficient 1")
    f2 = firedrake.Function(V, name="parametrized coefficient 2")
    f3 = firedrake.Function(V, name="parametrized coefficient 3")

    dx = firedrake.dx
    grad = firedrake.grad
    inner = firedrake.inner

    a1 = f3 * f2 * inner(grad(u), grad(v)) * dx + f2 * u.dx(0) * v * dx + f1 * u * v * dx
    assert ufl4rom.utils.name(a1) == "72260f6a6b5ad3fee82cc86ac80bf38e5f117554"


def test_name_scalar_13_dofin() -> None:
    """We now introduce dolfin constants in the expression."""
    dolfin = pytest.importorskip("dolfin")

    mesh = dolfin.UnitSquareMesh(2, 2)
    scalar_V = dolfin.FunctionSpace(mesh, "Lagrange", 1)
    vector_V = dolfin.VectorFunctionSpace(mesh, "Lagrange", 1)
    tensor_V = dolfin.TensorFunctionSpace(mesh, "Lagrange", 1)

    u = dolfin.TrialFunction(scalar_V)
    v = dolfin.TestFunction(scalar_V)
    f1 = dolfin.Function(scalar_V, name="parametrized coefficient 1, scalar")
    f2 = dolfin.Function(vector_V, name="parametrized coefficient 2, vector")
    f3 = dolfin.Function(tensor_V, name="parametrized coefficient 3, tensor")
    c1 = ufl4rom.utils.DolfinNamedConstant("parametrized constant 1, scalar", 1.0)
    c2 = dolfin.Constant(((1.0, 2.0), (3.0, 4.0)))

    dx = dolfin.dx
    grad = dolfin.grad
    inner = dolfin.inner

    a13 = (
        inner(c2 * f3 * c1 * grad(u), grad(v)) * dx + inner(c1 * f2, grad(u)) * v * dx
        + c1 * f1 * u * v * dx
    )
    assert ufl4rom.utils.name(a13) == "06b6fad151908da18f966cfcf17b2ecf6850006c"


def test_name_scalar_13_dofinx() -> None:
    """We now introduce dolfinx constants in the expression."""
    ufl = pytest.importorskip("ufl")
    dolfinx = pytest.importorskip("dolfinx")
    pytest.importorskip("dolfinx.fem")
    pytest.importorskip("dolfinx.mesh")
    mpi4py = pytest.importorskip("mpi4py")
    pytest.importorskip("mpi4py.MPI")

    mesh = dolfinx.mesh.create_unit_square(mpi4py.MPI.COMM_WORLD, 2, 2)
    scalar_V = dolfinx.fem.FunctionSpace(mesh, ("Lagrange", 1))
    vector_V = dolfinx.fem.VectorFunctionSpace(mesh, ("Lagrange", 1))
    tensor_V = dolfinx.fem.TensorFunctionSpace(mesh, ("Lagrange", 1))

    u = ufl.TrialFunction(scalar_V)
    v = ufl.TestFunction(scalar_V)
    f1 = dolfinx.fem.Function(scalar_V, name="parametrized coefficient 1, scalar")
    f2 = dolfinx.fem.Function(vector_V, name="parametrized coefficient 2, vector")
    f3 = dolfinx.fem.Function(tensor_V, name="parametrized coefficient 3, tensor")
    c1 = ufl4rom.utils.DolfinxNamedConstant("parametrized constant 1, scalar", mesh, 1.0)
    c2 = dolfinx.fem.Constant(mesh, ((1.0, 2.0), (3.0, 4.0)))

    dx = ufl.dx
    grad = ufl.grad
    inner = ufl.inner

    a13 = (
        inner(c2 * f3 * c1 * grad(u), grad(v)) * dx + inner(c1 * f2, grad(u)) * v * dx
        + c1 * f1 * u * v * dx
    )
    assert ufl4rom.utils.name(a13) == "148abf0ce5b66db179e05a6f9c03880cd4dfc32b"


def test_name_scalar_13_firedrake() -> None:
    """We now introduce firedrake constants in the expression."""
    firedrake = pytest.importorskip("firedrake")

    mesh = firedrake.UnitSquareMesh(2, 2)
    scalar_V = firedrake.FunctionSpace(mesh, "Lagrange", 1)
    vector_V = firedrake.VectorFunctionSpace(mesh, "Lagrange", 1)
    tensor_V = firedrake.TensorFunctionSpace(mesh, "Lagrange", 1)

    u = firedrake.TrialFunction(scalar_V)
    v = firedrake.TestFunction(scalar_V)
    f1 = firedrake.Function(scalar_V, name="parametrized coefficient 1, scalar")
    f2 = firedrake.Function(vector_V, name="parametrized coefficient 2, vector")
    f3 = firedrake.Function(tensor_V, name="parametrized coefficient 3, tensor")
    c1 = ufl4rom.utils.FiredrakeNamedConstant("parametrized constant 1, scalar", 1.0)
    c2 = firedrake.Constant(((1.0, 2.0), (3.0, 4.0)))

    dx = firedrake.dx
    grad = firedrake.grad
    inner = firedrake.inner

    a13 = (
        inner(c2 * f3 * c1 * grad(u), grad(v)) * dx + inner(c1 * f2, grad(u)) * v * dx
        + c1 * f1 * u * v * dx
    )
    assert ufl4rom.utils.name(a13) == "e869ce69d844731a95d97a5d560cd833c61335d1"


def test_name_scalar_failure_coefficient_dolfin() -> None:
    """Test a variation of form 1 that will fail due to not having used (dolfin) named coefficients."""
    dolfin = pytest.importorskip("dolfin")
    mesh = dolfin.UnitSquareMesh(2, 2)
    V = dolfin.FunctionSpace(mesh, "Lagrange", 1)

    u = dolfin.TrialFunction(V)
    v = dolfin.TestFunction(V)
    f1 = dolfin.Function(V)
    f2 = dolfin.Function(V)
    f3 = dolfin.Function(V)

    dx = dolfin.dx
    grad = dolfin.grad
    inner = dolfin.inner

    a1 = f3 * f2 * inner(grad(u), grad(v)) * dx + f2 * u.dx(0) * v * dx + f1 * u * v * dx
    with pytest.raises(AssertionError) as excinfo:
        ufl4rom.utils.name(a1)
    assert str(excinfo.value) == "Please provide a name to the Function"


def test_name_scalar_failure_coefficient_dolfinx() -> None:
    """Test a variation of form 1 that will fail due to not having used (dolfinx) named coefficients."""
    ufl = pytest.importorskip("ufl")
    dolfinx = pytest.importorskip("dolfinx")
    pytest.importorskip("dolfinx.fem")
    pytest.importorskip("dolfinx.mesh")
    mpi4py = pytest.importorskip("mpi4py")
    pytest.importorskip("mpi4py.MPI")
    mesh = dolfinx.mesh.create_unit_square(mpi4py.MPI.COMM_WORLD, 2, 2)
    V = dolfinx.fem.FunctionSpace(mesh, ("Lagrange", 1))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    f1 = dolfinx.fem.Function(V)
    f2 = dolfinx.fem.Function(V)
    f3 = dolfinx.fem.Function(V)

    dx = ufl.dx
    grad = ufl.grad
    inner = ufl.inner

    a1 = f3 * f2 * inner(grad(u), grad(v)) * dx + f2 * u.dx(0) * v * dx + f1 * u * v * dx
    with pytest.raises(AssertionError) as excinfo:
        ufl4rom.utils.name(a1)
    assert str(excinfo.value) == "Please provide a name to the Function"


def test_name_scalar_failure_coefficient_firedrake() -> None:
    """Test variation of form 1 that will fail due to not having used (firedrake) named coefficients."""
    firedrake = pytest.importorskip("firedrake")
    mesh = firedrake.UnitSquareMesh(2, 2)
    V = firedrake.FunctionSpace(mesh, "Lagrange", 1)

    u = firedrake.TrialFunction(V)
    v = firedrake.TestFunction(V)
    f1 = firedrake.Function(V)
    f2 = firedrake.Function(V)
    f3 = firedrake.Function(V)

    dx = firedrake.dx
    grad = firedrake.grad
    inner = firedrake.inner

    a1 = f3 * f2 * inner(grad(u), grad(v)) * dx + f2 * u.dx(0) * v * dx + f1 * u * v * dx
    with pytest.raises(AssertionError) as excinfo:
        ufl4rom.utils.name(a1)
    assert str(excinfo.value) == "Please provide a name to the Function"
