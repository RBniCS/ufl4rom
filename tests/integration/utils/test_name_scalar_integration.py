# Copyright (C) 2021-2024 by the ufl4rom authors
#
# This file is part of ufl4rom.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Integration tests for ufl4rom.utils.name module."""

import pytest

import ufl4rom.utils


def test_name_scalar_1_dolfinx() -> None:
    """Test a basic advection-diffusion-reaction parametrized form, with all parametrized dolfinx coefficients."""
    ufl = pytest.importorskip("ufl")
    dolfinx = pytest.importorskip("dolfinx")
    pytest.importorskip("dolfinx.fem")
    pytest.importorskip("dolfinx.mesh")
    mpi4py = pytest.importorskip("mpi4py")
    pytest.importorskip("mpi4py.MPI")

    mesh = dolfinx.mesh.create_unit_square(mpi4py.MPI.COMM_WORLD, 2, 2)
    scalar_function_space = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))

    u = ufl.TrialFunction(scalar_function_space)
    v = ufl.TestFunction(scalar_function_space)
    f1 = dolfinx.fem.Function(scalar_function_space, name="parametrized coefficient 1")
    f2 = dolfinx.fem.Function(scalar_function_space, name="parametrized coefficient 2")
    f3 = dolfinx.fem.Function(scalar_function_space, name="parametrized coefficient 3")

    dx = ufl.dx
    grad = ufl.grad
    inner = ufl.inner

    a1 = f3 * f2 * inner(grad(u), grad(v)) * dx + f2 * u.dx(0) * v * dx + f1 * u * v * dx
    assert ufl4rom.utils.name(a1) == "d846657469443c859c51904fab665237345a14c7"


def test_name_scalar_1_firedrake() -> None:
    """Test a basic advection-diffusion-reaction parametrized form, with all parametrized firedrake coefficients."""
    firedrake = pytest.importorskip("firedrake")

    mesh = firedrake.UnitSquareMesh(2, 2)
    scalar_function_space = firedrake.FunctionSpace(mesh, "Lagrange", 1)

    u = firedrake.TrialFunction(scalar_function_space)
    v = firedrake.TestFunction(scalar_function_space)
    f1 = firedrake.Function(scalar_function_space, name="parametrized coefficient 1")
    f2 = firedrake.Function(scalar_function_space, name="parametrized coefficient 2")
    f3 = firedrake.Function(scalar_function_space, name="parametrized coefficient 3")

    dx = firedrake.dx
    grad = firedrake.grad
    inner = firedrake.inner

    a1 = f3 * f2 * inner(grad(u), grad(v)) * dx + f2 * u.dx(0) * v * dx + f1 * u * v * dx
    assert ufl4rom.utils.name(a1) == "72260f6a6b5ad3fee82cc86ac80bf38e5f117554"


def test_name_scalar_13_dolfinx() -> None:
    """We now introduce dolfinx constants in the expression."""
    ufl = pytest.importorskip("ufl")
    dolfinx = pytest.importorskip("dolfinx")
    pytest.importorskip("dolfinx.fem")
    pytest.importorskip("dolfinx.mesh")
    mpi4py = pytest.importorskip("mpi4py")
    pytest.importorskip("mpi4py.MPI")
    np = pytest.importorskip("numpy")
    petsc4py = pytest.importorskip("petsc4py")
    pytest.importorskip("petsc4py.PETSc")

    mesh = dolfinx.mesh.create_unit_square(mpi4py.MPI.COMM_WORLD, 2, 2)
    scalar_function_space = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
    vector_function_space = dolfinx.fem.functionspace(mesh, ("Lagrange", 1, (mesh.geometry.dim, )))
    tensor_function_space = dolfinx.fem.functionspace(mesh, ("Lagrange", 1, (mesh.geometry.dim, mesh.geometry.dim)))

    u = ufl.TrialFunction(scalar_function_space)
    v = ufl.TestFunction(scalar_function_space)
    f1 = dolfinx.fem.Function(scalar_function_space, name="parametrized coefficient 1, scalar")
    f2 = dolfinx.fem.Function(vector_function_space, name="parametrized coefficient 2, vector")
    f3 = dolfinx.fem.Function(tensor_function_space, name="parametrized coefficient 3, tensor")
    c1 = ufl4rom.utils.DolfinxNamedConstant("parametrized constant 1, scalar", 1.0, mesh)
    c2 = dolfinx.fem.Constant(mesh, np.array([[1.0, 2.0], [3.0, 4.0]], petsc4py.PETSc.ScalarType))

    dx = ufl.dx
    grad = ufl.grad
    inner = ufl.inner

    a13 = (
        inner(c2 * f3 * c1 * grad(u), grad(v)) * dx + inner(c1 * f2, grad(u)) * v * dx
        + c1 * f1 * u * v * dx
    )
    if np.issubdtype(petsc4py.PETSc.ScalarType, np.complexfloating):  # names differ due to different c2 dtype
        expected_name = "ed0b562f5e0b662cfe00e4da0c28635e8700087d"
    else:
        expected_name = "8c948af3d427de0acc8f76c012a2c109c2b36187"
    assert ufl4rom.utils.name(a13) == expected_name


def test_name_scalar_13_firedrake() -> None:
    """We now introduce firedrake constants in the expression."""
    firedrake = pytest.importorskip("firedrake")
    np = pytest.importorskip("numpy")
    petsc4py = pytest.importorskip("petsc4py")
    pytest.importorskip("petsc4py.PETSc")

    mesh = firedrake.UnitSquareMesh(2, 2)
    scalar_function_space = firedrake.FunctionSpace(mesh, "Lagrange", 1)
    vector_function_space = firedrake.VectorFunctionSpace(mesh, "Lagrange", 1)
    tensor_function_space = firedrake.TensorFunctionSpace(mesh, "Lagrange", 1)

    u = firedrake.TrialFunction(scalar_function_space)
    v = firedrake.TestFunction(scalar_function_space)
    f1 = firedrake.Function(scalar_function_space, name="parametrized coefficient 1, scalar")
    f2 = firedrake.Function(vector_function_space, name="parametrized coefficient 2, vector")
    f3 = firedrake.Function(tensor_function_space, name="parametrized coefficient 3, tensor")
    c1 = ufl4rom.utils.FiredrakeNamedConstant("parametrized constant 1, scalar", 1.0)
    c2 = firedrake.Constant(((1.0, 2.0), (3.0, 4.0)))

    dx = firedrake.dx
    grad = firedrake.grad
    inner = firedrake.inner

    a13 = (
        inner(c2 * f3 * c1 * grad(u), grad(v)) * dx + inner(c1 * f2, grad(u)) * v * dx
        + c1 * f1 * u * v * dx
    )
    if np.issubdtype(petsc4py.PETSc.ScalarType, np.complexfloating):  # names differ due to different c2 dtype
        expected_name = "3cd044e2a94c786d61061c6b893dd744a7ba95df"
    else:
        expected_name = "bf19de5b633b22e5c9c738ac7f96882048e792e1"
    assert ufl4rom.utils.name(a13) == expected_name


def test_name_scalar_failure_coefficient_dolfinx() -> None:
    """Test a variation of form 1 that will fail due to not having used (dolfinx) named coefficients."""
    ufl = pytest.importorskip("ufl")
    dolfinx = pytest.importorskip("dolfinx")
    pytest.importorskip("dolfinx.fem")
    pytest.importorskip("dolfinx.mesh")
    mpi4py = pytest.importorskip("mpi4py")
    pytest.importorskip("mpi4py.MPI")

    mesh = dolfinx.mesh.create_unit_square(mpi4py.MPI.COMM_WORLD, 2, 2)
    scalar_function_space = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))

    u = ufl.TrialFunction(scalar_function_space)
    v = ufl.TestFunction(scalar_function_space)
    f1 = dolfinx.fem.Function(scalar_function_space)
    f2 = dolfinx.fem.Function(scalar_function_space)
    f3 = dolfinx.fem.Function(scalar_function_space)

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
    scalar_function_space = firedrake.FunctionSpace(mesh, "Lagrange", 1)

    u = firedrake.TrialFunction(scalar_function_space)
    v = firedrake.TestFunction(scalar_function_space)
    f1 = firedrake.Function(scalar_function_space)
    f2 = firedrake.Function(scalar_function_space)
    f3 = firedrake.Function(scalar_function_space)

    dx = firedrake.dx
    grad = firedrake.grad
    inner = firedrake.inner

    a1 = f3 * f2 * inner(grad(u), grad(v)) * dx + f2 * u.dx(0) * v * dx + f1 * u * v * dx
    with pytest.raises(AssertionError) as excinfo:
        ufl4rom.utils.name(a1)
    assert str(excinfo.value) == "Please provide a name to the Function"
