# Copyright (C) 2021 by the ufl4rom authors
#
# This file is part of ufl4rom.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import hashlib
import re
from ufl import Form
from ufl.algorithms.map_integrands import map_integrand_dags
from ufl.algorithms.renumbering import renumber_indices
from ufl.corealg.map_dag import map_expr_dags
from ufl.corealg.multifunction import MultiFunction
from ufl.integral import Integral
from ufl4rom.utils.backends import (
    DolfinConstant, DolfinFunction, DolfinxConstant, DolfinxFunction, FiredrakeConstant, FiredrakeFunction)
from ufl4rom.utils.named_coefficient import NamedCoefficient
from ufl4rom.utils.named_constant import (
    DolfinNamedConstant, DolfinxNamedConstant, FiredrakeNamedConstant, NamedConstant)


def name(e):
    # Preprocess indices first, as their numeric value might change from run to run, but they
    # are always sorted the same way
    e = renumber_indices(e)
    # Construct name from the preprocessed expression
    nh = NameHandler()
    if isinstance(e, (Form, Integral)):
        e = map_integrand_dags(nh, e)
    else:
        e = map_expr_dags(nh, [e])
    # Obatain string representation
    repr_e = repr(e)
    # Some backends (e.g., firedrake.mesh.MeshTopology) do not define repr for every object, and thus
    # we may still have strings like <ClassName object at address> in the representation.
    # In those cases simply discard the address in order to have a reproducible representation.
    repr_e = re.sub(r"\<(.+?) object at (.+?)\>", r"\1", repr_e)
    # All backends keep an internal id associated to the mesh object, and such id is not stripped by
    # renumber_indices. Strip the mesh id manually.
    repr_e = re.sub(
        r"Mesh\(VectorElement\(FiniteElement\('Lagrange', (.+?), (.+?)\), dim=(.+?)\), (.+?)\)",
        r"Mesh(VectorElement(FiniteElement('Lagrange', \1, \2), dim=\3))",
        repr_e
    )
    # Compute SHA of the representation
    return hashlib.sha1(repr_e.encode("utf-8")).hexdigest()


class NameHandler(MultiFunction):
    expr = MultiFunction.reuse_if_untouched

    def constant(self, o):
        if isinstance(o, NamedConstant):
            return o
        elif isinstance(o, DolfinxConstant):
            # dolfinx subclasses ufl.Constant in its definition of a Constant value.
            # It defines a value attribute: use it to define a new NamedConstant.
            if isinstance(o, DolfinxNamedConstant):
                return NamedConstant(str(o), o._ufl_domain, o._ufl_shape)
            else:
                return NamedConstant(str(o.value), o._ufl_domain, o._ufl_shape)
        else:
            raise RuntimeError(
                "The case of plain UFL constants is not handled, because its value cannot be extracted")

    def coefficient(self, o):
        if isinstance(o, NamedCoefficient):
            return o
        elif isinstance(o, (DolfinConstant, FiredrakeConstant)):
            # dolfin and firedrake subclass ufl.Coefficient (rather than ufl.Constant) in their definition
            # of a Constant value.
            if isinstance(o, (DolfinNamedConstant, FiredrakeNamedConstant)):
                return NamedCoefficient(str(o), o._ufl_function_space)
            else:
                # Both of them provide a values attribute: use it to define a new NamedCoefficient
                return NamedCoefficient(str(o.values()), o._ufl_function_space)
        elif isinstance(o, DolfinFunction):
            # dolfin default name for functions starts with f_
            assert not o.name().startswith("f_"), "Please provide a name to the Function"
            # Use the non-default name to return a NamedCoefficient
            return NamedCoefficient(o.name(), o._ufl_function_space)
        elif isinstance(o, DolfinxFunction):
            # dolfinx default name for functions starts with f_
            assert not o.name.startswith("f_"), "Please provide a name to the Function"
            # Use the non-default name to return a NamedCoefficient
            return NamedCoefficient(o.name, o._ufl_function_space)
        elif isinstance(o, FiredrakeFunction):
            # firedrake default name for functions starts with function_
            assert not o.name().startswith("function_"), "Please provide a name to the Function"
            # Use the non-default name to return a NamedCoefficient
            return NamedCoefficient(o.name(), o._ufl_function_space)
        else:
            raise RuntimeError(
                "The case of plain UFL coefficients is not handled, because its name cannot be changed")
