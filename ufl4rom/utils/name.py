# Copyright (C) 2021-2023 by the ufl4rom authors
#
# This file is part of ufl4rom.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Compute a stable name for an expression, an integral or a form."""

import hashlib
import re
import typing

import ufl
import ufl.algorithms.map_integrands
import ufl.algorithms.renumbering
import ufl.core.expr
import ufl.corealg.map_dag
import ufl.corealg.multifunction

from ufl4rom.utils.backends import DolfinxConstant, DolfinxFunction, FiredrakeConstant, FiredrakeFunction
from ufl4rom.utils.named_coefficient import NamedCoefficient
from ufl4rom.utils.named_constant import DolfinxNamedConstant, FiredrakeNamedConstant, NamedConstant, NamedConstantValue


def name(  # type: ignore[no-any-unimported]
    e: typing.Union[ufl.core.expr.Expr, ufl.Form, ufl.Integral], debug: bool = False
) -> str:
    """Compute a stable name for an expression, an integral or a form."""
    if debug:
        print(f"Original expression:\n{e}\n")
    # Preprocess indices first, as their numeric value might change from run to run, but they
    # are always sorted the same way
    e = ufl.algorithms.renumbering.renumber_indices(e)
    if debug:
        print(f"Expression after index renumbering:\n{e}\n")
    # Construct name from the preprocessed expression
    nh = NameHandler()
    if isinstance(e, (ufl.Form, ufl.Integral)):
        e = ufl.algorithms.map_integrands.map_integrand_dags(nh, e)
    else:
        assert isinstance(e, ufl.core.expr.Expr)
        e, = ufl.corealg.map_dag.map_expr_dags(nh, [e])
    # Obatain string representation
    repr_e = repr(e)
    if debug:
        print(f"Expression after name replacement:\n{repr_e}\n")
    # Some backends (e.g., firedrake.mesh.MeshTopology) do not define repr for every object, and thus
    # we may still have strings like <ClassName object at address> in the representation.
    # In those cases simply discard the address in order to have a reproducible representation.
    repr_e = re.sub(r"\<(.+?) object at (.+?)\>", r"\1", repr_e)
    if debug:
        print(f"Expression after address discard:\n{repr_e}\n")
    # All backends keep an internal id associated to the mesh object, and such id is not stripped by
    # renumber_indices. Strip the mesh id manually.
    repr_e = re.sub(  # ufl
        r'Mesh\(ufl.finiteelement.FiniteElement\("Lagrange", (.+?), (.+?), (.+?), (.+?), (.+?)\), (.+?)\)',
        r'Mesh(ufl.finiteelement.FiniteElement("Lagrange", \1, \2, \3, \4, \5))',
        repr_e
    )
    repr_e = re.sub(  # firedrake
        r"Mesh\(VectorElement\(FiniteElement\('Lagrange', (.+?), (.+?)\), dim=(.+?)\), (.+?)\)",
        r"Mesh(VectorElement(FiniteElement('Lagrange', \1, \2), dim=\3))",
        repr_e
    )
    repr_e = re.sub(  # FEniCSx
        r"Mesh\(blocked element \(Basix element \(P, (.+?), (.+?), (.+?), (.+?), (.+?)\), (.+?)\), (.+?)\)",
        r"Mesh(blocked element (Basix element (P, \1, \2, \3, \4, \5), \6))",
        repr_e
    )
    if debug:
        print(f"Expression after id discard:\n{repr_e}\n")
    # Compute SHA of the representation
    sha = hashlib.sha1(repr_e.encode("utf-8")).hexdigest()
    if debug:
        print(f"Computed name:\n{sha}")
    return sha


class NameHandler(ufl.corealg.multifunction.MultiFunction):  # type: ignore[misc, no-any-unimported]
    """Replace named objects, or objects with a deducible name, in an expression."""

    expr = ufl.corealg.multifunction.MultiFunction.reuse_if_untouched

    def constant(self, o: ufl.Constant) -> NamedConstant:  # type: ignore[no-any-unimported]
        """
        Replace a ufl Constant with a ufl4rom NamedConstant, when possible.

        Processes the following backends:
        * dolfinx: preserves name if provided, otherwise sets the name to the current value of the constant

        Note that the following backends:
        * firedrake
        actually implement their Constant objects inheriting from ufl ConstantValue, so these cases are considered
        in the method below.

        Raises an error when an unnamed ufl Constant is provided.
        """
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

    def constant_value(  # type: ignore[no-any-unimported]
        self, o: ufl.constantvalue.ConstantValue
    ) -> ufl.constantvalue.ConstantValue:
        """
        Replace a ufl ConstantValue with a ufl4rom NamedConstantValue, when possible.

        Processes the following backends:
        * firedrake: preserves name if provided, otherwise sets the name to the current value of the constant

        Return the constant value unchanged when an unnamed ufl ConstantValue object is provided,
        since it is safe to do so because that object is not counted.
        """
        if isinstance(o, NamedConstantValue):  # pragma: no cover
            return o
        elif isinstance(o, FiredrakeConstant):
            # firedrake subclass ufl.ConstantValue (rather than ufl.Constant) in their definition
            # of a Constant value.
            if isinstance(o, FiredrakeNamedConstant):
                return NamedConstantValue(str(o), o._ufl_shape)
            else:
                # Both of them provide a values attribute: use it to define a new NamedCoefficient
                return NamedConstantValue(str(o.values()), o._ufl_shape)
        else:
            return o

    def coefficient(self, o: ufl.Coefficient) -> NamedCoefficient:  # type: ignore[no-any-unimported]
        """
        Replace a ufl Coefficient with a ufl4rom NamedCoefficient, when possible.

        Processes the following backends which use Coefficient for Function objects:
        * dolfinx: preserves name if provided, otherwise raises an error
        * firedrake: preserves name if provided, otherwise raises an error

        Raises an error also when an unnamed ufl Coefficient is provided.
        """
        if isinstance(o, NamedCoefficient):
            return o
        elif isinstance(o, DolfinxFunction):
            # dolfinx default name for functions is f
            assert not o.name == "f", "Please provide a name to the Function"
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
