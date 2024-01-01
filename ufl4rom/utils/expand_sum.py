# Copyright (C) 2021-2024 by the ufl4rom authors
#
# This file is part of ufl4rom.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Expand a form continaing the integral of a sum to contain instead the sum of several integrals."""

import itertools
import typing

import ufl
import ufl.algorithms.map_integrands
import ufl.classes
import ufl.core.multiindex
import ufl.corealg.multifunction


def expand_sum(form: ufl.Form) -> ufl.Form:  # type: ignore[no-any-unimported]
    """
    Expand a form continaing the integral of a sum to contain instead the sum of several integrals.

    Expansion is applied to every multilinear operator so that the returned integrals contain no sum.
    """
    expander = SumExpander()
    # Call sum expander for each integrand
    expanded_form = ufl.algorithms.map_integrands.map_integrand_dags(expander, form)
    # Split resulting sums into separate integrals
    expanded_split_form_integrals = list()
    for integral in expanded_form.integrals():
        expanded_split_form_integrands = expander.split_sum(integral.integrand())
        expanded_split_form_integrals.extend(
            [integral.reconstruct(integrand=integrand) for integrand in expanded_split_form_integrands])
    return ufl.Form(expanded_split_form_integrals)


class SumExpander(ufl.corealg.multifunction.MultiFunction):  # type: ignore[misc, no-any-unimported]
    """UFL MultiFunction that carries out the sum expansion."""

    def __init__(self) -> None:
        super().__init__()
        self.ufl_to_replaced_ufl: typing.Dict[  # type: ignore[no-any-unimported]
            ufl.core.expr.Expr, ufl.core.expr.Expr] = dict()
        self.ufl_to_split_ufl: typing.Dict[  # type: ignore[no-any-unimported]
            ufl.core.expr.Expr, typing.List[ufl.core.expr.Expr]] = dict()

    expr = ufl.corealg.multifunction.MultiFunction.reuse_if_untouched

    def product(  # type: ignore[no-any-unimported]
        self, o: ufl.core.expr.Expr, *ops: ufl.core.expr.Expr
    ) -> ufl.core.expr.Expr:
        """Transorm products as multilinear operators."""
        return self._transform_multilinear_operator(o, *ops)

    def conj(  # type: ignore[no-any-unimported]
        self, o: ufl.core.expr.Expr, *ops: ufl.core.expr.Expr
    ) -> ufl.core.expr.Expr:
        """Transorm conjugation as multilinear operators."""
        return self._transform_multilinear_operator(o, *ops)

    def real(  # type: ignore[no-any-unimported]
        self, o: ufl.core.expr.Expr, *ops: ufl.core.expr.Expr
    ) -> ufl.core.expr.Expr:
        """Transorm real part as multilinear operators."""
        return self._transform_multilinear_operator(o, *ops)

    def imag(  # type: ignore[no-any-unimported]
        self, o: ufl.core.expr.Expr, *ops: ufl.core.expr.Expr
    ) -> ufl.core.expr.Expr:
        """Transorm imaginary part as multilinear operators."""
        return self._transform_multilinear_operator(o, *ops)

    def inner(  # type: ignore[no-any-unimported]
        self, o: ufl.core.expr.Expr, *ops: ufl.core.expr.Expr
    ) -> ufl.core.expr.Expr:
        """Transorm inner product as multilinear operators."""
        return self._transform_multilinear_operator(o, *ops)

    def dot(  # type: ignore[no-any-unimported]
        self, o: ufl.core.expr.Expr, *ops: ufl.core.expr.Expr
    ) -> ufl.core.expr.Expr:
        """Transorm dot product as multilinear operators."""
        return self._transform_multilinear_operator(o, *ops)

    def grad(  # type: ignore[no-any-unimported]
        self, o: ufl.core.expr.Expr, *ops: ufl.core.expr.Expr
    ) -> ufl.core.expr.Expr:
        """Transorm gradient as multilinear operators."""
        return self._transform_multilinear_operator(o, *ops)

    def div(  # type: ignore[no-any-unimported]
        self, o: ufl.core.expr.Expr, *ops: ufl.core.expr.Expr
    ) -> ufl.core.expr.Expr:
        """Transorm divergence as multilinear operators."""
        return self._transform_multilinear_operator(o, *ops)

    def curl(  # type: ignore[no-any-unimported]
        self, o: ufl.core.expr.Expr, *ops: ufl.core.expr.Expr
    ) -> ufl.core.expr.Expr:
        """Transorm curl as multilinear operators."""
        return self._transform_multilinear_operator(o, *ops)

    def nabla_grad(  # type: ignore[no-any-unimported]
        self, o: ufl.core.expr.Expr, *ops: ufl.core.expr.Expr
    ) -> ufl.core.expr.Expr:
        """Transorm gradient (different index convention) as multilinear operators."""
        return self._transform_multilinear_operator(o, *ops)

    def nabla_div(  # type: ignore[no-any-unimported]
        self, o: ufl.core.expr.Expr, *ops: ufl.core.expr.Expr
    ) -> ufl.core.expr.Expr:
        """Transorm divergence (different index convention) as multilinear operators."""
        return self._transform_multilinear_operator(o, *ops)

    def indexed(  # type: ignore[no-any-unimported]
        self, o: ufl.core.expr.Expr, *ops: ufl.core.expr.Expr
    ) -> ufl.core.expr.Expr:
        """Transorm list components as multilinear operators."""
        return self._transform_multilinear_operator(o, *ops)

    def component_tensor(  # type: ignore[no-any-unimported]
        self, o: ufl.core.expr.Expr, *ops: ufl.core.expr.Expr
    ) -> ufl.core.expr.Expr:
        """Transorm tensor components as multilinear operators."""
        return self._transform_multilinear_operator(o, *ops)

    def _transform_multilinear_operator(  # type: ignore[no-any-unimported]
        self, o: ufl.core.expr.Expr, *ops: ufl.core.expr.Expr
    ) -> ufl.core.expr.Expr:
        """Split sums in multilinear operators."""
        if o not in self.ufl_to_replaced_ufl:
            split_ops = list()
            at_least_one_split = False
            for op in ops:
                split_op = self.split_sum(ufl.algorithms.map_integrands.map_integrand_dags(self, op))
                split_ops.append(split_op)
                if len(split_op) > 1:
                    at_least_one_split = True
            if at_least_one_split:
                new_o = sum(o._ufl_expr_reconstruct_(*ops_) for ops_ in itertools.product(*split_ops))
                self.ufl_to_replaced_ufl[o] = new_o
            else:
                self.ufl_to_replaced_ufl[o] = o
        return self.ufl_to_replaced_ufl[o]

    def split_sum(self, input_: ufl.core.expr.Expr) -> ufl.core.expr.Expr:  # type: ignore[no-any-unimported]
        """Split sums in an UFL expression."""
        if input_ not in self.ufl_to_split_ufl:
            output = list()
            if isinstance(input_, ufl.classes.Sum):
                for operand in input_.ufl_operands:
                    output.extend(self.split_sum(operand))
            elif isinstance(input_, ufl.classes.IndexSum):
                assert len(input_.ufl_operands) == 2
                summand, multiindex = input_.ufl_operands
                assert isinstance(multiindex, ufl.core.multiindex.MultiIndex)
                output_0 = self.split_sum(summand)
                output.extend([ufl.classes.IndexSum(summand_0, multiindex) for summand_0 in output_0])
            else:
                output.append(input_)
            self.ufl_to_split_ufl[input_] = output
        return self.ufl_to_split_ufl[input_]
