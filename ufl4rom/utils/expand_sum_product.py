# Copyright (C) 2021-2022 by the ufl4rom authors
#
# This file is part of ufl4rom.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import itertools
from ufl import Form
from ufl.algorithms.map_integrands import map_integrand_dags
from ufl.classes import IndexSum, Sum
from ufl.core.multiindex import MultiIndex
from ufl.corealg.multifunction import MultiFunction


def expand_sum_product(form):
    expander = SumProductExpander()
    # Call sum product expander for each integrand
    expanded_form = map_integrand_dags(expander, form)
    # Split resulting sums into separate integrals
    expanded_split_form_integrals = list()
    for integral in expanded_form.integrals():
        expanded_split_form_integrands = expander.split_sum(integral.integrand())
        expanded_split_form_integrals.extend(
            [integral.reconstruct(integrand=integrand) for integrand in expanded_split_form_integrands])
    return Form(expanded_split_form_integrals)


class SumProductExpander(MultiFunction):
    def __init__(self):
        MultiFunction.__init__(self)
        self.ufl_to_replaced_ufl = dict()
        self.ufl_to_split_ufl = dict()

    expr = MultiFunction.reuse_if_untouched

    def product(self, o, *ops):
        return self._transform_multilinear_operator(o, *ops)

    def conj(self, o, *ops):
        return self._transform_multilinear_operator(o, *ops)

    def real(self, o, *ops):
        return self._transform_multilinear_operator(o, *ops)

    def imag(self, o, *ops):
        return self._transform_multilinear_operator(o, *ops)

    def inner(self, o, *ops):
        return self._transform_multilinear_operator(o, *ops)

    def dot(self, o, *ops):
        return self._transform_multilinear_operator(o, *ops)

    def grad(self, o, *ops):
        return self._transform_multilinear_operator(o, *ops)

    def div(self, o, *ops):
        return self._transform_multilinear_operator(o, *ops)

    def curl(self, o, *ops):
        return self._transform_multilinear_operator(o, *ops)

    def nabla_grad(self, o, *ops):
        return self._transform_multilinear_operator(o, *ops)

    def nabla_div(self, o, *ops):
        return self._transform_multilinear_operator(o, *ops)

    def diff(self, o, *ops):
        return self._transform_multilinear_operator(o, *ops)

    def indexed(self, o, *ops):
        return self._transform_multilinear_operator(o, *ops)

    def component_tensor(self, o, *ops):
        return self._transform_multilinear_operator(o, *ops)

    def _transform_multilinear_operator(self, o, *ops):
        if o not in self.ufl_to_replaced_ufl:
            split_ops = list()
            at_least_one_split = False
            for op in ops:
                split_op = self.split_sum(map_integrand_dags(self, op))
                split_ops.append(split_op)
                if len(split_op) > 1:
                    at_least_one_split = True
            if at_least_one_split:
                new_o = sum(o._ufl_expr_reconstruct_(*ops_) for ops_ in itertools.product(*split_ops))
                self.ufl_to_replaced_ufl[o] = new_o
            else:
                self.ufl_to_replaced_ufl[o] = o
        return self.ufl_to_replaced_ufl[o]

    def _transform_and_attach_multi_index(self, o, *ops):
        if o not in self.ufl_to_replaced_ufl:
            assert len(ops) == 2
            assert isinstance(ops[1], MultiIndex)
            split_op_0 = self.split_sum(map_integrand_dags(self, ops[0]))
            if len(split_op_0) > 1:
                new_o = sum(o._ufl_expr_reconstruct_(op_0, ops[1]) for op_0 in split_op_0)
                self.ufl_to_replaced_ufl[o] = new_o
            else:
                self.ufl_to_replaced_ufl[o] = o
        return self.ufl_to_replaced_ufl[o]

    def split_sum(self, input_):
        if input_ not in self.ufl_to_split_ufl:
            output = list()
            if isinstance(input_, Sum):
                for operand in input_.ufl_operands:
                    output.extend(self.split_sum(operand))
            elif isinstance(input_, IndexSum):
                assert len(input_.ufl_operands) == 2
                summand, multiindex = input_.ufl_operands
                assert isinstance(multiindex, MultiIndex)
                output_0 = self.split_sum(summand)
                output.extend([IndexSum(summand_0, multiindex) for summand_0 in output_0])
            else:
                output.append(input_)
            self.ufl_to_split_ufl[input_] = output
        return self.ufl_to_split_ufl[input_]
