# Copyright (C) 2021-2022 by the ufl4rom authors
#
# This file is part of ufl4rom.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from ufl.algorithms.map_integrands import map_integrand_dags
from ufl.corealg.multifunction import MultiFunction


def rewrite_quotients(form):
    """
    Rewrite quotient expr1 / expr2 as expr1 * (1 / expr2)
    """
    return map_integrand_dags(RewriteQuotientsReplacer(), form)


class RewriteQuotientsReplacer(MultiFunction):
    expr = MultiFunction.reuse_if_untouched

    def division(self, o, n, d):
        return n * (1 / d)
