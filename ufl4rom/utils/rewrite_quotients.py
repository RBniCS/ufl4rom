# Copyright (C) 2021-2022 by the ufl4rom authors
#
# This file is part of ufl4rom.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Rewrite quotient expr1 / expr2 as expr1 * (1 / expr2)."""

import ufl
import ufl.algorithms.map_integrands
import ufl.corealg.multifunction


def rewrite_quotients(form: ufl.Form) -> ufl.Form:  # type: ignore[no-any-unimported]
    """Rewrite quotient expr1 / expr2 as expr1 * (1 / expr2)."""
    return ufl.algorithms.map_integrands.map_integrand_dags(RewriteQuotientsReplacer(), form)


class RewriteQuotientsReplacer(ufl.corealg.multifunction.MultiFunction):  # type: ignore[misc, no-any-unimported]
    """UFL MultiFunction object that carries out division replacement."""

    expr = ufl.corealg.multifunction.MultiFunction.reuse_if_untouched

    def division(  # type: ignore[no-any-unimported]
        self, o: ufl.core.expr.Expr, n: ufl.core.expr.Expr, d: ufl.core.expr.Expr
    ) -> ufl.core.expr.Expr:
        """Replace n / d with n * (1 / d)."""
        return n * (1 / d)
