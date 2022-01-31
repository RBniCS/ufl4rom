# Copyright (C) 2021-2022 by the ufl4rom authors
#
# This file is part of ufl4rom.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Import specialization of UFL classes from dolfin, dolfinx and firedrake backends."""

try:
    import dolfin  # noqa: F401
except ImportError:
    class DolfinConstant(object):
        """Mock dolfin.Constant class."""

        pass

    class DolfinFunction(object):
        """Mock dolfin.Function class."""

        pass
else:
    from dolfin import Constant as DolfinConstant, Function as DolfinFunction  # noqa: F401, I2041, I2045

try:
    import dolfinx  # noqa: F401
except ImportError:
    class DolfinxConstant(object):
        """Mock dolfinx.fem.Constant class."""

        pass

    class DolfinxFunction(object):
        """Mock dolfinx.fem.Function class."""

        pass
else:
    from dolfinx.fem import Constant as DolfinxConstant, Function as DolfinxFunction  # noqa: F401, I2041, I2045

try:
    import firedrake  # noqa: F401
except ImportError:
    class FiredrakeConstant(object):
        """Mock firedrake.Constant class."""

        pass

    class FiredrakeFunction(object):
        """Mock firedrake.Function class."""

        pass
else:
    from firedrake import Constant as FiredrakeConstant, Function as FiredrakeFunction  # noqa: F401, I2041, I2045
