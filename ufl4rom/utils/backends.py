# Copyright (C) 2021-2022 by the ufl4rom authors
#
# This file is part of ufl4rom.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

try:
    import dolfin  # noqa: F401
except ImportError:
    class DolfinConstant(object):
        pass

    class DolfinFunction(object):
        pass
else:
    from dolfin import Constant as DolfinConstant, Function as DolfinFunction  # noqa: F401

try:
    import dolfinx  # noqa: F401
except ImportError:
    class DolfinxConstant(object):
        pass

    class DolfinxFunction(object):
        pass
else:
    from dolfinx.fem import Constant as DolfinxConstant, Function as DolfinxFunction  # noqa: F401

try:
    import firedrake  # noqa: F401
except ImportError:
    class FiredrakeConstant(object):
        pass

    class FiredrakeFunction(object):
        pass
else:
    from firedrake import Constant as FiredrakeConstant, Function as FiredrakeFunction  # noqa: F401
