# Copyright (C) 2021 by the ufl4rom authors
#
# This file is part of ufl4rom.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from ufl import Constant
from ufl4rom.utils.backends import DolfinConstant, DolfinxConstant, FiredrakeConstant


class NamedConstant(Constant):
    def __init__(self, name, domain, shape=(), count=None):
        Constant.__init__(self, domain, shape, count)
        self._name = name

        # Neglect the count argument when preparing the representation string, as we aim to
        # get a representation which is independent on the internal counter
        self._repr = "NamedConstant({}, {}, {})".format(
            repr(self._name), repr(self._ufl_domain), repr(self._ufl_shape))

    def __str__(self):
        return self._name


class DolfinNamedConstant(DolfinConstant):
    def __init__(self, name, value, cell=None):
        DolfinConstant.__init__(self, value, cell, name)


class DolfinxNamedConstant(DolfinxConstant):
    def __init__(self, name, value, domain):
        DolfinxConstant.__init__(self, value, domain)
        self._name = name

        # Neglect the count argument when preparing the representation string, as we aim to
        # get a representation which is independent on the internal counter
        self._repr = "DolfinxNamedConstant({}, {}, {})".format(
            repr(self._name), repr(self.value), repr(self._ufl_domain))

    def __str__(self):
        return self._name


class FiredrakeNamedConstant(FiredrakeConstant):
    def __init__(self, name, value, domain=None):
        FiredrakeConstant.__init__(self, value, domain)
        self._name = name

        # Neglect the count argument when preparing the representation string, as we aim to
        # get a representation which is independent on the internal counter
        self._repr = "FiredrakeNamedConstant({}, {}, {})".format(
            repr(self._name), repr(self.values()), repr(self._ufl_function_space._ufl_domain))

    def __str__(self):
        return self._name
