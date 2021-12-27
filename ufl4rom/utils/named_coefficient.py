# Copyright (C) 2021 by the ufl4rom authors
#
# This file is part of ufl4rom.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import re
from ufl import Coefficient


class NamedCoefficient(Coefficient):
    def __init__(self, name, function_space, count=None):
        Coefficient.__init__(self, function_space, count)
        self._name = name

        # Neglect the count argument when preparing the representation string, as we aim to
        # get a representation which is independent on the internal counter
        self._repr = "NamedCoefficient({}, {})".format(
            repr(self._name), repr(self._ufl_function_space))
        self._repr = re.sub(" +", " ", self._repr)
        self._repr = re.sub(r"\[ ", "[", self._repr)
        self._repr = re.sub(r" \]", "]", self._repr)

    def __str__(self):
        return self._name
