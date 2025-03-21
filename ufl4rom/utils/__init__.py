# Copyright (C) 2021-2025 by the ufl4rom authors
#
# This file is part of ufl4rom.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""ufl4rom utils module."""

from ufl4rom.utils.backends import DolfinxConstant, DolfinxFunction, FiredrakeConstant, FiredrakeFunction
from ufl4rom.utils.expand_sum import expand_sum
from ufl4rom.utils.name import name
from ufl4rom.utils.named_coefficient import NamedCoefficient
from ufl4rom.utils.named_constant import DolfinxNamedConstant, FiredrakeNamedConstant, NamedConstant, NamedConstantValue
from ufl4rom.utils.rewrite_quotients import rewrite_quotients
