# Copyright 2018 The Lucid Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Utility functions for Objectives."""

from __future__ import absolute_import, division, print_function


def _make_arg_str(arg):
    arg = str(arg)
    too_big = len(arg) > 15 or "\n" in arg
    return "..." if too_big else arg


def _extract_act_pos(acts, x=None, y=None):
    shape = acts.shape
    x = shape[2] // 2 if x is None else x
    y = shape[3] // 2 if y is None else y
    return acts[:, :, y:y+1, x:x+1]


def _T_handle_batch(T, batch=None):
    def T2(name):
        t = T(name)
        if isinstance(batch, int):
            return t[batch:batch+1]
        else:
            return t
    return T2
