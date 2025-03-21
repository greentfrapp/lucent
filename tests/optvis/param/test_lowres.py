# Copyright 2020 The Lucent Authors. All Rights Reserved.
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

from __future__ import absolute_import, division, print_function

from lucent.optvis import param


def test_lowres():
	# Without offset
    params, image_f = param.lowres_tensor((6, 3, 128, 128), (1, 3, 64, 64))
    assert params[0].shape == (1, 3, 64, 64)
    assert image_f().shape == (6, 3, 128, 128)
    # With offset as scalar
    params, image_f = param.lowres_tensor((6, 3, 128, 128), (1, 3, 64, 64), offset=5)
    assert params[0].shape == (1, 3, 64, 64)
    assert image_f().shape == (6, 3, 128, 128)
    # With offset as list
    params, image_f = param.lowres_tensor((6, 3, 128, 128), (1, 3, 64, 64), offset=[1, False, 1, True])
    assert params[0].shape == (1, 3, 64, 64)
    assert image_f().shape == (6, 3, 128, 128)
