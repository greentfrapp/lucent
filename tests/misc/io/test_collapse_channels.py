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

import pytest

import numpy as np
import lucent.misc.io.collapse_channels as collapse_channels
import IPython.display


def test_hue_to_rgb():
	assert (collapse_channels.hue_to_rgb(0) == [1, 0, 0]).all()
	assert (collapse_channels.hue_to_rgb(120) == [0, 1, 0]).all()
	assert (collapse_channels.hue_to_rgb(240) == [0, 0, 1]).all()


def test_sparse_channels_to_rgb():
	sparse = np.array([
		[1, 0, 0],
		[0, 1, 0],
		[0, 0, 1]
	])
	print(collapse_channels.sparse_channels_to_rgb(sparse))
	assert np.allclose(collapse_channels.sparse_channels_to_rgb(sparse), sparse, atol=1e-4)
