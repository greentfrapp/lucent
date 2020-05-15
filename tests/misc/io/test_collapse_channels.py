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