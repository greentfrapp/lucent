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
import pytest

import numpy as np
from lucent.misc.channel_reducer import ChannelReducer


def test_reduction_alg_name():
    with pytest.raises(ValueError):
        channel = ChannelReducer(reduction_alg="wrong")
    names = ["NMF", "PCA", "FastICA"]
    for name in names:
        channel = ChannelReducer(reduction_alg=name)


def test_n_components():
    with pytest.raises(ValueError):
        channel = ChannelReducer(n_components=None)
    with pytest.raises(ValueError):
        channel = ChannelReducer(n_components="string")
    with pytest.raises(ValueError):
        channel = ChannelReducer(n_components=0)
    channel = ChannelReducer(n_components=1)
    channel = ChannelReducer(n_components=3)


def test_fit_and_transform():
    nb_examples = 4
    height = 8
    width = 8
    feature_maps = 16
    n_components = 5
    np.random.seed(42)
    acts = np.random.uniform(size=(nb_examples, height, width, feature_maps))

    # fit then transform
    channel_reducer = ChannelReducer(n_components=n_components)
    channel_reducer.fit(acts)
    assert channel_reducer._is_fit
    assert channel_reducer.transform(acts).shape == (
        nb_examples,
        height,
        width,
        n_components,
    )
    # fit_transform
    channel_reducer = ChannelReducer(n_components=n_components)
    assert channel_reducer.fit_transform(acts).shape == (
        nb_examples,
        height,
        width,
        n_components,
    )

    # wrong shape for transform
    channel_reducer = ChannelReducer(n_components=n_components)
    channel_reducer.fit(acts)
    with pytest.raises(ValueError):
        acts_ = np.random.uniform(size=(nb_examples, height, width, 5))
        channel_reducer.transform(acts_)
    with pytest.raises(ValueError):
        acts_ = np.random.uniform(size=(nb_examples, height, width))
        channel_reducer.transform(acts_)

    # equivalence between fit_transform and fit then transform
    channel_reducer = ChannelReducer(n_components=n_components)
    y1 = channel_reducer.fit_transform(acts)
    channel_reducer = ChannelReducer(n_components=n_components)
    channel_reducer.fit(acts)
    y2 = channel_reducer.transform(acts)
    assert np.allclose(y1, y2, atol=1e-2)


def test_call():
    nb_examples = 4
    height = 8
    width = 8
    feature_maps = 16
    n_components = 5
    np.random.seed(42)
    acts = np.random.uniform(size=(nb_examples, height, width, feature_maps))

    channel_reducer = ChannelReducer(n_components=n_components)
    channel_reducer.fit(acts)
    y = channel_reducer(acts)
    assert np.allclose(y, channel_reducer.transform(acts), atol=1e-2)

    channel_reducer = ChannelReducer(n_components=n_components)
    y = channel_reducer(acts)
    assert np.allclose(y, channel_reducer.transform(acts), atol=1e-2)


def test_get_attr():
    nb_examples = 4
    height = 8
    width = 8
    feature_maps = 16
    n_components = 5
    np.random.seed(42)
    acts = np.random.uniform(size=(nb_examples, height, width, feature_maps))
    channel_reducer = ChannelReducer(n_components=n_components)
    channel_reducer.fit(acts)
    assert channel_reducer.components.shape == (n_components, feature_maps)
