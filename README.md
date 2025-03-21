![](https://github.com/greentfrapp/lucent/raw/master/images/lucent_header.jpg)

# Lucent

<!--*It's still magic even if you know how it's done. GNU Terry Pratchett*-->

[![Travis build status](https://img.shields.io/travis/greentfrapp/lucent.svg)](https://travis-ci.org/greentfrapp/lucent)
[![Code coverage](https://img.shields.io/coveralls/github/greentfrapp/lucent.svg)](https://coveralls.io/github/greentfrapp/lucent)

*PyTorch + Lucid = Lucent*

The wonderful [Lucid](https://github.com/tensorflow/lucid) library adapted for the wonderful PyTorch!

**Lucent is not affiliated with Lucid or OpenAI's Clarity team, although we would love to be!**
Credit is due to the original Lucid authors, we merely adapted the code for PyTorch and we take the blame for all issues and bugs found here.

# Usage

Lucent is still in pre-alpha phase and can be installed locally with the following command:

```
pip install torch-lucent
```

In the spirit of Lucid, get up and running with Lucent immediately, thanks to Google's [Colab](https://colab.research.google.com/notebooks/welcome.ipynb)! 

You can also clone this repository and run the notebooks locally with [Jupyter](http://jupyter.org/install.html).

## Quickstart

```python
import torch

from lucent.optvis import render
from lucent.modelzoo import inceptionv1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = inceptionv1(pretrained=True)
model.to(device).eval()

render.render_vis(model, "mixed4a:476")
```

## Tutorials

<a href="https://colab.research.google.com/github/greentfrapp/lucent-notebooks/blob/master/notebooks/tutorial.ipynb">
<img src="https://github.com/greentfrapp/lucent-notebooks/raw/master/images/tutorial_card.jpg" width="500" alt=""></img></a>

<a href="https://colab.research.google.com/github/greentfrapp/lucent-notebooks/blob/master/notebooks/modelzoo.ipynb"><img src="https://github.com/greentfrapp/lucent-notebooks/raw/master/images/modelzoo_card.jpg" width="500" alt=""></img></a>

## Other Notebooks

Here, we have tried to recreate some of the Lucid notebooks! You can also check out the [lucent-notebooks](https://github.com/greentfrapp/lucent-notebooks) repo to clone all the notebooks.

<a href="https://colab.research.google.com/github/greentfrapp/lucent-notebooks/blob/master/notebooks/diversity.ipynb"><img src="https://github.com/greentfrapp/lucent-notebooks/raw/master/images/diversity_card.jpg" width="500" alt=""></img></a>

<a href="https://colab.research.google.com/github/greentfrapp/lucent-notebooks/blob/master/notebooks/neuron_interaction.ipynb"><img src="https://github.com/greentfrapp/lucent-notebooks/raw/master/images/neuron_interaction_card.jpg" width="500" alt=""></img></a>

<a href="https://colab.research.google.com/github/greentfrapp/lucent-notebooks/blob/master/notebooks/feature_inversion.ipynb">
<img src="https://github.com/greentfrapp/lucent-notebooks/raw/master/images/feature_inversion_card.jpg" width="500" alt=""></img>
</a>

<a href="https://colab.research.google.com/github/greentfrapp/lucent-notebooks/blob/master/notebooks/style_transfer.ipynb">
<img src="https://github.com/greentfrapp/lucent-notebooks/raw/master/images/style_transfer_card.jpg" width="500" alt=""></img>
</a>

<a href="https://colab.research.google.com/github/greentfrapp/lucent-notebooks/blob/master/notebooks/activation_grids.ipynb">
<img src="https://github.com/greentfrapp/lucent-notebooks/raw/master/images/activation_grids_card.jpg" width="500" alt=""></img>
</a>

# Recommended Readings

* [Feature Visualization](https://distill.pub/2017/feature-visualization/)
* [The Building Blocks of Interpretability](https://distill.pub/2018/building-blocks/)
* [Using Artiﬁcial Intelligence to Augment Human Intelligence](https://distill.pub/2017/aia/)
* [Visualizing Representations: Deep Learning and Human Beings](http://colah.github.io/posts/2015-01-Visualizing-Representations/)
* [Differentiable Image Parameterizations](https://distill.pub/2018/differentiable-parameterizations/)
* [Activation Atlas](https://distill.pub/2019/activation-atlas/)

## Related Talks
* [Lessons from a year of Distill ML Research](https://www.youtube.com/watch?v=jlZsgUZaIyY) (Shan Carter, OpenVisConf)
* [Machine Learning for Visualization](https://www.youtube.com/watch?v=6n-kCYn0zxU) (Ian Johnson, OpenVisConf)

# Slack

Check out `#proj-lucid` and `#circuits` on the [Distill slack](http://slack.distill.pub)!

# Additional Information

## License and Disclaimer

You may use this software under the Apache 2.0 License. See [LICENSE](https://github.com/greentfrapp/lucent/blob/master/LICENSE).
