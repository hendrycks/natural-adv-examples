# Natural Adversarial Examples

We introduce [natural adversarial examples](https://arxiv.org/abs/1907.07174) -- real-world, unmodified, and naturally occurring examples that cause machine learning model performance to significantly degrade.

__[Download the natural adversarial example dataset ImageNet-A for image classifiers here](https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar).__

__[Download the natural adversarial example dataset ImageNet-A for out-of-distribution detectors here](https://people.eecs.berkeley.edu/~hendrycks/imagenet-o.tar).__

<img align="center" src="examples.png" width="400">
Natural adversarial examples from ImageNet-A and ImageNet-O. The black text is the actual class, and
the red text is a ResNet-50 prediction and its confidence. ImageNet-A contains images that classifiers should be
able to classify, while ImageNet-O contains anomalies of unforeseen classes which should result in low-confidence
predictions. ImageNet-1K models do not train on examples from “Photosphere” nor “Verdigris” classes, so these images
are anomalous. Many natural adversarial examples lead to wrong predictions, despite having no adversarial modifications as they are examples which occur naturally.

## Citation

If you find this useful in your research, please consider citing:

    @article{hendrycks2019nae,
      title={Natural Adversarial Examples},
      author={Dan Hendrycks and Kevin Zhao and Steven Basart and Jacob Steinhardt and Dawn Song},
      journal={arXiv preprint arXiv:1907.07174},
      year={2019}
    }
