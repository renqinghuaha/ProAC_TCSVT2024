# Exploring Prototype-Anchor Contrast for Semantic Segmentation
IEEE Transactions on Circuits and Systems for Video Technology 2024, [Paper](https://ieeexplore.ieee.org/document/10445499)

Abstract
---
Pixel-wise contrastive learning recently offers a new training paradigm in semantic segmentation by directly shaping the pixel embedding space. Compared with pixel-pixel contrast that often requires large memory and high computation cost, pixel-prototype contrast exploits the semantic correlations among pixels in a more efficient way by pulling positive pixel-prototype pairs close and pushing negative pairs apart. However, most existing work treats pixels as anchors to form contrast, either failing to capture the intra-class variance or introducing extra computational overhead. In this work, we propose Prototype-Anchor Contrast (ProAC), a novel prototypical contrastive learning paradigm that strengthens pixel-prototype associations in a simple yet effective fashion. First, ProAC pre-defines class prototypes (serving as cluster centroids) by exploiting the uniformity on the hypersphere in the feature space and thus requires no prototype updating during network optimization, which greatly simplifies the network training process. Second, by treating prototypes as anchors, ProAC builds a novel prototype-to-pixel learning path, where a large amount of negative pixels can naturally be generated to describe rich semantic information without relying on auxiliary sample augmentation techniques. Finally, as a plug-and-play regularization term, ProAC can be attached to most existing segmentation models and assist the network optimization by directly shaping the pixel embedding space. Extensive experiments on different benchmarks show that our ProAC brings an mIoU increase from 1.4% to 2.0% for fully-supervised models and from 0.9% to 6.0% for domain-adaptive models, respectively. It also leads to a gain of mIoU, ranging from 1.8% to 2.7% in more challenging cases, including different resolutions, diverse illuminations and masked scenarios.

Usage
---
This is an example of training UDA models using our ProAC loss.

- Prepare Datasets: [GTA5](https://download.visinf.tu-darmstadt.de/data/from_games/), [SYNTHIA](https://synthia-dataset.net/), [Cityscapes](https://www.cityscapes-dataset.com/), [ADE20K](https://groups.csail.mit.edu/vision/datasets/ADE20K/).

- Download the pretrained models, e.g., [FADA](https://github.com/JDAI-CV/FADA), [FDA](https://github.com/YanchaoYang/FDA), [ProDA](https://github.com/microsoft/ProDA), and generate pseudo labels.

- Train the UDA model using our ProAC loss
```
python train.py
```

---
We use [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) for training the fully-supervised models.

Citation
---
```
@article{ren2024exploring,
  title={Exploring Prototype-Anchor Contrast for Semantic Segmentation},
  author={Ren, Qinghua and Lu, Shijian and Mao, Qirong and Dong, Ming},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2024},
  publisher={IEEE}
}
```
Acknowledgments
---
This code is heavily borrowed from [CAG_UDA](https://github.com/RogerZhangzz/CAG_UDA).
