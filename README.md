## PointNetLK_Revisited
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[Xueqian Li](https://lilac-lee.github.io/), [Jhony Kaesemodel Pontes](https://jhonykaesemodel.com/), 
[Simon Lucey](https://www.adelaide.edu.au/directory/simon.lucey)

2021 Conference on Computer Vision and Pattern Recognition (CVPR) (**oral**)

arXiv link: https://arxiv.org/pdf/2008.09527v2.pdf


| ModelNet40 | 3DMatch | KITTI |
|:-:|:-:|:-:|
| <img src="imgs/modelnet_registration.gif" width="172" height="186"/>| <img src="imgs/3dmatch_registration.gif" width="190" height="186"/> | <img src="imgs/kitti_registration.gif" width="200" height="166"/> |

### Prerequisites
This code is based on PyTorch implementation, and tested on 1.0.0<=torch<=1.6.0. You may go to the PyTorch official site (https://pytorch.org/) to decide which torch/torchvision version is suitable for your system. You may also need to go to the tensorflow website (https://www.tensorflow.org/install) to download tensorboard. Other packages can be installed through,
```
pip install -r requirements.txt
```

### Demo Notebook


### Dataset
You may download dataset used in the paper from these websites.

| ModelNet40 | ShapeNet | 3DMatch | KITTI   |
|:-:|:-:|:-:|:-:|
| https://modelnet.cs.princeton.edu | https://shapenet.org | | |

### Training
```
python train.py
```

### Testing
```
python test.py
```

### Acknowledgement
This code is adapted from the original PointNetLK, https://github.com/hmgoforth/PointNetLK.



### Citation
If you find the project useful for your research, you may cite,
```

```
