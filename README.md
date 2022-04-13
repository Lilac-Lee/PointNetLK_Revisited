## [PointNetLK Revisited](https://openaccess.thecvf.com/content/CVPR2021/papers/Li_PointNetLK_Revisited_CVPR_2021_paper.pdf)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[Xueqian Li](https://lilac-lee.github.io/), [Jhony Kaesemodel Pontes](https://jhonykaesemodel.com/), 
[Simon Lucey](https://www.adelaide.edu.au/directory/simon.lucey)

Will appear on 2021 Conference on Computer Vision and Pattern Recognition (CVPR) (**oral**)

arXiv link: https://arxiv.org/pdf/2008.09527.pdf


| ModelNet40 | 3DMatch | KITTI |
|:-:|:-:|:-:|
| <img src="imgs/modelnet_registration.gif" width="172" height="186"/>| <img src="imgs/3dmatch_registration.gif" width="190" height="186"/> | <img src="imgs/kitti_registration.gif" width="200" height="166"/> |

### Prerequisites
This code is based on PyTorch implementation, and tested on 1.0.0<=torch<=1.6.0. You may go to the PyTorch official site (https://pytorch.org/) to decide which torch/torchvision version is suitable for your system. You may also need to go to the tensorflow website (https://www.tensorflow.org/install) to download tensorboard. Other packages can be installed through,
```
pip install -r requirements.txt
```
**NOTE**: we use open3d interactive visualization window in the jupyter notebook, which is only available for open3d>=0.13.0. You may need to wait for a few seconds for the interactive visualization to show.


### Demo Notebook
We provide a jupyter notebook (you may need to install the jupyter notebook) of toy example in the demo folder. 
You can run ./demo/test_toysample.ipynb to play with a point cloud pairs from 3DMatch dataset.

Here is a short video showing how to run the demo.
|Demo notebook|
|:-:|
|<img src="imgs/code_demo.gif" width="600" height="330"/>|


### Dataset
You may download dataset used in the paper from these websites.

| ModelNet40 | ShapeNet | KITTI   |
|:-:|:-:|:-:|
| https://modelnet.cs.princeton.edu | https://shapenet.org | http://www.cvlibs.net/datasets/kitti/eval_odometry.php |

We used 3DMatch dataset prepared by authors of Deep Global Registration, you may download using this script provided by them, https://github.com/chrischoy/DeepGlobalRegistration/blob/master/scripts/download_3dmatch.sh.

After you download the dataset, you can create a symbolic link in the ./dataset folder as ```./dataset/ModelNet``` and ```./dataset/ThreeDMatch```.


### Notation on voxelization
In the current setting, the voxelization for overlapped source and target point clouds still relies on "registered point clouds" (such as results from a global registration). The code for voxelization after "transformation" is updated by setting argument ```--voxel_after_transf``` to **True**.
In the real-world application, our algorithm would be better suited for local registration problem.


### Training
```
python train.py
```

### Evaluation
```
python test.py
```


### Pre-Trained Models
You can find the pre-trained model in ***logs/model_trained_on_ModelNet40_model_best.pth***.
This model is trained on ModelNet40 dataset, and is used to produce most results (except results of trained/tested on 3DMatch dataset) in the paper.


### Acknowledgement
This code is mostly adapted from the original PointNetLK, https://github.com/hmgoforth/PointNetLK.

The 3DMatch data loader code was adapted from the Deep Global Registration, https://github.com/chrischoy/DeepGlobalRegistration.

Part of the data voxelization code was adapted from the SECOND, https://github.com/traveller59/second.pytorch.

Part of the metric computaion code was adapted from the Deep Closest Point, https://github.com/WangYueFt/dcp.


### Contributing
If you find the project useful for your research, you may cite,
```
@InProceedings{Li_2021_CVPR,
    author    = {Li, Xueqian and Pontes, Jhony Kaesemodel and Lucey, Simon},
    title     = {PointNetLK Revisited},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {12763-12772}
}
```
