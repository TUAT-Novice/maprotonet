# MAProtoNet
Welcome to the official implementation of MAProtoNet (Multi-scale Attentive Prototypical Network).

MAProtoNet is an interpretable 3D MRI model for brain tumor, training on the public BraTS datasets. We improve the localization capability of the prototypical part network by introducing the novel quadruplet attention layers as well as the multi-scale module.

[[Paper]](https://export.arxiv.org/abs/2404.08917)
[[Code]](https://github.com/TUAT-Novice/maprotonet)

<img src="figures/framework.png" alt="Framework of MAProtoNet" width="901.8" height="441.45">

Please check the following sections for more details:

|Repository Directory|
| --- |
| [Directory Structure](#Structure) |
| [Environment Configurations](#Environment) |
| [Datasets](#Datasets) |
| [Experiments Reproduction](#Experiments) |
| [Results](#Results) |
| [Acknowledgment](#Acknowledgment) |
| [Citation](#Citation) |

If necessary, please check further details from our paper "MAProtoNet: A Multi-scale Attentive Interpretable Prototypical Network for 3D Magnetic Resonance Imaging Brain Tumor Classification", or contact us through email: s237857s@st.go.tuat.ac.jp




## Directory Structure <a id="Structure"></a>
The directories of this repository are established as below:
* **tumor_cls.py** (main function for running our experiments)
* **train.py** (code for training and evaluation)
* **models/** (code for our MAProtoNet model)
* **utils/** (code for utility functions)
* **data/** (code for BraTS datasets preparing)
* **figures/** (images for this repository)
* **requirements.txt** (environment configurations)
* **readme.md**

Source code for data pre-processing:

- data
  - data1.py
  - data2.py
- model
  - cnn.py


## Environment Configurations <a id="Environment"></a>




## Datasets <a id="Datasets"></a>




## Experiments Reproduction <a id="Experiments"></a>




## Results <a id="Results"></a>
| Methods           | BAC | AP | IDS |
|:-----------------:|:-----:|:----:|:-----:|
| CNN               | 内容1 | 内容2 | 内容3 |
| ProtoPNet         | 内容4 | 内容5 | 内容6 |
| XProtoNet         | 内容4 | 内容5 | 内容6 |
| MProtoNet         | 内容4 | 内容5 | 内容6 |
| MAProtoNet (ours) | 内容4 | 内容5 | 内容6 |




## Acknowledgment <a id="Acknowledgment"></a>
This repository is established upon the source code of MProtoNet, from https://github.com/aywi/mprotonet developed by Yuanyuan Wei, Roger Tam and Xiaoying Tang. We appreciate their work.





## Citation <a id="Citation"></a>

