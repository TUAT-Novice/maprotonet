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

If necessary, please check further details from our paper "MAProtoNet: A Multi-scale Attentive Interpretable Prototypical Part Network for 3D Magnetic Resonance Imaging Brain Tumor Classification", or contact us through email: s237857s@st.go.tuat.ac.jp




## Directory Structure <a id="Structure"></a>
The directories of this repository are established as below:
* **tumor_cls.py** (main function for running our experiments)
* **train.py**  (code for training and evaluation)
* **run.sh** (shell to run the experiments)
* **models/**  (code for our MAProtoNet model)
* **utils/**  (code for utility functions)
* **data/**  (code for BraTS datasets preparing)
* **figures/**  (images for this repository)
* **requirements.txt**  (environment configurations)
* **readme.md**


## Environment Configurations <a id="Environment"></a>


## Datasets <a id="Datasets"></a>
We employ [BraTS2018](https://www.med.upenn.edu/sbia/brats2018/), [BraTS2019](https://www.med.upenn.edu/cbica/brats-2019/), [BraTS2020](https://www.med.upenn.edu/cbica/brats2020/) datasets in our experiments. After applying for and downloading the datasets, for BraTS2018 dataset, please run data/get_mapping.py to generate name_mapping.csv for training.


## Experiments Reproduction <a id="Experiments"></a>




## Results <a id="Results"></a>
| Methods           | BAC | AP | IDS |
|:-----------------:|:-----:|:----:|:-----:|
| CNN               | 85.5 | 10.7 | 13.8 |
| ProtoPNet         | 84.3 | 11.8 | 24.2 |
| XProtoNet         | 84.7 | 16.9 | 16.3 |
| MProtoNet         | 85.8 | 81.2 | 6.2 |
| MAProtoNet (ours) | 86.7 | 85.8 | 6.2 |




## Acknowledgment <a id="Acknowledgment"></a>
This repository is established upon the source code of MProtoNet, from https://github.com/aywi/mprotonet developed by Yuanyuan Wei, Roger Tam and Xiaoying Tang. We appreciate their work.





## Citation <a id="Citation"></a>

