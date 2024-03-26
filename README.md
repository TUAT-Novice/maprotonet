# MAProtoNet
Welcome to the official implementation of MAProtoNet (Multi-scale Attentive Prototypical Network).

MAProtoNet is an interpretable 3D MRI model for brain tumor, training on the public BraTS datasets. We extend the XProtoNet and MProtoNet by introducing the novel 
<!-- quadruplet attention layers -->
as well as the 
<!-- multi-scale module. -->

[[Paper]]()
[[Code]](https://github.com/TUAT-Novice/maprotonet)

<img src="figures/framework.png" alt="Framework of MAProtoNet" width="901.8" height="441.45">

Please check the following sections for more details:

* [Directory Structure](#Structure)
* Environment Configurations
* Datasets
* Experiments
* [Results](#Results)
* [Acknowledgment](Acknowledgmen)
* Citation

If necessary, please check further details from our paper "MAProtoNet: A Multi-scale Attentive Interpretable Prototypical Network for 3D Magnetic Resonance Imaging Brain Tumor Classification", or contact us through email: s237857s@st.go.tuat.ac.jp




## Directory Structure {#Structure}
The directories of this repository are established as below:
* **src/** source code for our MAProtoNet
  * tumor_cls.py
* **figures/** 
* **data/** code for BraTS pre-processing 
* **readme.md**

requirements.txt: Environment configurations






## Results{#Results}
| Methods           | BAC | AP | IDS |
|:-----------------:|:-----:|:----:|:-----:|
| CNN               | 内容1 | 内容2 | 内容3 |
| ProtoPNet         | 内容4 | 内容5 | 内容6 |
| XProtoNet         | 内容4 | 内容5 | 内容6 |
| MProtoNet         | 内容4 | 内容5 | 内容6 |
| MAProtoNet (ours) | 内容4 | 内容5 | 内容6 |

 [//]: # (Results) 




## Acknowledgment  {#Acknowledgment}
This repository is established upon the source code of MProtoNet, from https://github.com/aywi/mprotonet developed by Yuanyuan Wei, Roger Tam and Xiaoying Tang. We appreciate their work.
