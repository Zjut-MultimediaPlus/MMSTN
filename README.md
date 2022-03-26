# MMSTN: a Multi-Modal Spatial-Temporal Network for Tropical Cyclone Short-Term Prediction
## Introduction
Contribution:

1. The MMSTN proposed in this paper can receive both the trajectory modal data and the intensity modal data of a TC and extract the relationship between those two modals. In addition, we propose a Feature Updating Mechanism (FUM) in this framework to alleviate the forgetting problem of the recurrent neural network. These are beneficial for improving the precision of TC trajectory and intensity prediction.
2. The MMSTN can not only predict a TC's central pressure, winds, and the location of its center, but also forecast a ***cone of probability*** of TC through its GAN module. Furthermore, compared to traditional TC prediction methods, the MMSTN can be trained with data that are easier to obtain, and it yield predictive results more quickly.
3. To prove the effectiveness of the MMSTN, we evaluated it on the data from the years 2017,2018, and 2019 on the CMA Tropical Cyclone Best Track Dataset. The experimental results show that our method obtained significant improvement compared with other state-of-the-art deep learning methods.

The explanation of ***cone of probability***:

When we use MMSTN to make a prediction of TC, we will generate k possible tendencies. By calculating these k possible tendencies, we obtain the ***cone of probability***. Like the figure showing below: 

![***cone of probability***](https://github.com/Zjut-MultimediaPlus/MMSTN/blob/main/Data/example24.png)

As for the calculation of evaluation index, we choose the **best prediction** through these k possible tendencies (including every time points) as our final prediction.

This is the source code of MMSTN.
## Requirements 
* python 3.7.7
* Pytorch 1.10.0
* CUDA 10.2
```python
##Install required libraries##
pip install -r requirements.txt
```
## Train
```python
##before train##
python -m visdom.server
##custom train##
python train.py
```
## Test
```python
## test on data of the year 2019##
python evaluate_model_ME.py --dset_type test2019
```
## Training new models
Instructions for training new models can be [found here](https://github.com/Zjut-MultimediaPlus/MMSTN/blob/main/TRAINING.md).

## The data we used
We used two open access dataset: [the CMA Tropical Cyclone Best Track Dataset](https://tcdata.typhoon.org.cn/en/zjljsjj_sm.html) 
and the results of [the CMO's tropical cyclone predictions](http://typhoon.nmc.cn/web.html).

To facilitate our readers, we arrange these data and upload them in [Data](https://github.com/Zjut-MultimediaPlus/MMSTN/tree/main/Data)

If you are interesting in these data, you can click [the CMA Tropical Cyclone Best Track Dataset](https://tcdata.typhoon.org.cn/en/zjljsjj_sm.html) and
[the CMO's tropical cyclone data](http://typhoon.nmc.cn/web.html) to obtain more details. 



## Note
Our codes were modified from the implementation of ["Social GAN: Socially Acceptable Trajectories with Generative Adversarial Networks"](https://github.com/agrimgupta92/sgan). Please cite the two papers (SGAN and MMSTN) when you use the codes.
## Citing SGAN & MMSTN
```
@inproceedings{gupta2018social,
  title={Social GAN: Socially Acceptable Trajectories with Generative Adversarial Networks},
  author={Gupta, Agrim and Johnson, Justin and Fei-Fei, Li and Savarese, Silvio and Alahi, Alexandre},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  number={CONF},
  year={2018}
}
```

```
@article{https://doi.org/10.1029/2021GL096898,
author = {Huang, Cheng and Bai, Cong and Chan, Sixian and Zhang, Jinglin},
title = {MMSTN: A Multi-Modal Spatial-Temporal Network for Tropical Cyclone Short-Term Prediction},
journal = {Geophysical Research Letters},
volume = {49},
number = {4},
pages = {e2021GL096898},
doi = {https://doi.org/10.1029/2021GL096898},
url = {https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2021GL096898},
year = {2022}
}
```
