# MMSTN: a Multi-Modal Spatial-Temporal Network for Tropical Cyclone Short-Term Prediction
## Introduction
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
@xxx{xxx,
  title={Social GAN: Socially Acceptable Trajectories with Generative Adversarial Networks},
  author={Cheng Huang,Cong Bai, Sixian Chan, Jinglin Zhang},
  booktitle={xxx},
  number={xxx},
  year={xxx}
}
```
