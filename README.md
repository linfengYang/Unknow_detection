# Unknown Load Detection for Non-Intrusive Load Monitoring 

## Overview

Paper Todo list

## Datasets

We evaluated the DC-LUNAR model using  two public datasets, PLAID2017 and WHITEDv1.1. To obtain these datasets, you can download as follow

### Manual download:

You can manually download the datasets using the provided link and place them into the pre-made directory.

PLAID2017 [PLAID 2017 (figshare.com)](https://figshare.com/articles/dataset/PLAID_2017/11605215?file=21003861)

WHITED [WHITED - Decentralized Information Systems and Data Management (tum.de)](https://www.cs.cit.tum.de/dis/resources/whited/)

## Setup

_Instructions refer to Unix-based systems (e.g. Linux, MacOS)._

This code has been tested with `Python 3.7` and `3.8`.

`pip install -r requirements.txt`

## Run

To see all command options with explanations, run: `python main.py --help`
In `main.py` you can select the datasets and modify the model parameters.
For example:

`python main.py --epochs 1500 `

## Results

### Only know appliances

#### WHITED

![image-20240928095952648](./Fig/WHITED.png)

#### PLAID2017

![image-20240928100108900](./Fig/Plaid2017.png)

### Consider the scenario of an unknown electrical appliance

detail in paper（TODO ）

### Consider the scenario of multiple unknown electrical appliance


#### PLAID2017 
consider (Compact fluorescent lamp, Fan, Vaccum) to be unknown appliances at the same time

take all 2,9 appliance identification results on 1(they are all unknow)

![multiple_plaid](./Fig/multiple_plaid.png)
#### WHITED
consider (Ap 5, Ap7, Ap11) to be unknown appliances at the same time

take all 5,11 appliance identification results on 7(they are all unknow)

![multiple_whited](./Fig/multiple_whited.png)

### Graph

T-sne for Origin sequence(2D)

![origin_2D](./Fig/origin_2D.png)

T-sne for Feature embedding sequence(2D)

![all_train_2D](./Fig/all_train_2D.png)

## Acknowledgments

**RoPE** [Rotary Position Embedding for Vision Transformer](https://github.com/naver-ai/rope-vit)
**Coordinate attention** [Coordinate Attention for Efficient Mobile Network Design](https://github.com/houqb/CoordAttention)
**ConvTran** [Transformers for Multivariate Time Series Classification](https://github.com/Navidfoumani/ConvTran)
**librosa** [A python package for music and audio analysis.](https://github.com/librosa/librosa)
**PyOD** [Python library for detecting anomalous/outlying objects in multivariate data](https://github.com/yzhao062/pyod)

Some parts of the Code are taken from the above repository