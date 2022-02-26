# CLSR: Disentangling Long and Short-Term Interests for Recommendation

This is the official implementation of our WWW'22 paper:  

Yu Zheng, Chen Gao, Jianxin Chang, Yanan Niu, Yang Song, Depeng Jin, Yong Li, **Disentangling Long and Short-Term Interests for Recommendation**, In Proceedings of the Web Conference 2022.

The code is tested under a Linux desktop with TensorFlow 1.15.2 and Python 3.6.8.



## Data Pre-processing


Run the script `reco_utils/dataset/sequential_reviews.py` to generate the data for training and evaluation.


  

## Model Training

Use the following command to train a CLSR model on `Taobao` dataset: 

```
python examples/00_quick_start/sequential.py --dataset taobao
```

or on `Kuaishou` dataset:

```
python examples/00_quick_start/sequential.py --dataset kuaishou
``` 

## Note

The implemention is based on *[Microsoft Recommender](https://github.com/microsoft/recommenders)*.
