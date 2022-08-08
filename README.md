# CLSR: Disentangling Long and Short-Term Interests for Recommendation

This is the official implementation of our WWW'22 paper:  

Yu Zheng, Chen Gao, Jianxin Chang, Yanan Niu, Yang Song, Depeng Jin, Yong Li, **Disentangling Long and Short-Term Interests for Recommendation**, In Proceedings of the Web Conference 2022.

The code is tested under a Linux desktop with TensorFlow 1.15.2 and Python 3.6.8.

Please cite our paper if you use this repository.
```
@inproceedings{zheng2022disentangling,
  title={Disentangling Long and Short-Term Interests for Recommendation},
  author={Zheng, Yu and Gao, Chen and Chang, Jianxin and Niu, Yanan and Song, Yang and Jin, Depeng and Li, Yong},
  booktitle={Proceedings of the ACM Web Conference 2022},
  pages={2256--2267},
  year={2022}
}
```

## Data Pre-processing


Run the script `reco_utils/dataset/sequential_reviews.py` to generate the data for training and evaluation.

Details of the data are available at [Data](./tests/resources/deeprec/sequential/README.md).


  

## Model Training

Use the following commands to train a CLSR model on `Taobao` dataset: 

```
cd ./examples/00_quick_start/
python sequential.py --dataset taobao
```

or on `Kuaishou` dataset:

```
cd ./examples/00_quick_start/
python sequential.py --dataset kuaishou
``` 


## Pretrained Model Evaluation

We provide a pretrained model for the `Taobao` dataset at [Model](./examples/00_quick_start/CLSR/taobao-clsr-debug/README.md).

```
cd ./examples/00_quick_start/
python sequential.py --dataset taobao --only_test
```

The performance of the provided pretrained model is as follows:
| AUC | GAUC | MRR | NDCG@2 |
| ---- | ---- | ---- | ---- |
| 0.8954 | 0.8936 | 0.4384 | 0.3807 |


## Note

The implemention is based on *[Microsoft Recommender](https://github.com/microsoft/recommenders)*.
