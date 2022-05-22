# RFNet for Incomplete/Missing Multi-modal Brain Tumor Segmentation
Official PyTorch implementation of [RFNet: Region-aware Fusion Network for Incomplete Multi-modal Brain Tumor Segmentation]([https://arxiv.org/abs/2203.03884](https://openaccess.thecvf.com/content/ICCV2021/papers/Ding_RFNet_Region-Aware_Fusion_Network_for_Incomplete_Multi-Modal_Brain_Tumor_Segmentation_ICCV_2021_paper.pdf)), ICCV2021.

## Results
### Brats2020

All missing and full-set situations (15 situations) are considered during testing. The average results are reported here. Please refer to our paper for more details.
| Method                      | Complete | Core | Enhancing | 
| --------------------------- | -------- | -------- | -------- |
| HeMIS                       |  75.10   |  65.45   |  47.73   |
| U-HVED                      |  81.24   |  67.19   |  48.55   | 
| RobustSeg                   |  84.17   |  73.45   |  55.49   |
| RFNet (Ours)                |  86.98   |  78.23   |  61.47   | 

Complete, Core and Enhancing denote the dice score (%) of the whole tumor, the tumor core and the enhancing tumor.

### Brats2018

Brats2018 contains three different training and test splits and the average results are reported here.
| Method                      | Complete | Core | Enhancing | 
| --------------------------- | -------- | -------- | -------- |
| HeMIS                       |  78.60   |  59.70   |  48.10   |
| U-HVED                      |  80.10   |  64.00   |  50.00   | 
| RobustSeg                   |  84.37   |  69.78   |  51.02   |
| RFNet (Ours)                |  85.67   |  76.53   |  57.12   | 

### Brats2015
| Method                      | Complete | Core | Enhancing | 
| --------------------------- | -------- | -------- | -------- |
| HeMIS                       |  68.22   |  54.07   |  43.86   |
| U-HVED                      |  81.57   |  64.68   |  56.76   | 
| RobustSeg                   |  84.45   |  69.19   |  57.33   |
| RFNet (Ours)                |  86.13    |  71.93  |  64.13   | 

###Checkpoints
| Brats2020 | Brats2018 split1 | Brats2018 split2 | Brats2018 split3 | Brats2015 |
|--------------------------- | -------- | -------- | -------- | -------- | 
|[model]() | [model]() | [model]() | [model]() | [model]() |
|[log]() | [log]() | [log]() | [log]() | [log]() |

## Installation
get dataset and environment [here]()
we use pytorch1.2.0 and cuda9.0.

## Usage
'''
bash job.sh
'''


## Citation
```bibtex
@inproceedings{ding2021rfnet,
  title={RFNet: Region-Aware Fusion Network for Incomplete Multi-Modal Brain Tumor Segmentation},
  author={Ding, Yuhang and Yu, Xin and Yang, Yi},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={3975--3984},
  year={2021}
}
```
