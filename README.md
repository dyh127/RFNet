# RFNet for Incomplete/Missing Multi-modal Brain Tumor Segmentation
Official PyTorch implementation of [RFNet: Region-aware Fusion Network for Incomplete Multi-modal Brain Tumor Segmentation]([https://arxiv.org/abs/2203.03884](https://openaccess.thecvf.com/content/ICCV2021/papers/Ding_RFNet_Region-Aware_Fusion_Network_for_Incomplete_Multi-Modal_Brain_Tumor_Segmentation_ICCV_2021_paper.pdf)), ICCV2021.

## Results
### Brats2020

All missing and full-set situations (15 situations) are considered during testing. The average results are reported here.
| Method                      | Complete | Core | Enhancing | 
| --------------------------- | -------- | -------- | -------- |
| HeMIS                       |  75.10   |  65.45   |  47.73   |
| U-HVED                      |  81.24   |  67.19   |  48.55   | 
| RobustSeg                   |  84.17   |  73.45   |  55.49   |
| RFNet (Ours)                |  86.98   |  78.23   |  61.47   |  
