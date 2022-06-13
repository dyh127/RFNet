# RFNet for Incomplete/Missing Multi-modal Brain Tumor Segmentation
Official implementation of [RFNet: Region-aware Fusion Network for Incomplete Multi-modal Brain Tumor Segmentation](https://openaccess.thecvf.com/content/ICCV2021/papers/Ding_RFNet_Region-Aware_Fusion_Network_for_Incomplete_Multi-Modal_Brain_Tumor_Segmentation_ICCV_2021_paper.pdf)), ICCV2021.

## Results
### Brats2020

All missing and full-set situations (15 situations) are considered during testing. The average results are reported here. Please refer to our paper for more details.
| Method                      | Complete | Core | Enhancing | 
| --------------------------- | -------- | -------- | -------- |
| HeMIS                       |  75.10   |  65.45   |  47.73   |
| U-HVED                      |  81.24   |  67.19   |  48.55   | 
| RobustSeg                   |  84.17   |  73.45   |  55.49   |
| RFNet (Ours)                |  86.98   |  78.23   |  61.47   | 

Complete, Core and Enhancing denote the dice score (%) of the whole tumor, the tumor core and the enhancing tumor, respectively.

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

### Checkpoints and logs
| Brats2020 | Brats2018 split1 | Brats2018 split2 | Brats2018 split3 | Brats2015 |
|--------------------------- | -------- | -------- | -------- | -------- | 
|[model](https://drive.google.com/file/d/1jK9KAaWfXXBpn3NlGBkn9NxrqSHu-rYG/view?usp=sharing) | [model](https://drive.google.com/file/d/1fEMQ_BZoOcrqDiKKqb9A6-WDibz91h5p/view?usp=sharing) | [model](https://drive.google.com/file/d/1Lg9iSvl0vYY6djuEozkJdAlm36REjdJX/view?usp=sharing) | [model](https://drive.google.com/file/d/17NHjTB3AKqWXxLvzXTHOjO_0tRdOGGp_/view?usp=sharing) | [model](https://drive.google.com/file/d/1TXKJM9-tkt60K7tDYIhMy-UUzQ1XQFA6/view?usp=sharing) |
|[log](https://github.com/dyh127/RFNet/blob/main/logs/Brats2020.log) | [log](https://github.com/dyh127/RFNet/blob/main/logs/Brats2018_split1.log) | [log](https://github.com/dyh127/RFNet/blob/main/logs/Brats2018_split2.log) | [log](https://github.com/dyh127/RFNet/blob/main/logs/Brats2018_split3.log) | [log](https://github.com/dyh127/RFNet/blob/main/logs/Brats2015.log) |


## Installation
We use pytorch1.2.0 and cuda9.0.

For all datasets, we train our networks with ```2 * V100 (16G)```. 

get dataset and environment [here](https://drive.google.com/drive/folders/1AwLwGgEBQwesIDTlWpubbwqxxd8brt5A?usp=sharing) and unzip them.
```
tar -xzf BRATS2020_Training_none_npy.tar.gz
tar -xzf BRATS2018_Training_none_npy.tar.gz
tar -xzf BRATS2015_Training_none_npy.tar.gz
tar -xzf pytorch_1.2.0a0+8554416-py36tf.tar.gz
tar -xzf cuda-9.0.tar.gz
```

## Usage
1. Set dataname, pypath and cudapath in ```job.sh```.

2. Set different splits for Brats2018 in L99-100 in ```train.py```.

3. Then run:
```
bash job.sh
```

### Note
1. We obtain the results by evaluating our models in the last epoch with the test set. If you want to evaluate models in other epochs, please use the --resume as in ```job.sh```.

2. We also provide the preprocessing code ```preprocess.py```. When using ```preprocess.py```, you need to set the path of raw data 'src_path' and the path of processed data 'tar_path' in ```preprocess.py```. The data structure in 'src_path' is shown as below:
```
BraTS20_Training_001/
    BraTS20_Training_001_flair.nii.gz
    BraTS20_Training_001_t1ce.nii.gz
    BraTS20_Training_001_t2.nii.gz
    BraTS20_Training_001_seg.nii.gz
    BraTS20_Training_001_t1.nii.gz
BraTS20_Training_002/
    BraTS20_Training_002_flair.nii.gz
    BraTS20_Training_002_t1ce.nii.gz
    BraTS20_Training_002_t2.nii.gz
    BraTS20_Training_002_seg.nii.gz
    BraTS20_Training_002_t1.nii.gz
BraTS20_Training_003/
...
...
BraTS20_Training_369/
```


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
