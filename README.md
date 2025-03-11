# Surgical SAM 2: Real-time Segment Anything in Surgical Video by Efficient Frame Pruning

Official implementation for SurgSAM2, an innovative model that leverages the power of the Segment Anything Model 2 (SAM2), integrating it with an efficient frame pruning mechanism for real-time surgical video segmentation.

> [Surgical SAM 2: Real-time Segment Anything in Surgical Video by Efficient Frame Pruning](https://openreview.net/forum?id=WSDrF5mKVp)
>
> Haofeng Liu, Erli Zhang, Junde Wu, Mingxuan Hong, Yueming Jin
>
> NeurIPS 2024 Workshop AIM-FM

## Overview

We introduce Surgical SAM 2 (SurgSAM-2), an innovative model that leverages the power of the Segment Anything Model 2 (SAM2), integrating it with an efficient frame pruning mechanism for real-time surgical video segmentation. The proposed SurgSAM-2

- dramatically reduces memory usage and computational cost of SAM2 for real-time clinical application;
- achieves superior performance with 3× FPS (86 FPS), making real-time surgical segmentation in resource-constrained environments a feasible reality.

![architecture](./assets/architecture.png)

## Dataset Acquisition and Preprocessing

### Data Download

1. Please download the training and validation sets used in our experiments:
   1. [VOS-Endovis17](https://drive.google.com/file/d/1tw7KzpXqOC3HsjsUknro4MOqQ2Nr3vD1/view?usp=drive_link)
   2. [VOS-Endovis18](https://drive.google.com/file/d/1Vod5jKoC8CAEqlYdiXZ2HMexP9IFXbGp/view?usp=drive_link)
2. The original image data can be obtained from the official websites:
   1. [Endovis17 Official Dataset](https://endovissub2017-roboticinstrumentsegmentation.grand-challenge.org/)
   2. [Endovis18 Official Dataset](http://endovissub2018-roboticscenesegmentation.grand-challenge.org/)

### Data Preprocessing

Follow the data preprocessing instructions provided in the [ISINet](https://github.com/BCV-Uniandes/ISINet) repository. 

### Dataset Structure

After downloading, organize your data according to the following structure:

```
project_root/
└── datasets/
    └── VOS-Endovis18/
        └──  train/
        	└──  JPEGImages/
        	└──  Annotations/
        └──  valid/
        	└──  JPEGImages/
        	└──  Annotations/
        	└──  VOS/
```

## Training

To train the model, run:

```
CUDA_VISIBLE_DEVICES=0 python training/train.py --config configs/sam2.1_training/sam2.1_hiera_s_endovis18_instrument
```

## Evaluation

Download the pretrained weights from [sam2.1_hiera_s_endo18](https://drive.google.com/file/d/1DyrrLKst1ZQwkgKM7BWCCwLxSXAgOcMI/view?usp=drive_link). Place the file at `project_root/checkpoints/sam2.1_hiera_s_endo18.pth`.

```
python tools/vos_inference.py --sam2_cfg configs/sam2.1/sam2.1_hiera_s.yaml --sam2_checkpoint ./checkpoints/sam2.1_hiera_s_endo18.pth --output_mask_dir ./results/sam2.1/endovis_2018/instrument --input_mask_dir ./datasets/VOS-Endovis18/valid/VOS/Annotations_vos_instrument --base_video_dir ./datasets/VOS-Endovis18/valid/JPEGImages --gt_root ./datasets/VOS-Endovis18/valid/Annotations --gpu_id 0
```

## Demo

Demo data from Endovis 2018 can be downloaded from  [2018 demo data](https://drive.google.com/file/d/1RG9DIGXFQwXckYpaTLEUyxYq4DexgBOY/view?usp=sharing). 

After downloading, arrange the files according to the following structure:

```
project_root/
└── datasets/
    └── endovis18/
        └── images/
            └── seq_2/
                └── ...
```

## Acknowledgement

This research utilizes datasets from [Endovis 2017](https://endovissub2017-roboticinstrumentsegmentation.grand-challenge.org/Downloads/) and [Endovis 2018](https://endovissub2018-roboticscenesegmentation.grand-challenge.org/Downloads/).. If you wish to use these datasets, please request access through their respective official websites.

Our implementation builds upon the [segment anything 2](https://github.com/facebookresearch/segment-anything-2) framework. We extend our sincere appreciation to the authors for their outstanding work and significant contributions to the field of video segmentation.

## Citation

```
@misc{liu2024surgicalsam2realtime,
 title={Surgical SAM 2: Real-time Segment Anything in Surgical Video by Efficient Frame Pruning}, 
author={Haofeng Liu and Erli Zhang and Junde Wu and Mingxuan Hong and Yueming Jin},
 year={2024},
 eprint={2408.07931},
 archivePrefix={arXiv},
 primaryClass={cs.CV},
 url={https://arxiv.org/abs/2408.07931}, 
}
```