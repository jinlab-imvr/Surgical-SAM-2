# Surgical SAM 2: Real-time Segment Anything in Surgical Video by Efficient Frame Pruning

Official implementation for SurgSAM2, an innovative model that leverages the power of the Segment Anything Model 2 (SAM2), integrating it with an efficient frame pruning mechanism for real-time surgical video segmentation. Paper in [arxiv](https://arxiv.org/abs/2408.07931).

We introduce Surgical SAM 2 (SurgSAM-2), an innovative model that leverages the power of the Segment Anything Model 2 (#SAM2), integrating it with an efficient frame pruning mechanism for real-time surgical video segmentation. The proposed SurgSAM-2

- dramatically reduces memory usage and computational cost of SAM2 for real-time clinical application;
- achieves superior performance with 3× FPS (86 FPS), making real-time surgical segmentation in resource-constrained environments a feasible reality.

![architecture](./assets/architecture.png)

## Training

The source code is coming soon

## Evaluation data and pretrained weighted

The demo data from Endovis 2018 could be downloaded from [2018 demo data](https://drive.google.com/file/d/1RG9DIGXFQwXckYpaTLEUyxYq4DexgBOY/view?usp=sharing). Please put the data to construct the data construction as: project_root/datasets/endovis18/images/seq_2/...

The pretrained weighted could be downloaded from [sam2_hiera_s_endo18](https://drive.google.com/file/d/102AsbwpntPfUV96ANO2AWe7L6feskYnM/view?usp=sharing). Please put the weight to construct project_root/model_weights/sam2_hiera_s_endo18.pth

## Acknowledgement

This model was trained with the datasets from Endovis 2017, Endovis 2018. If you need to use the data, please apply for the usage from their website [Endovis 2017](https://endovissub2017-roboticinstrumentsegmentation.grand-challenge.org/Downloads/) and [Endovis 2018](https://endovissub2018-roboticscenesegmentation.grand-challenge.org/Downloads/).

This code was adapted from [segment anything 2](https://github.com/facebookresearch/segment-anything-2). We are grateful for their excellent code and contribution to video segmentation.

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