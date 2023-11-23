# Baboon MMB

## Overview

This repository contains the Python implementation of the algorithms described in the paper: "Detecting and Tracking Small and Dense Moving Objects in Satellite Videos: A Benchmark" by Qian Yin et al. 

Please note that this is not the original implementation by the authors, as their code was not publicly available.

## Getting Started

### Prerequisites
- Python 3.12 or higher
  -  numpy 1.23.5
  - opencv_contrib_python 4.7.0.68
  - opencv_python 4.7.0.68
  - scipy 1.11.3
- MatLab R2023b or higher
  - Image Processing Toolbox 23.2

### Running the Code
To run the project with the default parameters, use the following command:
```bash
python3.12 main.py <path_to_video>
```
### Configuration

You can configure the behavior of the algorithms by setting different command-line arguments. The following table lists the parameters you can adjust:

| Parameter | Default | Suggested Range | Description |
| --- | --- | --- | --- |
| k | 4 | [0.0, 8.0] | Parameter k in Eq. 4 |
| connectivity | 8 | {4, 8} | Connectivity parameter in Algorithm 1 |
| area_min | 5 | [0, area_max) | Minimum area of a blob in Eq. 7 |
| area_max | 80 | (area_min, 100] | Maximum area of a blob in Eq. 7 |
| aspect_ratio_min | 1.0 | [0.0, aspect_ratio_max) | Minimum aspect ratio of a blob in Eq. 7 |
| aspect_ratio_max | 6.0 | (aspect_ratio_min, 10.0] | Maximum aspect ratio of a blob in Eq. 7 |
| l | 4 | [1, 10] | Parameter l in Eq. 9 |
| kernel | 3 | {1, 3, 5, 7, 9, 11} | Kernel size in Algorithm 1 |
| pipeline_length | 5 | [1, 10] | Pipeline filter length in Step 1 of Pipeline Filter |
| pipeline_size | 7 | {3, 5, 7, 9, 11} | Pipeline filter size in Step 1 of Pipeline Filter |
| h | 3 | [1, pipeline_length] | Parameter h in Step 4 of Pipeline Filter |
| max_niter_param | 10 | [1, 20] | Parameter max_niter in fRMC (Background subtraction via fast robust matrix completion, Rezaei et al., 2017) |
| gamma1_param | 0.3 | [0.0, 1.0] | Parameter gamma1 in fRMC (Background subtraction via fast robust matrix completion, Rezaei et al., 2017) |
| gamma2_param | 0.8 | [gamma1_param, 1.0] | Parameter gamma2 in fRMC (Background subtraction via fast robust matrix completion, Rezaei et al., 2017) |
| bitwise_and | True | {True, False} | Whether to use bitwise and vs or in Pipeline Filter |
| output_images | False | Whether to also output images of the final bounding boxes |
| video_file | String | path to video file | Path to video file |

### Output
The output of the program is a csv file containing the following columns:

|frame|x|y|w|h|
| --- | --- | --- | --- | --- |
|Frame number|X coordinate of the center of the bounding box|Y coordinate of the center of the bounding box|Width of the bounding box|Height of the bounding box|

### Example
```bash
python3.12 main.py --k 4 --connectivity 8 --area_min 5 --area_max 80 --aspect_ratio_min 1.0 --aspect_ratio_max 6.0 --l 4 --kernel 3 <path_to_video>
```

## Citations
If you use this code in your research, please cite the following papers
```
@article{yin2021detecting,
      title={Detecting and Tracking Small and Dense Moving Objects in Satellite Videos: A Benchmark},
      author={Yin, Qian and Hu, Qingyong and Liu, Hao and Zhang, Feng and Wang, Yingqian and Lin, Zaiping and An, Wei and Guo, Yulan},
      journal={IEEE Transactions on Geoscience and Remote Sensing},
      year={2021},
      publisher={IEEE}
    }
```

```
@inproceedings{rezaei2017background,
        title={Background Subtraction via Fast Robust Matrix Completion},
        author={Rezaei, Behnaz and Ostadabbas, Sarah},
        booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
        pages={1871--1879},
        year={2017}
    }
```