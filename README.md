# Baboon MMB

## Overview
This repository contains the MATLAB implementation of the Motion Modeling Baseline (MMB) described in the paper "Detecting and Tracking Small and Dense Moving Objects in Satellite Videos: A Benchmark" by Qian Yin et al.

Please note that this is not the original implementation by the authors, as their code was not publicly available.

## Getting Started
### Prerequisites
* MATLAB
  * Computer Vision Toolbox
  * Global Optimization Toolbox
  * Image Processing Toolbox
  * Optimization Toolbox
  * Parallel Computing Toolbox

### Running the Code

#### Baboon MMB
To run the project with the default parameters, use the following command in the MATLAB console:

```matlab
baboon_mmb('IMAGE_SEQUENCE', 'path/to/image/sequence/', 'FRAME_RATE', framerate)
```

This will create an `output/` directory which contains:
* `amfd/`, `amfdMasks.mat`: Binary masks outputted by the Accumulative Multi-Frame Difference (AMFD) module.
* `lrmc/`, `lrmcMasks.mat`: Binary masks outputted by the Low Rank Matrix Completion (LRMC) module.
* `combined/`, `combinedMasks.mat`: Binary masks generated by taking the BITWISE_AND (or BITWISE_OR) of the outputs of the AMFD and LRMC modules.
* `objects.txt`, `objects.mat`: A struct of all detected objects. Each object contains `frameNumber`, `id`, `x`, `y`, `width`, `height`.
* `frames/`: The detected bounding boxes drawn over the original image sequence.

##### Configuration
You can configure the behavior of the algorithms by setting different command-line arguments. The following table lists the parameters you can adjust:

| Parameter        | Default | Range                  | Description                                                                                                                                                                     |
| ---------------- | ------- | ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| K                | 4       | [0, Inf)               | Controls the thresholding in the AMFD algorithm. It influences the sensitivity of detection, where a higher value makes the detection more selective.                           |
| CONNECTIVITY     | 8       | {4, 8}                 | Defines the connectivity used in morphological operations for the AMFD algorithm. It can be either 4 or 8, determining how pixel connectivity is considered.                    |
| AREA_MIN         | 5       | [1, width * height]    | The minimum area threshold for detected regions in the AMFD algorithm. Regions smaller than this value are ignored.                                                             |
| AREA_MAX         | 80      | [1, width * height]    | The maximum area threshold for detected regions in the AMFD algorithm. Regions larger than this value are ignored.                                                              |
| ASPECT_RATIO_MIN | 1       | [1, max(width, height)]| The minimum aspect ratio for detected bounding boxes in the AMFD algorithm. Aspect ratios smaller than this value are ignored.                                                  |
| ASPECT_RATIO_MAX | 6       | [1, max(width, height)]| The maximum aspect ratio for detected bounding boxes in the AMFD algorithm. Aspect ratios larger than this value are ignored.                                                   |
| L                | 4       | [0, total_seconds]     | The length of frames considered for the LRMC algorithm. This parameter controls how many frames are processed together to detect changes.                                       |
| MAX_NITER_PARAM  | 10      | [1, Inf)               | The maximum number of iterations for the LRMC algorithm to converge.                                                                                                            |
| GAMMA1_PARAM     | 0.3     | [0, 1]                 | A parameter that controls the regularization strength in the LRMC algorithm.                                                                                                     |
| GAMMA2_PARAM     | 0.8     | [0, 1]                 | Another parameter that controls the regularization strength in the LRMC algorithm.                                                                                               |
| KERNEL           | 3       | [0, max(width, height)]| The size of the structuring element used in morphological operations in both AMFD and LRMC algorithms.                                                                          |
| BITWISE_OR       | false   | {true, false}          | A boolean parameter that, if true, combines the masks from the AMFD and LRMC algorithms using a bitwise OR operation. If false, it combines them using a bitwise AND operation. |
| H                | 3       | [0, frameCount - 1]    | The minimum number of consistent object detections required across the pipeline for the object to be considered valid in the pipeline filter (PF) algorithm.                    |
| PIPELINE_LENGTH  | 5       | [0, frameCount - 1]    | The number of frames to consider in the pipeline PF algorithm. This parameter controls the length of the buffer used for object tracking.                                       |
| PIPELINE_SIZE    | 7       | [0, frame_diagonal]    | The maximum distance (in pixels) allowed between object detections in consecutive frames for them to be considered the same object in the PF algorithm.                         |
| FRAME_RATE       | 10      | [1, Inf)               | The frame rate of the input image sequence, used for temporal processing in the LRMC algorithm.                                                                                 |
| IMAGE_SEQUENCE   | ''      | N/A                    | The path to the folder containing the input image sequence. The images are loaded and processed in the sequence.                                                                |
| DEBUG            | true    | {true, false}          | A boolean parameter that, if true, enables saving intermediate results and additional debugging outputs.                                                                        |

#### Optimize
If you would prefer to iteratively run `baboon_mmb()` to determine the best hyperparameters, `optimize.m` can be configured and compiled. Note this operation is long and is intended to be run on a cloud computer.

To use the `optimize` function, you need to configure the `config.json` file. Here is an example of the `config.json` file:

```json
{
  "lb": [
    "0",
    "1",
    "1",
    "1",
    "1",
    "1",
    "0",
    "0",
    "1",
    "0",
    "0",
    "0",
    "1",
    "0",
    "0"
  ],
  "ub": [
    "Inf",
    "2",
    "Inf",
    "Inf",
    "Inf",
    "Inf",
    "Inf",
    "Inf",
    "2",
    "Inf",
    "Inf",
    "Inf",
    "Inf",
    "1",
    "1"
  ],
  "mu": [
    "4",
    "2",
    "5",
    "80",
    "1",
    "6",
    "4",
    "3",
    "1",
    "5",
    "7",
    "3",
    "10",
    "0.3",
    "0.8"
  ],
  "std": [
    "1",
    "0.25",
    "1",
    "19.75",
    "0.25",
    "1.25",
    "1",
    "0.75",
    "0.25",
    "1.25",
    "1.75",
    "0.75",
    "2.25",
    "0.075",
    "0.05"
  ],
  "intIndices": [
    2,
    3,
    4,
    8,
    9,
    10,
    11,
    12,
    13
  ],
  "InputPath": "input/viso_video_1",
  "GroundTruthPath": "input/viso_video_1_gt.txt",
  "FrameRate": "10",
  "PopulationSize": "1000",
  "MaxGenerations": "1e9",
  "FunctionTolerance": "1e-10",
  "MaxStallGenerations": "1e6",
  "UseParallel": "false",
  "ParetoFraction": "0.7",
  "Display": "iter",
}
```

## Citations
If you use this code in your research, please cite the following papers:

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