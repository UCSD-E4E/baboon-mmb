"""
// ============================================================================
// This is a Rust implementation of the paper:
// Detecting and Tracking Small and Dense Moving Objects in Satellite Videos: A Benchmark
// Qian Yin, QingyongHu, HaoLiu, Feng Zhang, Yingqian Wang, Zaiping Lin, Wei An, Yulan Guo
// This is NOT the original implemtation by the authors, as there seems to be no code available.
// ============================================================================
"""

import shutil
import sys
import os
import argparse

from amfd import amfd
from lrmc import lrmc
from pf import pf

"""
// ============================================================================
// List of parameters with their default values, suggested range, and reference.
// ============================================================================
// k: 4, [0.0, 8.0], Eq. 4
//
// connectivity: 8, {4, 8}, Algorithm 1
//
// area_min: 5, [0, area_max), Eq. 7
//
// area_max: 80, (area_min, 100], Eq. 7
//
// aspect_ratio_min: 1.0, [0.0, aspect_ratio_max), Eq. 7
//
// aspect_ratio_max: 6.0, (aspect_ratio_min, 10.0], Eq. 7
//
// l: 4, [1, 10], Eq. 9
//
// kernel: 3, {1, 3, 5, 7, 9, 11}, Algorithm 1
//
// pipeline_length: 5, [1, 10], Step 1 of Pipeline Filter
//
// pipeline_size: 7, {3, 5, 7, 9, 11}, Step 1 of Pipeline Filter
//
// h: 3, [1, pipeline_length], Step 4 of Pipeline Filter
//
// max_niter_param: 10, [1, 20], fRMC (Background subtraction via fast robust matrix completion, Rezaei et al., 2017)
//
// gamma1_param: 0.8, [0.0, 1.0], fRMC (Background subtraction via fast robust matrix completion, Rezaei et al., 2017)
//
// gamma2_param: 0.8, [gamma1_param, 1.0], fRMC (Background subtraction via fast robust matrix completion, Rezaei et al., 2017)
//
// video_file: String, path to video file
// ============================================================================
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MMB Paper parameters")

    parser.add_argument(
        "--K", type=float, default=4, help="Eq. 4 in MMB paper ([0.0, 8.0])"
    )
    parser.add_argument(
        "--CONNECTIVITY",
        type=int,
        choices=[4, 8],
        default=4,
        help="Algorithm 1 in MMB paper (4 or 8)",
    )
    parser.add_argument(
        "--AREA_MIN", type=int, default=5, help="Eq. 7 in MMB paper ([1, 10])"
    )
    parser.add_argument(
        "--AREA_MAX", type=int, default=80, help="Eq. 7 in MMB paper ([50, 100])"
    )
    parser.add_argument(
        "--ASPECT_RATIO_MIN",
        type=float,
        default=1.0,
        help="Eq. 7 in MMB paper ([0.8, 1.5])",
    )
    parser.add_argument(
        "--ASPECT_RATIO_MAX",
        type=float,
        default=6.0,
        help="Eq. 7 in MMB paper ([4.0, 8.0])",
    )
    parser.add_argument("--L", type=int, default=4, help="Eq. 9 in MMB paper ([1, 8])")
    parser.add_argument(
        "--KERNAL",
        type=int,
        choices=[1, 3, 5, 7, 9, 11],
        default=3,
        help="Algorithm 1 in MMB paper (1, 3, 5, 7, 9, or 11)",
    )
    parser.add_argument(
        "--PIPELINE_LENGTH",
        type=int,
        default=5,
        help="Step 1 of Pipeline Filter in MMB paper ([3, 10])",
    )
    parser.add_argument(
        "--PIPELINE_SIZE",
        type=int,
        choices=[3, 5, 7, 9, 11],
        default=7,
        help="Step 1 of Pipeline Filter in MMB paper (3, 5, 7, 9, or 11)",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=3,
        help="Step 4 of Pipeline Filter in MMB paper ([1, PIPELINE_LENGTH])",
    )
    parser.add_argument(
        "--MAX_NITER_PARAM",
        type=int,
        default=10,
        help="Max number of iterations ([1, 20])",
    )
    parser.add_argument(
        "--GAMMA1_PARAM", type=float, default=0.3, help="Gamma1 parameter ([0.0, 1.0])"
    )
    parser.add_argument(
        "--GAMMA2_PARAM", type=float, default=0.8, help="Gamma2 parameter ([0.0, 1.0])"
    )
    parser.add_argument(
        "video_file", type=str, help="Path to the video file to be processed."
    )

    args = parser.parse_args()

    K = args.K
    CONNECTIVITY = args.CONNECTIVITY
    AREA_MIN = args.AREA_MIN
    AREA_MAX = args.AREA_MAX
    ASPECT_RATIO_MIN = args.ASPECT_RATIO_MIN
    ASPECT_RATIO_MAX = args.ASPECT_RATIO_MAX
    L = args.L
    KERNAL = args.KERNAL
    PIPELINE_LENGTH = args.PIPELINE_LENGTH
    PIPELINE_SIZE = args.PIPELINE_SIZE
    H = args.H
    MAX_NITER_PARAM = args.MAX_NITER_PARAM
    GAMMA1_PARAM = args.GAMMA1_PARAM
    GAMMA2_PARAM = args.GAMMA2_PARAM

    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <video file>")
        sys.exit(1)

    VIDEO_FILE = args.video_file
    
    # Rather than storing the resulting binary masks from AMFD and LRMC in memory, we opt to save them to disk. 
    # Although it may be feasible to merge the AMFD and LRMC processes (and possibly the Pipeline Filter process), we decided to keep them separate. 
    # Combining them could make the code harder to understand and maintain, and could also result in a significant increase in the overhead required by the MATLAB scripts. 
    # Additionally, we save the frames to disk since we only need to retrieve them again for the Pipeline Filter step. 
    # By doing so, we avoid keeping them in memory when they are not needed for an extended period.
    if not os.path.exists("processing/amfd"):
        os.makedirs("processing/amfd")
    if not os.path.exists("processing/lrmc"):
        os.makedirs("processing/lrmc")

    # The LRMC algorithm necessitates the division of the video into individual frames, as certain operating systems pose difficulties for MATLAB's video reading capabilities.
    # However, relying solely on frames is insufficient, as knowledge of the video's frame rate is also needed.
    if not os.path.exists("processing/frames"):
        os.makedirs("processing/frames")

    amfd(K, CONNECTIVITY, AREA_MIN, AREA_MAX, ASPECT_RATIO_MIN, ASPECT_RATIO_MAX, KERNAL, VIDEO_FILE)
    lrmc(L, KERNAL, MAX_NITER_PARAM, GAMMA1_PARAM, GAMMA2_PARAM, VIDEO_FILE)
    pf(PIPELINE_LENGTH, PIPELINE_SIZE, H, VIDEO_FILE)

    shutil.rmtree("processing")
