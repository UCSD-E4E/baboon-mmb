"""
// ============================================================================
// This is a Rust implementation of the paper:
// Detecting and Tracking Small and Dense Moving Objects in Satellite Videos: A Benchmark
// Qian Yin, QingyongHu, HaoLiu, Feng Zhang, Yingqian Wang, Zaiping Lin, Wei An, Yulan Guo
// This is NOT the original implemtation by the authors, as there seems to be no code available.
// ============================================================================
"""

import shutil
import signal
import subprocess
import sys
import os
import argparse


def signal_handler(sig, frame):
    print("Stopping subprocesses...")
    for p in subprocesses:
        p.terminate()
        p.wait()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)

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
// bitewise_and: true, {true, false}, Bitewise AND the masks
//
// output_images: false, {true, false}, Output final images to disk
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
        "--BITEWISE_AND", type=bool, default=True, help="Bitewise AND the masks"
    )
    parser.add_argument(
        "--OUTPUT_IMAGES", type=bool, default=False, help="Output final images to disk"
    )
    parser.add_argument(
        "--FRAME_RATE", type=int, default=10, help="Frame rate of video file"
    )
    parser.add_argument(
        "IMAGE_SEQUENCE",
        type=str,
        help="Path to the folder containing image sequence for video",
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
    BITEWISE_AND = args.BITEWISE_AND
    OUTPUT_IMAGES = args.OUTPUT_IMAGES
    FRAME_RATE = args.FRAME_RATE

    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <video file>")
        sys.exit(1)

    IMAGE_SEQUENCE = args.IMAGE_SEQUENCE

    # Rather than storing the resulting binary masks from AMFD and LRMC in memory, we opt to save them to disk.
    # Although it may be feasible to merge the AMFD and LRMC processes (and possibly the Pipeline Filter process), we decided to keep them separate.
    # Combining them could make the code harder to understand and maintain, and could also result in a significant increase in the overhead required by the MATLAB scripts.
    # Additionally, we save the frames to disk since we only need to retrieve them again for the Pipeline Filter step.
    # By doing so, we avoid keeping them in memory when they are not needed for an extended period.
    if not os.path.exists("processing/amfd"):
        os.makedirs("processing/amfd")
    if not os.path.exists("processing/lrmc"):
        os.makedirs("processing/lrmc")

    subprocesses = []

    amfd_args = [
        "python3",
        "-c",
        "from amfd import amfd; amfd({}, {}, {}, {}, {}, {}, {}, '{}')".format(
            K,
            CONNECTIVITY,
            AREA_MIN,
            AREA_MAX,
            ASPECT_RATIO_MIN,
            ASPECT_RATIO_MAX,
            KERNAL,
            IMAGE_SEQUENCE,
        ),
    ]
    subprocesses.append(subprocess.Popen(amfd_args))

    lrmc_args = [
        "python3",
        "-c",
        "from lrmc import lrmc; lrmc({}, {}, {}, {}, {}, {}, '{}')".format(
            L,
            KERNAL,
            MAX_NITER_PARAM,
            GAMMA1_PARAM,
            GAMMA2_PARAM,
            FRAME_RATE,
            IMAGE_SEQUENCE,
        ),
    ]
    subprocesses.append(subprocess.Popen(lrmc_args))

    pf_args = [
        "python3",
        "-c",
        "from pf import pf; pf({}, {}, {}, {}, '{}', {})".format(
            PIPELINE_LENGTH,
            PIPELINE_SIZE,
            H,
            BITEWISE_AND,
            IMAGE_SEQUENCE,
            OUTPUT_IMAGES,
        ),
    ]
    subprocesses.append(subprocess.Popen(pf_args))

    # signal.pause()

    for p in subprocesses:
        p.wait()

    shutil.rmtree("./processing")
