import os
import subprocess
import sys

"""
Step 2: Low-Rank Matrix Completion
"""


def lrmc(
    L, KERNAL, MAX_NITER_PARAM, GAMMA1_PARAM, GAMMA2_PARAM, FRAME_RATE, IMAGE_SEQUENCE
):
    files = [f for f in os.listdir(IMAGE_SEQUENCE) if f.endswith(".jpg")]
    if not files:
        print("No images found in the specified folder")
        sys.exit(1)

    frame_count = len(files)

    N = frame_count / (L * FRAME_RATE)

    lrmc_script = f"pkg load image; fRMC({MAX_NITER_PARAM}, {GAMMA1_PARAM}, {GAMMA2_PARAM}, {N}, {frame_count}, {KERNAL}, '{IMAGE_SEQUENCE}')"
    command = ["octave", "--eval", lrmc_script]
    subprocess.run(command)
