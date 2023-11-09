import subprocess
import sys

import cv2

"""
Step 2: Low-Rank Matrix Completion
"""
def lrmc(L, KERNAL, MAX_NITER_PARAM, GAMMA1_PARAM, GAMMA2_PARAM, VIDEO_FILE):
    cap = cv2.VideoCapture(VIDEO_FILE)
    if not cap.isOpened():
        print("Error opening video file")
        sys.exit(1)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = cap.get(cv2.CAP_PROP_FPS)

    cap.release()

    N = frame_count / (L * frame_rate)

    lrmc_script = f"fRMC({MAX_NITER_PARAM}, {GAMMA1_PARAM}, {GAMMA2_PARAM}, {N}, {frame_count}, {KERNAL})"
    command = ["matlab", "-nodisplay", "-nosplash", "-r", f"{lrmc_script}; exit"]
    subprocess.run(command)
