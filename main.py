import sys
import cv2
import numpy as np
import os
import csv

from scipy.optimize import linear_sum_assignment


if __name__ == "__main__":
    # HYPER PARAMETERS
    K = 4  # Eq. 4 in MMB paper
    CONNECTIVITY = 4  # Algorithm 1 in MMB paper
    AREA_MIN = 5  # Eq. 7 in MMB paper
    AREA_MAX = 80  # Eq. 7 in MMB paper
    ASPECT_RATIO_MIN = 1.0  # Eq. 7 in MMB paper
    ASPECT_RATIO_MAX = 6.0  # Eq. 7 in MMB paper
    L = 4  # Eq. 9 in MMB paper
    KERNAL = np.ones((3, 3), np.uint8)  # Algorithm 1 in MMB paper
    PIPELINE_LENGTH = 5  # Step 1 of Pipeline Filter in MMB paper
    PIPELINE_SIZE = 7  # Step 1 of Pipeline Filter in MMB paper
    H = 3  # Step 4 of Pipeline Filter in MMB paper

    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <video file>")
        sys.exit(1)

    video_file = sys.argv[1]

    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print(f"Unable to open video: {video_file}")
        sys.exit(1)

    if not os.path.exists("processing/amfd"):
        os.makedirs("processing/amfd")
    if not os.path.exists("processing/frames"):
        os.makedirs("processing/frames")

    frame_count = 1

    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    """
    Step 1: Accumulative Multiframe Differencing
    """
    _, I_t_minus_1 = cap.read()
    _, I_t = cap.read()

    cv2.imwrite(f"processing/frames/{frame_count}.bmp", I_t_minus_1)
    cv2.imwrite(
        f"processing/amfd/{frame_count}.bmp", np.zeros((height, width, 3), np.uint8)
    )

    while True:
        frame_count += 1
        ret, I_t_plus_1 = cap.read()

        if not ret:
            cv2.imwrite(f"processing/frames/{frame_count}.bmp", I_t)
            cv2.imwrite(
                f"processing/amfd/{frame_count}.bmp",
                np.zeros((height, width, 3), np.uint8),
            )
            break

        # Calculate the differencing images Dt1, Dt2, Dt3
        # Dt1 = |It - It-1| (Eq. 1 in MMB paper)
        Dt1 = cv2.absdiff(I_t, I_t_minus_1)
        # Dt2 = |It+1 - It-1| (Eq. 2 in MMB paper)
        Dt2 = cv2.absdiff(I_t_plus_1, I_t_minus_1)
        # Dt3 = |It+1 - It| (Eq. 3 in MMB paper)
        Dt3 = cv2.absdiff(I_t_plus_1, I_t)

        # Calculate the accumulative response image Id
        # Id = (Dt1 + Dt2 + Dt3) / 3 (Eq. 4 in MMB paper)
        Id = (Dt1 + Dt2 + Dt3) / 3
        Id_gray = cv2.cvtColor(Id.astype(np.uint8), cv2.COLOR_BGR2GRAY)

        # Calculate the threshold T to extract targets
        mean_val, std_val = cv2.meanStdDev(Id_gray)
        # T = mean + k + std (Eq. 6 in MMB paper)
        T = mean_val[0][0] + K * std_val[0][0]

        # Convert the accumulative response image to a binary image
        # Id(x, y) = 255 if Id(x, y) >= T, 0 otherwise (Eq. 5 in MMB paper
        _, binary_image = cv2.threshold(Id_gray, T, 255, cv2.THRESH_BINARY)

        # Perform morphological operations on binary image
        binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, KERNAL)

        # Remove false alarms
        # Connected area  must satisfy the following conditions:
        # 1. Area must be between AREA_MIN and AREA_MAX
        # 2. Aspect ratio must be between ASPECT_RATIO_MIN and ASPECT_RATIO_MAX (Eq. 7 in MMB paper)
        # Compute connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_image, connectivity=CONNECTIVITY
        )

        # Iterate through each component and check the area and aspect ratio
        for i in range(1, num_labels):  # Start from 1 to ignore the background
            x, y, w, h, area = stats[i]
            aspect_ratio = float(w) / h

            if (
                area < AREA_MIN
                or area > AREA_MAX
                or aspect_ratio < ASPECT_RATIO_MIN
                or aspect_ratio > ASPECT_RATIO_MAX
            ):
                # Create a mask for the current component
                mask = (labels == i).astype(np.uint8) * 255
                # Subtract the mask from the binary image to remove the component
                binary_image = cv2.subtract(binary_image, mask)

        cv2.imshow("Binary Image", binary_image)
        cv2.waitKey(30)

        binary_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(f"processing/frames/{frame_count}.bmp", I_t)
        cv2.imwrite(f"processing/amfd/{frame_count}.bmp", binary_image)

        I_t_minus_1 = I_t
        I_t = I_t_plus_1

    cap.release()

    cv2.destroyAllWindows()

    """
    Step 2 TODO: Figure out a way to call the Demo_fRMC.m script from Python
    """

    """
    Step 3: Pipeline Filter
    """

    def read_images_from_directory(directory):
        images = []
        image_files = sorted(
            [f for f in os.listdir(directory) if f.endswith(".bmp")],
            key=lambda x: int(x.split(".")[0]),
        )

        for image_file in image_files:
            image_path = os.path.join(directory, image_file)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            images.append(image)

        return images

    amfd_images = read_images_from_directory("processing/amfd")
    lrmc_images = read_images_from_directory("processing/amfd")

    merged_images = [
        cv2.bitwise_or(amfd_image, lrmc_image)
        for amfd_image, lrmc_image in zip(amfd_images, lrmc_images)
    ]

    # The rest of the pipeline filter that was here was bad and didn't so it was removed.
    # Needs to be rewritten.
    # Should output a csv file of the bounding boxes of the targets in each frame.

