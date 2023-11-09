import sys
import cv2
import numpy as np

"""
# Step 1: Accumulative Multiframe Differencing
"""
def amfd(K, CONNECTIVITY, AREA_MIN, AREA_MAX, ASPECT_RATIO_MIN, ASPECT_RATIO_MAX, KERNAL, VIDEO_FILE):
    cap = cv2.VideoCapture(VIDEO_FILE)

    if not cap.isOpened():
        print("Error opening video file")
        sys.exit(1)
    
    # We want to know the height and width of the video frames in case we need to create a black image of the same size.
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    frame_count = 1

    # In order to calculate the AMFD, we need to read three frames at a times, the current frame (i_t) is the middle frame, so the first frame in the video will be skipped.
    _, I_t_minus_1 = cap.read()
    # The second frame in the video will be the current frame (i_t) and the first to be processed for an output.
    _, I_t = cap.read()

    # We save the raw frame to disk here because it will be needed in the LRMC process. 
    # Yes this is a bit of a hack.
    cv2.imwrite(f"processing/frames/{frame_count}.bmp", I_t_minus_1)
    # To ensure consistency in frame count, we designate the initial black image as the first frame processed by AMFD.
    # Although it may be preferable to use a white frame to prevent LRMC from being disregarded due to bitwise_or, we are uncertain if this will result in a significant impact.
    cv2.imwrite(
        f"processing/amfd/{frame_count}.bmp", np.zeros((height, width, 3), np.uint8)
    )

    # The AMFD process is repeated until the end of the video is reached.
    while True:
        # Get the next frame in the video and set it the next frame (i_t_plus_1).
        frame_count += 1
        ret, I_t_plus_1 = cap.read()

        # If there are no more frames to read, then we have reached the end of the video and can stop processing.
        if not ret:
            # Because we don't have a i_t_plus_1 for the last frame we save a black image to disk to pad the frame count.
            # We also save the raw frame disk (for the hack mentioned previously).
            cv2.imwrite(f"processing/frames/{frame_count}.bmp", I_t)
            cv2.imwrite(
                f"processing/amfd/{frame_count}.bmp",
                np.zeros((height, width, 3), np.uint8),
            )
            break

        I_t_gray = cv2.cvtColor(I_t, cv2.COLOR_BGR2GRAY)
        I_t_minus_1_gray = cv2.cvtColor(I_t_minus_1, cv2.COLOR_BGR2GRAY)
        I_t_plus_1_gray = cv2.cvtColor(I_t_plus_1, cv2.COLOR_BGR2GRAY)

        # Calculate the differencing images Dt1, Dt2, Dt3
        # Dt1 = |It - It-1| (Eq. 1 in MMB paper)
        Dt1 = cv2.absdiff(I_t_gray, I_t_minus_1_gray)
        # Dt2 = |It+1 - It-1| (Eq. 2 in MMB paper)
        Dt2 = cv2.absdiff(I_t_plus_1_gray, I_t_minus_1_gray)
        # Dt3 = |It+1 - It| (Eq. 3 in MMB paper)
        Dt3 = cv2.absdiff(I_t_plus_1_gray, I_t_gray)

        # Convert differencing images to a higher data type before addition to avoid overflow
        Dt1_16 = Dt1.astype(np.int16)
        Dt2_16 = Dt2.astype(np.int16)
        Dt3_16 = Dt3.astype(np.int16)

        # Calculate the accumulative response image Id
        # Id = (Dt1 + Dt2 + Dt3) / 3 (Eq. 4 in MMB paper)
        Id_16 = (
            Dt1_16 + Dt2_16 + Dt3_16
        ) // 3  # Use integer division to avoid floating point result

        # Convert back to 8-bit integer type after the calculation is done
        Id_gray = Id_16.astype(np.uint8)

        # Calculate the threshold T to extract targets
        mean_val, std_val = cv2.meanStdDev(Id_gray)
        # T = mean + k + std (Eq. 6 in MMB paper)
        T = mean_val[0][0] + K * std_val[0][0]

        # Convert the accumulative response image to a binary image
        # Id(x, y) = 255 if Id(x, y) >= T, 0 otherwise (Eq. 5 in MMB paper
        _, binary_image = cv2.threshold(Id_gray, T, 255, cv2.THRESH_BINARY)

        # Perform morphological operations on binary image
        binary_image = cv2.morphologyEx(
            binary_image, cv2.MORPH_OPEN, np.ones((KERNAL, KERNAL), np.uint8)
        )

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

        binary_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(f"processing/frames/{frame_count}.bmp", I_t)
        cv2.imwrite(f"processing/amfd/{frame_count}.bmp", binary_image)

        I_t_minus_1 = I_t
        I_t = I_t_plus_1