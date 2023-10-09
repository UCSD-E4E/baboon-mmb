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

    def get_blob_centroids_and_boxes(binary_img):
        # Find contours in the binary images
        contours, _ = cv2.findContours(
            binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        centroids = []
        bounding_boxes = []

        for contour in contours:
            # Calculate bounding box for each contour
            x, y, w, h = cv2.boundingRect(contour)
            bounding_boxes.append((x, y, w, h))

            # Caclulate centroid for each contour
            M = cv2.moments(contour)
            if M["m00"] != 0:
                c_x = int(M["m10"] / M["m00"])
                c_y = int(M["m01"] / M["m00"])
                centroids.append((c_x, c_y))
            else:
                c_x, c_y = 0, 0

            centroids.append((c_x, c_y))

        return centroids, bounding_boxes

    amfd_images = read_images_from_directory("processing/amfd")
    lrmc_images = read_images_from_directory("processing/lrmc")

    merged_images = [
        cv2.bitwise_or(amfd_image, lrmc_image)
        for amfd_image, lrmc_image in zip(amfd_images, lrmc_images)
    ]

    frame_count = 1

    with open("./processing/output.csv", 'w', newline='') as csvfile:
        fieldnames = ['frame', 'x', 'y', 'w', 'h']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    for idx in range(len(merged_images) - PIPELINE_LENGTH):
        final_bboxes = []

        current_frame = merged_images[idx]
        next_frames = merged_images[idx + 1 : idx + PIPELINE_LENGTH + 1]

        # 2. Identify candidate target points (centroids) and their bounding boxes
        candidate_centroids, candidate_bboxes = get_blob_centroids_and_boxes(
            current_frame
        )
        h = np.zeros(len(candidate_centroids))

        # 3. Check for object centroids in neighborhood in next frames
        for next_frame in next_frames:
            (
                next_candidate_centroids,
                next_candidate_bboxes,
            ) = get_blob_centroids_and_boxes(next_frame)
            cost_matrix = np.zeros(
                (len(candidate_centroids), len(next_candidate_centroids))
            )

            for i, current_centroid in enumerate(candidate_centroids):
                for j, next_centroid in enumerate(next_candidate_centroids):
                    s_x = abs(current_centroid[0] - next_centroid[0])  # Eq. 11
                    s_y = abs(current_centroid[1] - next_centroid[1])  # Eq. 12

                    if 0 < s_x < PIPELINE_SIZE and 0 < s_y < PIPELINE_SIZE:  # Eq. 10
                        cost_matrix[i, j] = np.sqrt(s_x**2 + s_y**2)
                    else:
                        cost_matrix[i, j] = float("inf")

            # Use Hungarian algorithm to find the optimal assignment
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            # Increase h for matched points and update the bounding box and centroid
            for i, j in zip(row_ind, col_ind):
                if cost_matrix[i, j] < float("inf"):
                    h[i] += 1
                    candidate_bboxes[i] = next_candidate_bboxes[j]
                    candidate_centroids[i] = next_candidate_centroids[j]

        # 4. Check object occurences
        for i, occurence in enumerate(h):
            if occurence >= H:
                final_bboxes.append(candidate_bboxes[i])
            elif 3 <= occurence <= 4:
                final_bboxes.append(candidate_bboxes[i])
        
        with open("./processing/output.csv", 'a', newline='') as csvfile:
            write = csv.DictWriter(csvfile, fieldnames=fieldnames)
            for bbox in final_bboxes:
                x, y, w, h = bbox
                write.writerow({'frame': frame_count, 'x': x, 'y': y, 'w': w, 'h': h})

        frame_count += 1
