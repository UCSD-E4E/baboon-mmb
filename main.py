import sys
import cv2
import numpy as np
import os
import csv
import shutil
import subprocess
import argparse

from scipy.optimize import linear_sum_assignment

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
        "--GAMMA2_PARAM", type=float, default=0.8, help="Gamma2 parameter ([0.0, 1.0])"
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
    GAMMA2_PARAM = args.GAMMA2_PARAM

    lrmc_script = "Demo_fRMC"

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
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

    """
    # Step 1: Accumulative Multiframe Differencing
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
    Step 2 Low Rank Matrix Completion
    """
    os.makedirs("processing/lrmc", exist_ok=True)
    N = frame_count / (L * frame_rate)
    for i in range(1, frame_count + 1):
        if frame_count - i < L:
            break
        lrmc_script = f"Demo_fRMC({MAX_NITER_PARAM}, {GAMMA2_PARAM}, {i}, {N})"
        command = ["matlab", "-nodisplay", "-nosplash", "-r", f"{lrmc_script}; exit"]
        subprocess.run(command)

    for i in range(frame_count - L, frame_count + 1):
        cv2.imwrite(f"processing/lrmc/{i}.bmp", np.zeros((height, width, 3), np.uint8))

    cv2.destroyAllWindows()

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
    lrmc_images = read_images_from_directory("processing/lrmc")

    merged_images = [
        cv2.bitwise_or(amfd_image, lrmc_image)
        for amfd_image, lrmc_image in zip(amfd_images, lrmc_images)
    ]

    def find_centers_and_bboxes(image):
        contours, _ = cv2.findContours(
            image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        centers = []
        bboxes = []

        for contour in contours:
            M = cv2.moments(contour)

            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = 0, 0

            centers.append((cX, cY))

            x, y, w, h = cv2.boundingRect(contour)
            bboxes.append((x, y, w, h))

        return centers, bboxes

    gravestone = (-1, [(-1, -1)])
    objects = [gravestone for _ in range(PIPELINE_LENGTH + 1)]

    with open("labels.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        writer.writerow(["frame #", "i", "j", "x", "y", "w", "h"])

    if os.path.exists("output"):
        shutil.rmtree("output")
    os.makedirs("output")

    for abs_current_image_idx in range(len(merged_images) - PIPELINE_LENGTH):
        for i, (frame_objs) in enumerate(objects):
            if frame_objs == gravestone:
                centers_frame, bboxes_frame = find_centers_and_bboxes(
                    merged_images[abs_current_image_idx + i]
                )
                objects[i] = list(zip(centers_frame, bboxes_frame))

        current_frame_obj_to_next_frame_obj_lookup = []
        next_frames_obj_corresponences = np.zeros((len(objects[0]), PIPELINE_LENGTH))
        for rel_next_frame_idx in range(1, PIPELINE_LENGTH + 1):
            cost_matrix = np.zeros((len(objects[0]), len(objects[rel_next_frame_idx])))
            for a, (point_a, _) in enumerate(objects[0]):
                for b, (point_b, _) in enumerate(objects[rel_next_frame_idx]):
                    SabX = abs(point_a[0] - point_b[0])
                    SabY = abs(point_a[1] - point_b[1])
                    if 0 < SabX < PIPELINE_SIZE and 0 < SabY < PIPELINE_SIZE:
                        cost_matrix[a, b] = 1

            row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)
            optimal_correspondences = list(zip(row_ind, col_ind))
            current_frame_obj_to_next_frame_obj_lookup.append(optimal_correspondences)
            for a, b in optimal_correspondences:
                next_frames_obj_corresponences[a][rel_next_frame_idx - 1] = 1

        for current_frame_current_center_idx in range(len(objects[0])):
            if (
                sum(next_frames_obj_corresponences[current_frame_current_center_idx])
                > H
            ):
                pointline = [None for _ in range(PIPELINE_LENGTH + 1)]
                pointline[0] = objects[0][current_frame_current_center_idx]

                for rel_next_frame_idx, results in enumerate(
                    current_frame_obj_to_next_frame_obj_lookup
                ):
                    next_center_idx = next(
                        (
                            b
                            for a, b in results
                            if a == current_frame_current_center_idx
                        ),
                        None,
                    )
                    if next_center_idx is not None:
                        pointline[rel_next_frame_idx + 1] = objects[
                            rel_next_frame_idx + 1
                        ][next_center_idx]

                for rel_all_frame_idx in range(1, PIPELINE_LENGTH):
                    if pointline[rel_all_frame_idx] is None:
                        next_non_none_index = next(
                            (
                                i
                                for i, v in enumerate(
                                    pointline[rel_all_frame_idx + 1 :],
                                    start=rel_all_frame_idx + 1,
                                )
                                if v is not None
                            ),
                            None,
                        )
                        prev_non_none_index = next(
                            i
                            for i, v in enumerate(
                                reversed(pointline[:rel_all_frame_idx])
                            )
                            if v is not None
                        )

                        if next_non_none_index is None:
                            if prev_non_none_index > 0:
                                x_prev, y_prev = pointline[prev_non_none_index - 1][0]
                                x_cur, y_cur = pointline[prev_non_none_index][0]
                                delta_x = x_cur - x_prev
                                delta_y = y_cur - y_prev
                                extrapolated_center = (
                                    int(x_cur + delta_x),
                                    int(y_cur + delta_y),
                                )

                                bbox_prev = pointline[prev_non_none_index - 1][1]
                                bbox_cur = pointline[prev_non_none_index][1]
                                delta_bbox = [
                                    cur - prev for cur, prev in zip(bbox_cur, bbox_prev)
                                ]
                                extrapolated_bbox = [
                                    int(cur + delta)
                                    for cur, delta in zip(bbox_cur, delta_bbox)
                                ]

                                pointline[rel_all_frame_idx] = (
                                    extrapolated_center,
                                    extrapolated_bbox,
                                )
                                objects[rel_all_frame_idx].append(
                                    pointline[rel_all_frame_idx]
                                )
                        else:
                            x0, y0 = pointline[prev_non_none_index][0]
                            x1, y1 = pointline[next_non_none_index][0]
                            delta_x = (x1 - x0) / (
                                next_non_none_index - prev_non_none_index + 1
                            )
                            delta_y = (y1 - y0) / (
                                next_non_none_index - prev_non_none_index + 1
                            )
                            interpolated_center = (
                                int(x0 + delta_x),
                                int(y0 + delta_y),
                            )

                            bbox0 = pointline[prev_non_none_index][1]
                            bbox1 = pointline[next_non_none_index][1]
                            average_bbox = [
                                int((b0 + b1) / 2) for b0, b1 in zip(bbox0, bbox1)
                            ]

                            pointline[rel_all_frame_idx] = (
                                interpolated_center,
                                average_bbox,
                            )
                            objects[rel_all_frame_idx].append(
                                pointline[rel_all_frame_idx]
                            )
            else:
                objects[0][current_frame_current_center_idx] = (None, None)

        color_image = cv2.imread(f"processing/frames/{abs_current_image_idx + 1}.bmp")
        for center, bbox in objects[0]:
            if center is not None and bbox is not None:
                i, j = center
                x, y, w, h = bbox
                cv2.circle(color_image, (i, j), 1, (0, 0, 255), -1)
                cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 255, 0), 1)
                # write to csv
                with open("labels.csv", "a", newline="") as csvfile:
                    writer = csv.writer(csvfile, delimiter=",")
                    writer.writerow([abs_current_image_idx, i, j, x, y, w, h])

        cv2.imshow("Pipeline Filter", color_image)
        cv2.imwrite(f"output/{abs_current_image_idx + 1}.bmp", color_image)
        cv2.waitKey(30)

        for i in range(PIPELINE_LENGTH):
            objects[i] = objects[i + 1]

        objects[PIPELINE_LENGTH] = gravestone

    cv2.destroyAllWindows()
    csvfile.close()

    shutil.rmtree("processing")
