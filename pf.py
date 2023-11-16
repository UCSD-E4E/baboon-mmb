"""
Step 3: Pipeline Filter
"""
import csv
import os
import sys
import cv2
import numpy as np

from scipy.optimize import linear_sum_assignment

from typing import NamedTuple


class Point(NamedTuple):
    x: int
    y: int


class BoundingBox(NamedTuple):
    x: int
    y: int
    w: int
    h: int


class Object(NamedTuple):
    center: Point
    bounding_box: BoundingBox


def pf(PIPELINE_LENGTH, PIPELINE_SIZE, H, BITWISE_AND, VIDEO_FILE):
    def get_objects(frame: int) -> [Object]:
        """
        Get the objects that were detected in the frame.
        """
        amfd_image_path = f"processing/amfd/{frame}.bmp"
        lrmc_image_path = f"processing/lrmc/{frame}.bmp"

        if not os.path.exists(amfd_image_path) or not os.path.exists(lrmc_image_path):
            print(f"Frame {frame} does not exist in one of the processing paths.")
            return []

        amfd_image = cv2.imread(amfd_image_path)
        if amfd_image is None:
            print(f"Failed to load AMFD image for frame {frame}.")
            return []
        os.remove(amfd_image_path)
        amfd_image_gray = cv2.cvtColor(amfd_image, cv2.COLOR_BGR2GRAY)

        lrmc_image = cv2.imread(lrmc_image_path)
        if lrmc_image is None:
            print(f"Failed to load LRMC image for frame {frame}.")
            return []
        os.remove(lrmc_image_path)
        lrmc_image_gray = cv2.cvtColor(lrmc_image, cv2.COLOR_BGR2GRAY)

        # Merge the AMFD and LRMC images.
        merged_image = np.zeros(amfd_image_gray.shape, dtype=np.uint8)

        if BITWISE_AND:
            merged_image = cv2.bitwise_and(amfd_image_gray, lrmc_image_gray)
        else:
            merged_image = cv2.bitwise_or(amfd_image_gray, lrmc_image_gray)

        # Find the contours in the merged image.
        contours, _ = cv2.findContours(
            merged_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Create an array to store the objects detected in the frame.
        objects = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            center = Point(x + w // 2, y + h // 2)
            bounding_box = BoundingBox(x, y, w, h)
            objects.append(Object(center, bounding_box))

        return objects

    def find_object_pairs(
        pipeline_observations: [[Object]],
    ) -> [[int] * PIPELINE_LENGTH]:
        """
        Table that keeps track if an object frot he current frame is in the next PIPELINE_LENGTH frames
        The number in the table represents the index of the object in the next frame
        |   |n+1|n+2|n+3|n+4|n+5|
        |ob1| 4 | F | 2 | F | 5 |
        |ob2| F | F | F | 4 | 1 |
        |ob3| F | 0 | F | F | F |
        |ob4| F | F | 3 | F | F |
        |ob5| 5 | F | F | F | F |
        |ob6| F | 3 | 6 | F | F |
        """
        object_in_next_frames = [
            [-1 for _ in range(PIPELINE_LENGTH)]
            for _ in range(len(pipeline_observations[0]))
        ]
        for pipeline_frame_idx in range(1, PIPELINE_LENGTH + 1):
            cost_matrix = np.zeros(
                (
                    len(pipeline_observations[0]),
                    len(pipeline_observations[pipeline_frame_idx]),
                )
            )
            for a_idx, a in enumerate(pipeline_observations[0]):
                for b_idx, b in enumerate(pipeline_observations[pipeline_frame_idx]):
                    SabX = abs(a.center.x - b.center.x)
                    SabY = abs(a.center.y - b.center.y)
                    if 0 < SabX < PIPELINE_SIZE and 0 < SabY < PIPELINE_SIZE:
                        cost_matrix[a_idx][b_idx] = 1

            row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)
            for a_idx, b_idx in zip(row_ind, col_ind):
                if cost_matrix[a_idx][b_idx] == 1:
                    object_in_next_frames[a_idx][pipeline_frame_idx - 1] = b_idx

        return object_in_next_frames

    cap = cv2.VideoCapture(VIDEO_FILE)
    if not cap.isOpened():
        print("Error opening video file")
        sys.exit(1)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.release()

    # Start a CSV file to store the labels and create an ouput directory to store the images.
    with open("labels.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "x", "y", "w", "h"])
    if not os.path.exists("output"):
        os.makedirs("output")

    # Start an array that contains Object observation for the [n'th, n-1'th, ..., n-PIPELINE_LENGTH + 1'th ] frames.
    pipeline_observations = [None for _ in range(PIPELINE_LENGTH + 1)]
    observations_to_draw = [[None] for _ in range(PIPELINE_LENGTH + 1)]
    for frame_idx in range(1, frame_count + 1):
        # Get the objects detected in the frame i
        for objects_idx, objects in enumerate(pipeline_observations):
            if objects is None:
                pipeline_observations[objects_idx] = get_objects(
                    frame_idx + objects_idx
                )

        if pipeline_observations[0] != []:
            object_idx_next_frames = find_object_pairs(pipeline_observations)

            for object_idx, object in enumerate(pipeline_observations[0]):
                h = sum(1 for i in object_idx_next_frames[object_idx] if i != -1)

                if h >= H and h != PIPELINE_LENGTH:
                    pointline = [None for _ in range(PIPELINE_LENGTH + 1)]
                    pointline[0] = pipeline_observations[0][object_idx]

                    for pipeline_frame_idx in range(1, PIPELINE_LENGTH + 1):
                        if (
                            object_idx_next_frames[object_idx][pipeline_frame_idx - 1]
                            != -1
                        ):
                            pointline[pipeline_frame_idx] = pipeline_observations[
                                pipeline_frame_idx
                            ][
                                object_idx_next_frames[object_idx][
                                    pipeline_frame_idx - 1
                                ]
                            ]

                    for pipeline_frame_idx in range(1, PIPELINE_LENGTH + 1):
                        if pointline[pipeline_frame_idx] is None:
                            next_point_idx = next(
                                (
                                    i
                                    for i in range(
                                        pipeline_frame_idx + 1, PIPELINE_LENGTH + 1
                                    )
                                    if pointline[i] is not None
                                ),
                                None,
                            )

                            prev_point_idx = next(
                                (
                                    i
                                    for i in range(pipeline_frame_idx - 1, -1, -1)
                                    if pointline[i] is not None
                                ),
                                None,
                            )

                            if next_point_idx is not None:
                                x_next, y_next = pointline[next_point_idx][0]
                                x_curr, y_curr = pointline[pipeline_frame_idx - 1][0]
                                x = (x_curr + x_next) / 2
                                y = (y_curr + y_next) / 2
                                interp_point = Point(x, y)

                                bbox_next = pointline[next_point_idx][1]
                                bbox_curr = pointline[pipeline_frame_idx - 1][1]
                                x = (bbox_curr[0] + bbox_next[0]) / 2
                                y = (bbox_curr[1] + bbox_next[1]) / 2
                                w = (bbox_curr[2] + bbox_next[2]) / 2
                                h = (bbox_curr[3] + bbox_next[3]) / 2
                                interp_bbox = BoundingBox(x, y, w, h)

                                pointline[pipeline_frame_idx] = Object(
                                    interp_point, interp_bbox
                                )
                                observations_to_draw[pipeline_frame_idx].append(
                                    pointline[pipeline_frame_idx]
                                )
                            else:
                                if prev_point_idx > 0:
                                    x_prev, y_prev = pointline[prev_point_idx - 1][0]
                                    x_curr, y_curr = pointline[prev_point_idx][0]
                                    x = x_curr + (x_curr - x_prev)
                                    y = y_curr + (y_curr - y_prev)
                                    extrap_point = Point(x, y)

                                    bbox_prev = pointline[prev_point_idx - 1][1]
                                    bbox_curr = pointline[prev_point_idx][1]
                                    x = bbox_curr[0] + (bbox_curr[0] - bbox_prev[0])
                                    y = bbox_curr[1] + (bbox_curr[1] - bbox_prev[1])
                                    w = bbox_curr[2] + (bbox_curr[2] - bbox_prev[2])
                                    h = bbox_curr[3] + (bbox_curr[3] - bbox_prev[3])
                                    extrap_bbox = BoundingBox(x, y, w, h)

                                    pointline[pipeline_frame_idx] = Object(
                                        extrap_point, extrap_bbox
                                    )
                                    observations_to_draw[pipeline_frame_idx].append(
                                        pointline[pipeline_frame_idx]
                                    )
                                else:
                                    pointline[pipeline_frame_idx] = pointline[
                                        prev_point_idx
                                    ]
                                    observations_to_draw[pipeline_frame_idx].append(
                                        pointline[pipeline_frame_idx]
                                    )
                else:
                    pipeline_observations[0][object_idx] = None

            # Add current confirmed objects to observations to draw
            for object in pipeline_observations[0]:
                if object is not None:
                    # Warn if object is out of bounds
                    if (
                        object.center.x < 0
                        or object.center.x > width
                        or object.center.y < 0
                        or object.center.y > height
                    ):
                        print(
                            f"Object {object.center.x, object.center.y} is out of bounds"
                        )

                    observations_to_draw[0].append(object)

        color_image = cv2.imread(f"processing/frames/{frame_idx}.bmp")
        if color_image is None:
            print(f"Failed to load color image for frame {frame_idx}.bmp")
            break  # Skip the current iteration if the image failed to load

        for object in observations_to_draw[0]:
            if object is not None:
                x, y, w, h = object.bounding_box
                # cast to int to avoid error
                x = int(x)
                y = int(y)
                w = int(w)
                h = int(h)

                with open("labels.csv", "a") as f:
                    writer = csv.writer(f)
                    writer.writerow([frame_idx, x, y, w, h])
                # Draw the rectangle on the image
                cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imwrite(f"output/{frame_idx}.bmp", color_image)

        for objects_idx in range(PIPELINE_LENGTH):
            pipeline_observations[objects_idx] = pipeline_observations[
                objects_idx + 1
            ].copy()

        pipeline_observations[PIPELINE_LENGTH] = None

        for objects_idx in range(PIPELINE_LENGTH):
            observations_to_draw[objects_idx] = observations_to_draw[
                objects_idx + 1
            ].copy()

        observations_to_draw[PIPELINE_LENGTH] = [None]

        os.remove(f"processing/frames/{frame_idx}.bmp")

    f.close()
