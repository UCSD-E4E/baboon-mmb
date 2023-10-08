import sys
import cv2
from mmb import MMB

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <video file>")
        sys.exit(1)

    mmb = MMB(sys.argv[1])

    while True:
        mask, frame = mmb.process_next_frame()

        cv2.imshow("frame", frame)
        cv2.imshow("mask", mask)

        # Wait for a keypress for 30ms
        if cv2.waitKey(30) > 0:
            break

    cv2.destroyAllWindows()
