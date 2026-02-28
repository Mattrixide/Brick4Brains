"""Generate printable ArUco 4x4_50 marker images (IDs 0-5).

Run once to create marker PNGs in the markers/ directory.
Print them out and use them to test ArUco detection.
"""

import os
import cv2

MARKER_SIZE = 400  # pixels
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "markers")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

    for marker_id in range(6):
        marker_img = cv2.aruco.generateImageMarker(dictionary, marker_id, MARKER_SIZE)
        # Add white border for easier printing/detection
        bordered = cv2.copyMakeBorder(
            marker_img, 40, 40, 40, 40,
            cv2.BORDER_CONSTANT, value=255,
        )
        path = os.path.join(OUTPUT_DIR, f"aruco_4x4_id{marker_id}.png")
        cv2.imwrite(path, bordered)
        print(f"Generated: {path}")

    print(f"\nDone. Print these markers and hold them in front of your webcam.")
    print(f"Marker dictionary: DICT_4X4_50, IDs 0-5, {MARKER_SIZE}px + 40px border")


if __name__ == "__main__":
    main()
