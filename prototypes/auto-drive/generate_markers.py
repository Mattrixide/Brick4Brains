"""Generate printable ArUco 4x4_50 marker images.

Run once to create marker PNGs in the markers/ directory.
Print them out and use them to test ArUco detection.

Also generates a full-page calibration marker (ID 10) for floor
plane calibration -- prints on 8.5x11" paper.
"""

import os
import cv2
import numpy as np

MARKER_SIZE = 400  # pixels for small markers
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "markers")

# Calibration marker: fills 8.5x11" page at 300 DPI
CALIB_MARKER_ID = 10
CALIB_DPI = 300
# Page: 8.5 x 11 inches
PAGE_W_IN, PAGE_H_IN = 8.5, 11.0
# Margins: 0.5" all around for printer compatibility
MARGIN_IN = 0.5
# Marker size: 7.5" square (largest square that fits with margins)
CALIB_MARKER_IN = PAGE_W_IN - 2 * MARGIN_IN  # 7.5"
CALIB_MARKER_MM = CALIB_MARKER_IN * 25.4      # 190.5mm

# IMPORTANT: If your printer scales the output, measure the actual
# printed marker (outer black square edge to edge) and update this:
ACTUAL_PRINTED_MM = 174.6  # 6-7/8" as measured -- update if yours differs


def generate_small_markers():
    """Generate small tracking markers (IDs 0-5)."""
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

    for marker_id in range(6):
        marker_img = cv2.aruco.generateImageMarker(dictionary, marker_id, MARKER_SIZE)
        bordered = cv2.copyMakeBorder(
            marker_img, 40, 40, 40, 40,
            cv2.BORDER_CONSTANT, value=255,
        )
        path = os.path.join(OUTPUT_DIR, f"aruco_4x4_id{marker_id}.png")
        cv2.imwrite(path, bordered)
        print(f"Generated: {path}")

    print(f"Small markers: DICT_4X4_50, IDs 0-5, {MARKER_SIZE}px + 40px border")


def generate_calibration_marker():
    """Generate a full-page calibration marker for floor plane calibration.

    Prints on 8.5x11" paper at 300 DPI.
    Marker is 7.5" (190.5mm) square, centered with 0.5" margins.
    Uses ArUco ID 10 from DICT_4X4_50.
    """
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

    page_w_px = int(PAGE_W_IN * CALIB_DPI)   # 2550
    page_h_px = int(PAGE_H_IN * CALIB_DPI)   # 3300
    marker_px = int(CALIB_MARKER_IN * CALIB_DPI)  # 2250

    # Generate marker image at the right pixel size
    marker_img = cv2.aruco.generateImageMarker(dictionary, CALIB_MARKER_ID, marker_px)

    # Create white page and center the marker
    page = np.ones((page_h_px, page_w_px), dtype=np.uint8) * 255
    x_off = (page_w_px - marker_px) // 2
    y_off = (page_h_px - marker_px) // 2
    page[y_off:y_off + marker_px, x_off:x_off + marker_px] = marker_img

    # Add label text at the bottom
    label = f"ArUco 4x4_50  ID {CALIB_MARKER_ID}  |  {CALIB_MARKER_MM:.1f}mm ({CALIB_MARKER_IN:.1f}in)  |  Floor Calibration Marker"
    cv2.putText(page, label,
                (x_off, page_h_px - int(0.2 * CALIB_DPI)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2, 0, 2)

    path = os.path.join(OUTPUT_DIR, f"calibration_marker_id{CALIB_MARKER_ID}.png")
    cv2.imwrite(path, page)
    print(f"Generated: {path}")
    print(f"  Print at 100% scale on 8.5x11\" paper (300 DPI)")
    print(f"  Marker size: {CALIB_MARKER_MM:.1f}mm ({CALIB_MARKER_IN:.1f}\")")
    print(f"  ArUco ID: {CALIB_MARKER_ID}")
    return path


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    generate_small_markers()
    print()
    generate_calibration_marker()
    print(f"\nDone. Print markers from {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
