"""The ChArUco board for calibrating the cutter-hand webcams.

Printing and detecting both use the constants here, so they always agree.
Print the PDF at 100 % (no "fit to page") on A4, fix it flat on something
stiff, then measure one square with a caliper and put that in SQUARE_SIZE.

    py -3.12 coripps/vision/charuco_board.py
"""

import os

from cv2 import aruco

from one.vision.ar_marker.make_pdfboard import make_charuco_board


NROW = 10                       # squares down the (portrait A4) page
NCOLUMN = 7                     # squares across
SQUARE_SIZE = 0.025             # m -- replace with the measured print
MARKER_RATIO = .57              # marker side / square side
MARKER_DICT = aruco.DICT_4X4_250
NAME = f'charuco_{NROW}x{NCOLUMN}_{SQUARE_SIZE * 1000:g}mm'


def charuco_board():
    """The board as OpenCV detects it (sizes in m)."""
    return aruco.CharucoBoard((NCOLUMN, NROW), SQUARE_SIZE,
                              MARKER_RATIO * SQUARE_SIZE,
                              aruco.getPredefinedDictionary(MARKER_DICT))


if __name__ == '__main__':
    path = make_charuco_board(NROW, NCOLUMN, marker_dict=MARKER_DICT,
                              square_size=SQUARE_SIZE * 1000,
                              marker_ratio=MARKER_RATIO,
                              save_path=os.path.dirname(
                                  os.path.abspath(__file__)),
                              name=NAME)
    print(path)
