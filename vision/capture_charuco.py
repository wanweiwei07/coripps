"""Live view of the left cutter-hand webcam with the ChArUco corners drawn;
save calibration images by key.

Hold the board (or move the hand) so that over the set of images the board
reaches every part of the picture -- corners and edges matter most, that is
where the lens distorts -- at different tilts and distances. 20-30 images.

Keys (in the window): space saves the current image (only when enough
corners are found), q or Esc quits.

    py -3.12 coripps/vision/capture_charuco.py
"""

import os
import sys

import cv2

from coripps.vision.charuco_board import charuco_board
from coripps.vision.cameras import find_by_port


PORT = 1                        # hub port of the left camera
WINDOW = 'charuco capture'
MIN_CORNERS = 12                # fewer than this is not worth saving
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'calib_images')


def main():
    by_port = find_by_port()
    if PORT not in by_port:
        print(f'no camera on hub port {PORT} (found: {by_port})')
        return False
    os.makedirs(OUT_DIR, exist_ok=True)
    n_saved = len([f for f in os.listdir(OUT_DIR) if f.endswith('.png')])
    detector = cv2.aruco.CharucoDetector(charuco_board())
    cap = cv2.VideoCapture(by_port[PORT], cv2.CAP_DSHOW)
    if not cap.isOpened():
        print('left camera: cannot open')
        return False
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    print(f'left camera -> {OUT_DIR} ({n_saved} images already there); '
          'space saves, q quits')
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print('left camera: lost')
                return False
            corners, ids, _, _ = detector.detectBoard(frame)
            n = 0 if ids is None else len(ids)
            shown = frame.copy()
            if n:
                cv2.aruco.drawDetectedCornersCharuco(shown, corners, ids)
            color = (0, 200, 0) if n >= MIN_CORNERS else (0, 0, 255)
            cv2.putText(shown, f'corners {n}  saved {n_saved}', (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.imshow(WINDOW, shown)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                return True
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                return True
            if key == ord(' '):
                if n < MIN_CORNERS:
                    print(f'only {n} corners, not saved')
                    continue
                path = os.path.join(OUT_DIR, f'{n_saved:03d}.png')
                cv2.imwrite(path, frame)     # the raw frame, no drawing
                n_saved += 1
                print(f'{path} ({n} corners)')
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
