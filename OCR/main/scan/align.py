import cv2
import numpy as np


class Aligner:
    def __init__(self, pts: np.ndarray):
        self.pts = pts

    def _calculate_width(self, tl, tr):
        return np.sqrt(((tl[1]-tr[1])**2) + ((tl[0]-tr[0])**2))

    def _calculate_height(self, tl, bl):
        return np.sqrt(((tl[1]-bl[1])**2) + ((tl[0]-bl[0])**2))

    def _get_matrix(self, src, dts):
        return cv2.getPerspectiveTransform(src, dts)

    def snip(self, image):
        tl, tr, br, bl = self.pts
        width = self._calculate_width(tl, tr)
        height = self._calculate_height(tl, bl)
        dts = np.array([[0, 0], [width, 0], [width, height],
                        [0, height]], dtype=np.float32)
        src = np.array([self.pts], dtype=np.float32)
        M = self._get_matrix(src, dts)
        warped = cv2.warpPerspective(image, M, (int(width), int(height)))
        return warped
