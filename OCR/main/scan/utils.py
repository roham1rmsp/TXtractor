import cv2
import numpy as np
from sklearn.cluster import KMeans


class Process:
    __COLOR = cv2.COLOR_BGR2GRAY
    __METHODS = [cv2.MORPH_RECT,
                 cv2.MORPH_CLOSE]
    __MODE = cv2.RETR_EXTERNAL
    __PROB = 0.005

    def _smoothen(self, image):
        gray = cv2.cvtColor(image, self.__COLOR)
        kernel = cv2.getStructuringElement(self.__METHODS[0], (7, 7))
        closed = cv2.dilate(cv2.morphologyEx(image,
                                             self.__METHODS[1], kernel),
                            kernel, iterations=1)
        return cv2.GaussianBlur(closed, (5, 5), 0)

    def _get_edges(self, image):
        return cv2.Canny(image, 120, 255)

    def _extract_cnts(self, image):
        cnts, _ = cv2.findContours(image, self.__MODE, cv2.CHAIN_APPROX_SIMPLE)
        return sorted(cnts, key=cv2.contourArea, reverse=True)

    def search(self, image: np.ndarray) -> tuple:
        modified = self._smoothen(image)
        modified = self._get_edges(modified)
        cnts = self._extract_cnts(modified)
        for cnt in cnts:
            p = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, self.__PROB * p, True)
            if len(approx) == 4:
                pts = []
                for ap in approx:
                    pts.append(ap[0])
                return pts, 1
            pts = []
            for ap in approx:
                pts.append(ap[0])
            return pts, 0
        return (), -1


class Suppress:
    def __init__(self):
        self.model = KMeans(n_clusters=4)

    def update(self, pts: np.ndarray) -> np.ndarray:
        self.model.fit(pts)
        return self.model.labels_
