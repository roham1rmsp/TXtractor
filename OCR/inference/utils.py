import cv2
import numpy as np 


class Process:
    __kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    __clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))

    def _to_gray(self, image: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def _denose(self, image: np.ndarray) -> np.ndarray:
        return cv2.fastNlMeansDenoising(image, None, 15, 7, 21)

    def _clahe(self, image: np.ndarray) -> np.ndarray:
        return self.__clahe.apply(image)

    def _blur(self, image: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(image, (5, 5), 0)

    def _find_edges(self, image: np.ndarray) -> np.ndarray:
        return cv2.Canny(image, 30, 150)

    def _close(self, image: np.ndarray) -> np.ndarray:
        return cv2.dilate(cv2.morphologyEx(image,
                 cv2.MORPH_CLOSE, self.__kernel), self.__kernel, iterations=1)

class CleanUp:
    pass