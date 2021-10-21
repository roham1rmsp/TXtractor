import os
import cv2
import string
import imutils
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import autocorrectq
from tensorflow import lite
from imutils.contours import sort_contours


class Process:
    def _to_gray(self, image: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def _blur(self, image: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(image, (5, 5), 0)

    def _find_edges(self, image: np.ndarray) -> np.ndarray:
        return cv2.Canny(image, 30, 150)


class Inference:
    def __init__(self, path: str):
        if not os.path.isfile(path):
            pass

        self.image = cv2.resize(cv2.imread(path), (640, 480))
        self.W, self.H = self.image.shape[:2][::-1]
        self.prep = Process()

    def __process(self, image):
        self.gray = self.prep._to_gray(image)
        blurred = self.prep._blur(self.gray)
        return self.prep._find_edges(blurred)

    def __get_contours(self):
        edges = self.__process(self.image)
        return cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)

    def __sort_cnts(self, cnts):
        return sort_contours(
            imutils.grab_contours(cnts), method="left-to-right")[0]

    def __find_grid(self, cnt):
        X, Y, W, H = cv2.boundingRect(cnt)
        if (self.W // W <= 30) and (self.H // H <= 30):
            ROI = self.gray[Y: Y + H, X: X + W]
            _, thresh = cv2.threshold(ROI, 0, 255,
                                      cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
            TH, TW = thresh.shape
            if TH > TW:
                thresh = imutils.resize(thresh, width=32)

            else:
                thresh = imutils.resize(thresh, height=32)

            return thresh
        return []

    def __process_grids(self, cnts):
        grids = []
        n = 0
        for cnt in cnts:
            grid = self.__find_grid(cnt)
            if len(grid) > 0:
                TH, TW = grid.shape
                dX = int(max(0, 32 - TW) / 2.0)
                dY = int(max(0, 32 - TH) / 2.0)
                padded = self.__pad(grid, dX, dY)
                n += 1
                print(n)
                cv2.imshow("", padded)
                cv2.waitKey(0)

    def __pad(self, grid, dX, dY):
        padded = cv2.copyMakeBorder(grid, dY, dY, dX, dX, cv2.BORDER_CONSTANT)
        paded = cv2.resize(padded, (32, 32))
        return padded

    def __predict(self, grid):
        pass

    def _draw(self, predictions: list,
              color: tuple) -> np.ndarray:  # Visualize the outcome
        pass

    def _cleanup(self, phrase: str) -> string:  # Correct OCR mistakes
        pass

    def _infer(self) -> list:  # Start the inference
        cnts = self.__get_contours()
        cnts = self.__sort_cnts(cnts)
        for cnt in cnts:
        	X, Y, W, H = cv2.boundingRect(cnt)
        	if self.W * self.H // cv2.contourArea(cnt) <= 100000:
	        	cv2.rectangle(self.image, (X, Y), (X + W, Y + H), (0, 255, 0))
        cv2.imshow("", self.image)
        cv2.waitKey(0)
        # grids = self.__process_grids(cnts)


if __name__ == "__main__":
    path = os.path.sep.join([os.getcwd().replace(
        "inference", "images"), "experiment_1.jpg"])
    inf = Inference(path)
    inf._infer()
