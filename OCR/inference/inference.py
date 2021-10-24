import os
import cv2
import string
import imutils
import numpy as np
import autocorrect
from tensorflow import lite
from process import Process
from imutils.contours import sort_contours


class Inference:
    def __init__(self, path: str):
        if not os.path.isfile(path):
            pass

        self.image = cv2.resize(cv2.imread(path), (640, 480))
        self.W, self.H = self.image.shape[:2][::-1]
        self.__prep = Process()
        self.__chars = list(
            string.digits + string.ascii_uppercase + string.ascii_lowercase)
        self.paths = [os.path.sep.join(name) for name in [
            ["CHARS74K", "binded_case", "binded_case.tflite"]
        ]]
        self.paths = [os.path.sep.join([os.getcwd().replace(
            "inference", "models"), path]) for path in self.paths]

        self.__model = lite.Interpreter(self.paths[0])
        self.__model.allocate_tensors()
        self.__model_in = self.__model.get_input_details()
        self.__model_out = self.__model.get_output_details()

    def __process(self, image):
        self.gray = self.__prep._to_gray(image)
        denosed = self.__prep._clahe(self.__prep._denose(self.gray))
        blurred = self.__prep._blur(denosed)
        return self.__prep._close(self.__prep._find_edges(blurred))

    def __get_contours(self):
        edges = self.__process(self.image)
        return cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)

    def __sort_cnts(self, cnts):
        return sort_contours(
            imutils.grab_contours(cnts), method="left-to-right")[0]

    def __find_grid(self, cnt):
        X, Y, W, H = cv2.boundingRect(cnt)
        if (self.W // W <= 55) and (self.H // H <= 55):
            ROI = self.gray[Y: Y + H, X: X + W]
            _, thresh = cv2.threshold(ROI, 0, 255,
                                      cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            TH, TW = thresh.shape
            if TH > TW:
                thresh = imutils.resize(thresh, width=32)

            else:
                thresh = imutils.resize(thresh, height=32)

            return thresh, (X, Y, W, H)
        return []

    def __process_grids(self, cnts):
        grids = []
        for cnt in cnts:
            grid = self.__find_grid(cnt)
            if len(grid) > 0:
                padded = self.__pad(grid[0])
                grids.append((padded, grid[1]))
        return grids

    def __pad(self, grid):
        padded = cv2.resize(grid, (64, 128))
        blank = np.ones((146, 86), np.uint8) * 255
        blank[10:138, 10:74] = padded
        cv2.waitKey(0)
        return np.float32(cv2.resize(blank, (64, 128)))

    def __predict(self, grid):
        sample = grid[0].reshape(-1, 128, 64, 1) / 255.0
        self.__model.set_tensor(self.__model_in[0]["index"], sample)
        self.__model.invoke()
        pred = np.argmax(self.__model.get_tensor(self.__model_out[0]["index"]))
        return (self.__chars[pred], grid[1])

    def __draw(self, predictions: tuple, color1: tuple,
               color2: tuple) -> np.ndarray:  # Visualize the outcome
        x, y, w, h = predictions[1]
        char = predictions[0]
        cv2.rectangle(self.image, (x, y), (x + w, y + h), color1, 2)
        cv2.putText(self.image, char, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color2)

    def _cleanup(self, phrase: str) -> string:  # Correct OCR mistakes
        pass

    def _infer(self) -> list:  # Start the inference
        cnts = self.__get_contours()
        cnts = self.__sort_cnts(cnts)
        grids = self.__process_grids(cnts)
        for grid in grids:
            pred = self.__predict(grid)
            self.__draw(pred, (0, 255, 0), (255, 255, 0))


if __name__ == "__main__":
    path = os.path.sep.join([os.getcwd().replace(
        "inference", "images"), "experiment_1", "sample.jpg"])
    inf = Inference(path)
    inf._infer()
