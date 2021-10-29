import os
import cv2
import string
import imutils
import numpy as np
import autocorrect
from tensorflow import lite
from utils import Process
from imutils.contours import sort_contours


class Inference:
    def __init__(self, path: str):
        if not os.path.isfile(path):
            pass

        self.image = cv2.resize(cv2.imread(path), (640, 480))
        self.W, self.H = self.image.shape[:2][::-1]
        self.__prep = Process()
        self.__chars = list(
            string.digits + string.ascii_uppercase)
        self.paths = [os.path.sep.join(name) for name in [
            ["CHARS74K", "type_clf", "base.tflite"],
            ["CHARS74K", "digit_clf", "sub2.tflite"],
            ["CHARS74K", "case_clf", "case_clf.tflite"],
            ["CHARS74K", "letter_clf", "sub1.tflite"]
        ]]
        self.paths = [os.path.sep.join([os.getcwd().replace(
            "inference", "models"), path]) for path in self.paths]

        # Binary classifier between digits and letters
        self.__base = lite.Interpreter(self.paths[0])
        self.__base.allocate_tensors()
        self.__base_in = self.__base.get_input_details()
        self.__base_out = self.__base.get_output_details()
        # Categorical classifier for 10 digits
        self.__digit_clf = lite.Interpreter(self.paths[1])
        self.__digit_clf.allocate_tensors()
        self.__digit_clf_in = self.__digit_clf.get_input_details()
        self.__digit_clf_out = self.__digit_clf.get_output_details()
        # Binary classifier between upper and lower case letters
        self.__case_clf = lite.Interpreter(self.paths[2])
        self.__case_clf.allocate_tensors()
        self.__case_clf_in = self.__case_clf.get_input_details()
        self.__case_clf_out = self.__case_clf.get_output_details()

        self.__letter_clf = lite.Interpreter(self.paths[3])
        self.__letter_clf.allocate_tensors()
        self.__letter_clf_in = self.__letter_clf.get_input_details()
        self.__letter_clf_out = self.__letter_clf.get_output_details()
        self.__SZ = (128, 64)
        self.__affine_flags = cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR

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
        blank = np.ones((154, 90), np.uint8) * 255
        blank[13:141, 13:77] = padded
        # cv2.imshow("", cv2.resize(self.__deskewed(blank), (64, 128)))
        # cv2.waitKey(0)
        return np.float32(cv2.resize(self.__deskewed(blank), (64, 128)))

    def __deskewed(self, image):
        moments = cv2.moments(image)
        if np.absolute(moments["mu02"]) < 0.01:
            return image.copy()
        skew = moments["mu11"] / moments["mu02"]
        M = np.float32([[1, skew, (-0.5 * self.__SZ[1] * skew)], [0, 1, 0]])
        img = cv2.warpAffine(image, M, self.__SZ, flags=self.__affine_flags) 
        return image

    def __predict(self, grid):
        sample = grid[0].reshape(-1, 128, 64, 1) / 255.0
        self.__base.set_tensor(self.__base_in[0]["index"], sample)
        self.__base.invoke()
        pred = np.argmax(self.__base.get_tensor(self.__base_out[0]["index"]))
        if pred:
            self.__case_clf.set_tensor(self.__case_clf_in[0]["index"], sample)
            self.__case_clf.invoke()
            pred = np.argmax(self.__case_clf.get_tensor(self.__case_clf_out[0]["index"]))
            self.__letter_clf.set_tensor(self.__letter_clf_in[0]["index"], sample)
            self.__letter_clf.invoke()
            pred = np.argmax(self.__letter_clf.get_tensor(self.__letter_clf_out[0]["index"]))
            return (self.__chars[pred + 10], grid[1])

        
        self.__digit_clf.set_tensor(self.__digit_clf_in[0]["index"], sample)
        self.__digit_clf.invoke()
        pred = np.argmax(self.__digit_clf.get_tensor(self.__digit_clf_out[0]["index"]))
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
            if pred is not None:
                self.__draw(pred, (0, 255, 0), (255, 255, 0))
        # cv2.imshow("winname", self.image)
        # cv2.waitKey(0)
        # cv2.imwrite("C:\\Users\\moeid\\Desktop\\TXractor\\TXtractor\\OCR\\images\\experiment_5\\prediction.jpg", self.image)


if __name__ == "__main__":
    path = os.path.sep.join([os.getcwd().replace(
        "inference", "images"), "experiment_1", "sample.jpg"])
    inf = Inference(path)
    inf._infer()
