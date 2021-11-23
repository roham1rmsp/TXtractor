import os
import cv2
import string
import imutils
import numpy as np
from tensorflow import lite
from inference.utils import Process
from imutils.contours import sort_contours


class Extract:
    def __init__(self, image):
        self.image = image
        self.W, self.H = self.image.shape[:2][::-1]
        self.image = self.image[10: self.H-10, 10: self.H-10]
        self.prep = Process()
        self.chars = list(
            string.digits + string.ascii_uppercase)
        self.paths = [os.path.sep.join(name) for name in [
            ["models", "CHARS74K", "type_clf", "base.tflite"],
            ["models", "CHARS74K", "digit_clf", "sub2.tflite"],
            ["models", "CHARS74K", "case_clf", "case_clf.tflite"],
            ["models", "CHARS74K", "letter_clf", "sub1.tflite"]
        ]]
        self.paths = [os.path.sep.join([os.getcwd().replace(
            "inference", "models"), path]) for path in self.paths]

        # Binary classifier between digits and letters
        self.base = lite.Interpreter(self.paths[0])
        self.base.allocate_tensors()
        self.base_in = self.base.get_input_details()
        self.base_out = self.base.get_output_details()
        # Categorical classifier for 10 digits
        self.digit_clf = lite.Interpreter(self.paths[1])
        self.digit_clf.allocate_tensors()
        self.digit_clf_in = self.digit_clf.get_input_details()
        self.digit_clf_out = self.digit_clf.get_output_details()
        # Binary classifier between upper and lower case letters
        self.case_clf = lite.Interpreter(self.paths[2])
        self.case_clf.allocate_tensors()
        self.case_clf_in = self.case_clf.get_input_details()
        self.case_clf_out = self.case_clf.get_output_details()

        self.letter_clf = lite.Interpreter(self.paths[3])
        self.letter_clf.allocate_tensors()
        self.letter_clf_in = self.letter_clf.get_input_details()
        self.letter_clf_out = self.letter_clf.get_output_details()
        self.SZ = (128, 64)
        self.affine_flags = cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR

    def _process(self, image):
        self.gray = self.prep._to_gray(image)
        cleared = self.prep._clahe(self.prep._clear(self.gray))
        if not self.prep._detect_noise(cleared):
            cleared = self.prep._denoise(cleared)
        blurred = self.prep._blur(cleared)
        return cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                             cv2.THRESH_BINARY_INV, 9, 2)

    def _get_contours(self):
        edges = self._process(self.image)
        return cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)

    def _sort_cnts(self, cnts):
        return sort_contours(
            imutils.grab_contours(cnts), method="left-to-right")[0]

    def _find_grid(self, cnt):
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

    def _process_grids(self, cnts):
        grids = []
        for cnt in cnts:
            grid = self._find_grid(cnt)
            if len(grid) > 0:
                padded = self._pad(grid[0])
                grids.append((padded, grid[1]))
        return grids

    def _pad(self, grid):
        padded = cv2.resize(grid, (64, 128))
        blank = np.ones((154, 90), np.uint8) * 255
        blank[13:141, 13:77] = padded
        # cv2.imshow("", cv2.resize(self._deskewed(blank), (64, 128)))
        # cv2.waitKey(0)
        return np.float32(cv2.resize(self._deskewed(blank), (64, 128)))

    def _deskewed(self, image):
        moments = cv2.moments(image)
        if np.absolute(moments["mu02"]) < 0.01:
            return image.copy()
        skew = moments["mu11"] / moments["mu02"]
        M = np.float32([[1, skew, (-0.5 * self.SZ[1] * skew)], [0, 1, 0]])
        img = cv2.warpAffine(image, M, self.SZ, flags=self.affine_flags)
        return image

    def _predict(self, grid):
        sample = grid[0].reshape(-1, 128, 64, 1) / 255.0
        self.base.set_tensor(self.base_in[0]["index"], sample)
        self.base.invoke()
        pred = np.argmax(self.base.get_tensor(self.base_out[0]["index"]))
        if pred:
            self.case_clf.set_tensor(self.case_clf_in[0]["index"], sample)
            self.case_clf.invoke()
            pred = np.argmax(self.case_clf.get_tensor(
                self.case_clf_out[0]["index"]))
            self.letter_clf.set_tensor(self.letter_clf_in[0]["index"], sample)
            self.letter_clf.invoke()
            pred = np.argmax(self.letter_clf.get_tensor(
                self.letter_clf_out[0]["index"]))
            return (self.chars[pred + 10], grid[1])

        self.digit_clf.set_tensor(self.digit_clf_in[0]["index"], sample)
        self.digit_clf.invoke()
        pred = np.argmax(self.digit_clf.get_tensor(
            self.digit_clf_out[0]["index"]))
        return (self.chars[pred], grid[1])

    def _draw(self, predictions: tuple, color1: tuple,
              color2: tuple) -> np.ndarray:  # Visualize the outcome
        x, y, w, h = predictions[1]
        char = predictions[0]
        cv2.rectangle(self.image, (x, y), (x + w, y + h), color1, 2)
        cv2.putText(self.image, char, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color2)

    def xtract(self) -> list:  # Start the prediction
        cnts = self._get_contours()
        cnts = self._sort_cnts(cnts)
        grids = self._process_grids(cnts)
        coords, x_letters = [], []
        for grid in grids:
            pred = self._predict(grid)
            pt = pred[1]
            center = list([(pt[0] + pt[2]) // 2, (pt[1] + pt[3]) // 2])
            coords.append(center)
            x_letters.append(pred[0])
        # cv2.imwrite("C:\\Users\\moeid\\Desktop\\TXractor\\TXtractor\\OCR\\images\\experiment_5\\prediction.jpg", self.image)
        return coords, x_letters
