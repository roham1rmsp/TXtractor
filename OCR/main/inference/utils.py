import os
import cv2
import pickle
import numpy as np
from spellchecker import SpellChecker

denoiser_path = os.path.sep.join([
    os.getcwd().replace(
        "inference", "models"),
    "NOISE", "denoiser", "denoiser.pickle"
])

noise_detector_path = os.path.sep.join([
    os.getcwd().replace(
        "inference", "models"),
    "NOISE", "noise_detector", "noise_detector.pickle"
])


class NoiseAbstract:
    def _pad(self, image):
        return cv2.copyMakeBorder(image, 2, 2, 2, 2,
                                  cv2.BORDER_REPLICATE)

    def _get_foreground(self, image):
        blur = cv2.medianBlur(image, 5)
        foreground = image.astype("float") - blur
        foreground[foreground > 0] = 0
        minVal = np.min(foreground)
        maxVal = np.max(foreground)
        foreground = (foreground - minVal) / (maxVal - minVal + 1e-7)
        return foreground

    def _get_rois(self, image):
        features = []
        for y in range(0, image.shape[0]):
            for x in range(0, image.shape[1]):
                roi = image[y:y + 5, x:x + 5]
                (rH, rW) = roi.shape[:2]
                if rW != 5 or rH != 5:
                    continue
                feature = roi.flatten()
                features.append(feature)
        return features


class Denoiser(NoiseAbstract):
    __denoiser_model = pickle.loads(open(denoiser_path, "rb").read())

    def denoise(self, image):
        shape = image.shape
        image = self._pad(image)
        image = self._get_foreground(image)
        features = self._get_rois(image)
        pred = self._predict_px(features)
        return pred

    def _predict_px(self, features):
        pred = self.__denoiser_model.predict(features)
        return (pred.reshape(shape) * 255.0).astype("uint8")


class NoiseDetector(NoiseAbstract):
    __noise_detector_model = pickle.loads(
        open(noise_detector_path, "rb").read())

    def detect(self, image):
        image = self._pad(image)
        image = self._get_foreground(image)
        features = self._get_rois(image)
        pred = self._predict_noise(features)
        return np.argmax(pred)

    def _predict_noise(self, features):
        pred = self.__noise_detector_model.predict(features)
        return np.array([len(pred[pred == 0]), len(pred[pred == 1])])


class Process(Denoiser, NoiseDetector):
    __kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    __clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))

    def _to_gray(self, image: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def _detect_noise(self, image: np.ndarray) -> np.ndarray:
        return self.detect(image)

    def _denoise(self, image: np.ndarray) -> np.ndarray:
        return self.denoise(image)

    def _clear(self, image: np.ndarray) -> np.ndarray:
        return cv2.fastNlMeansDenoising(image, None, 15, 7, 21)

    def _clahe(self, image: np.ndarray) -> np.ndarray:
        return self.__clahe.apply(image)

    def _blur(self, image: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(image, (5, 5), 0)

    def _find_edges(self, image: np.ndarray) -> np.ndarray:
        return cv2.Canny(image, 30, 150)

    def _close(self, image: np.ndarray) -> np.ndarray:
        return cv2.dilate(cv2.morphologyEx(image,
                                           cv2.MORPH_CLOSE, self.__kernel),
                          self.__kernel, iterations=1)


class CleanUp:
    def __init__(self):
        self.words = []
        self.spell = SpellChecker()

    def clean(self, words):
        for word in words:
            misspelled = self.spell.unknown([word])
            if len(misspelled) == 0:
                self.words.append(word)
                continue
            for word in misspelled:
                self.words.append(self.spell.correction(word))
        return self.words
