import os
import numpy as np
from inference.extractor import Extract
from inference.text.separator import Cluster
from inference.text.serialization import SaveAsText


class Infer:
    def __init__(self, image: np.ndarray):
        self.extractor = Extract(image)
        self.cluster = Cluster
        self.as_text = SaveAsText

    def extract(self):
        return self.extractor.xtract()

    def start(self):
        coords, x_letters = self.extract()
        self.cluster = self.cluster(np.array(coords), np.array(x_letters))
        words = self.cluster.sepr()
        self.as_text = self.as_text(words)
        self.as_text.save()

