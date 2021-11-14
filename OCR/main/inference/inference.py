import os
import numpy as np
from extractor import Extract
from text.separator import Cluster
from text.serialization import SaveAsText


class Infer:
    def __init__(self):
        self.path = os.path.sep.join([os.getcwd().replace(
            "inference", "images"), "experiment_1", "sample.jpg"])
        self.extractor = Extract(self.path)
        self.cluster = Cluster
        self.as_text = SaveAsText

    def extract(self):
        return self.extractor.xtract()

    def begin(self):
        coords, x_letters = self.extract()
        self.cluster = self.cluster(np.array(coords), np.array(x_letters))
        words = self.cluster.sepr()
        self.as_text = self.as_text(words)
        self.as_text.save()


if __name__ == "__main__":
    infer = Infer()
    infer.begin()
