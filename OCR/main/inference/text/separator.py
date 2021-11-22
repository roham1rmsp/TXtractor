import os
import sys 
sys.path.append('..')
import warnings
import numpy as np
from utils import CleanUp
from sklearn.cluster import DBSCAN

warnings.filterwarnings("ignore")

class Cluster:
    def __init__(self, pts: np.ndarray, x_letters: np.ndarray):
        self.pts = pts 
        self.x_letters = x_letters 
        self.pos = 0 
        self.words = []
        self.cleaner = CleanUp()

    def _helper_sort(self):
        def _wrapper(x):
            self.pos += 1 
            return self.pts[self.pos-1][1]
        return _wrapper

    def _sort(self):
        self.x_letters = np.array(
            sorted(self.x_letters, key=self._helper_sort())
            )
        self.pts = np.array(
            sorted(self.pts,key=lambda x:x[1])
            )

    def _scale(self):
        return (self.pts - self.pts.min()) / (self.pts.max() - self.pts.min())

    def sepr(self):
        self._sort()
        self.pts = self._scale()
        dbscan = DBSCAN(eps=0.2, min_samples=2)
        dbscan.fit(self.pts)
        labels = dbscan.labels_
        for num in np.unique(labels):
            word = "".join(self.x_letters[np.where(labels == num)])
            self.words.append(word)
        return self.cleaner.clean(self.words)
