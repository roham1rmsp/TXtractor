import cv2
import numpy as np
from align import Aligner
from utils import Process, Suppress


class Scanner:
    def __init__(self, image: np.ndarray):
        self.image = image
        self.prep = Process()

    def scan(self):
        pts, mode = self.prep.search(self.image)
        if mode == 1:
            pts = sorted(pts, key=lambda x: x[1])
            pts = np.concatenate((
                sorted(pts[:2], key=lambda x: x[0]),
                sorted(pts[2:], key=lambda x: x[0], reverse=True))
            )
            aligned = Aligner(pts).snip(self.image)
            return aligned
        elif not mode:
            new_pts = np.array(pts) / 255
            labels = Suppress().update(new_pts)
            new_pts = []
            for i in range(3, -1, -1):
                pt = np.mean(np.array(pts)[labels == i], axis=0)
                new_pts.append([int(pt[0]), int(pt[1])])
            new_pts = sorted(new_pts, key=lambda x: x[1])
            new_pts = np.concatenate((sorted(new_pts[:2],
                                             key=lambda x: x[0]), sorted(
                new_pts[2:], key=lambda x: x[0], reverse=True)))
            aligned = Aligner(new_pts).snip(image)
            return aligned
        return None
