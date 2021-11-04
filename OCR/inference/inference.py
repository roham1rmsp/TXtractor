import os
import warnings
import numpy as np
from extractor import Extract
from text.separator import Cluster

warnings.filterwarnings("ignore")


if __name__ == "__main__":
    path = os.path.sep.join([os.getcwd().replace(
        "inference", "images"), "experiment_1", "sample.jpg"])
    inf = Extract(path)
    coords, x_letters = inf._xtract()
    cluster = Cluster(range_acc=(-20, 20), axis=1)
    groups = sorted(cluster._transform(list(zip(coords, x_letters))), key=lambda x: x[0][0][1])
    words = []
    for group in groups:
        words.append("".join(np.array(group)[:, 1]))
    print(words)
