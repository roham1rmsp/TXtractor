import numpy as np

class Cluster:
    def __init__(self, range_acc: tuple=(-30,30), axis: int=0):
        self.__range_acc = range_acc
        self.__axis = axis

    def _transform(self, pts):
        groups = []
        diff = []
        for i, pt in enumerate(pts):
            axes = list(np.array(list(map(lambda p: p[0][self.__axis], pts)))-pt[0][self.__axis])
            diff.append(axes)
        binary = np.array([[1 if num in range(*self.__range_acc) else 0 for num in D] for D in diff])
        root = np.unique(binary, axis=0)
        indices = []
        for i in root:
            indices.append(np.where((binary == i).all(axis=1)))
        for part in indices:
            group = []
            part = np.array(part).ravel()
            for index in part:
                group.append(pts[index])
            groups.append(group)
                    
        return groups