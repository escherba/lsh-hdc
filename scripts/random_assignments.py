import numpy as np
import random
from lsh_hdc.metrics import ClusteringMetrics
from matplotlib import pyplot as plt


class Grid(object):

    def __init__(self, n=100000, size=20, max_classes=6, seed=None):
        if seed is not None:
            random.seed(seed)
        self.population = range(max_classes)
        self.n = n
        self.size = size
        self.grid = []
        self.fill_grid()

    @staticmethod
    def draw_sample(population, size):
        sample = []
        for _ in xrange(size):
            sample.extend(random.sample(population, 1))
        return sample

    def fill_grid(self):
        grid = self.grid = []
        population, size = self.population, self.size
        for _ in xrange(self.n):
            classes = self.draw_sample(population, size)
            clusters = self.draw_sample(population, size)
            grid.append((classes, clusters))

    def best_score(self, score='matthews_corr', minimum=True):
        best_index = -1
        if minimum:
            direction = 1
            curr_score = float('inf')
        else:
            direction = -1
            curr_score = float('-inf')
        for idx, labels in enumerate(self.grid):
            cm = ClusteringMetrics.from_labels(*labels)
            conf = cm.confusion_matrix_
            method = getattr(conf, score)
            new_score = method()
            if cmp(curr_score, new_score) == direction:
                best_index = idx
                curr_score = new_score
        return (best_index, curr_score)

    def get_scores(self, score='matthews_corr', dim=1):
        arr1 = np.empty((self.n, dim), dtype=np.float32)
        for idx, labels in enumerate(self.grid):
            cm = ClusteringMetrics.from_labels(*labels)
            conf = cm.confusion_matrix_
            arr1[idx, :] = getattr(conf, score)()
        return arr1

    def scatter(self, xs, ys, xlim=None, ylim=None, title=None):
        fig, ax = plt.subplots()
        ax.scatter(xs, ys, marker='.', s=1)
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_xlim(ylim)
        if title is not None:
            ax.set_title(title)
        fig.show()
