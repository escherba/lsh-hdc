import numpy as np
import random
import os
import warnings
from lsh_hdc.metrics import ClusteringMetrics
from itertools import product
from pymaptools.iter import izip_with_cycles, isiterable


class Grid(object):

    def __init__(self, n=10000, size=20, max_classes=6, seed=None):
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

    def find_best(self, score, minimum=False):
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

    def compute(self, scores, score_dim=1, dtype=np.float16):
        result = {}
        if not isiterable(scores):
            scores = [scores]
        for score, dim in izip_with_cycles(scores, score_dim):
            result[score] = np.empty((self.n, dim), dtype=dtype)
        for idx, labels in enumerate(self.grid):
            cm = ClusteringMetrics.from_labels(*labels)
            conf = cm.confusion_matrix_
            for score in scores:
                result[score][idx, :] = getattr(conf, score)()
        return result

    def corrplot(self, compute_result, save_to, **kwargs):
        items = compute_result.items()
        if not os.path.exists(save_to):
            os.mkdir(save_to)
        elif not os.path.isdir(save_to):
            raise IOError("save_to already exists and is a file")

        seen_pairs = set()
        for (lbl1, arr1), (lbl2, arr2) in product(items, items):
            if lbl1 == lbl2:
                continue
            elif (lbl2, lbl1) in seen_pairs:
                continue
            elif (lbl1, lbl2) in seen_pairs:
                continue
            filename = "%s_vs_%s.png" % (lbl1, lbl2)
            filepath = os.path.join(save_to, filename)
            if os.path.exists(filepath):
                warnings.warn("File exists: not overwriting %s" % filepath)
                seen_pairs.add((lbl1, lbl2))
                seen_pairs.add((lbl2, lbl1))
                continue
            self.plot([(arr1, arr2)], save_to=filepath, title=filename,
                      xlabel=lbl1, ylabel=lbl2)
            seen_pairs.add((lbl1, lbl2))
            seen_pairs.add((lbl2, lbl1))

    def plot(self, pairs, xlim=None, ylim=None, title=None,
             jitter=0.001, marker='.', s=0.01, color='black', alpha=1.0,
             save_to=None, xlabel=None, ylabel=None, **kwargs):
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots()
        for (xs, ys), jitter_, marker_, s_, color_, alpha_ in \
                izip_with_cycles(pairs, jitter, marker, s, color, alpha):
            if jitter_ is not None:
                xs = np.random.normal(xs, jitter_)
                ys = np.random.normal(ys, jitter_)
            ax.scatter(xs, ys, marker=marker_, s=s_, color=color_, alpha=alpha_,
                       **kwargs)
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_xlim(ylim)
        if title is not None:
            ax.set_title(title)
        if save_to is None:
            fig.show()
        else:
            fig.savefig(save_to)
            plt.close(fig)
