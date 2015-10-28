import numpy as np
import os
import warnings
from itertools import izip
from lsh_hdc.metrics import ClusteringMetrics, ConfusionMatrix2
from itertools import product
from pymaptools.iter import izip_with_cycles, isiterable


class Grid(object):

    def __init__(self, grid_type='clusters', n=10000, size=20, max_classes=5,
                 max_counts=100, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.max_classes = max_classes
        self.max_counts = max_counts
        self.n = n
        self.size = size
        if grid_type == 'clusters':
            self.grid = self.fill_clusters()
            self.grid_type = grid_type
            self.get_matrix = self.matrix_from_labels
        elif grid_type == 'matrices':
            self.grid = self.fill_matrices()
            self.grid_type = grid_type
            self.get_matrix = self.matrix_from_matrices
        else:
            raise ValueError("Unknown grid_type selection '%s'" % grid_type)

    def matrix_from_labels(self, *args):
        ltrue, lpred = args
        cm = ClusteringMetrics.from_labels(ltrue, lpred)
        return cm.confusion_matrix_

    def matrix_from_matrices(self, *args):
        arr = args[0]
        return ConfusionMatrix2.from_ccw(*arr)

    def iter_grid(self):
        return enumerate(izip(*self.grid))

    iter_clusters = iter_grid

    def iter_matrices(self):
        if self.grid_type == 'matrices':
            for idx, tup in self.iter_grid():
                yield idx, self.matrix_from_matrices(*tup)
        elif self.grid_type == 'clusters':
            for idx, labels in self.iter_grid():
                yield idx, self.matrix_from_labels(*labels)

    def describe_matrices(self):
        for idx, matrix in self.iter_matrices():
            tup = tuple(matrix.to_ccw())
            max_idx = tup.index(max(tup))
            if max_idx != 2:
                print idx, tup

    def fill_clusters(self):
        classes = np.random.randint(
            low=0, high=self.max_classes, size=(self.n, self.size))
        clusters = np.random.randint(
            low=0, high=self.max_classes, size=(self.n, self.size))
        return classes, clusters

    def fill_matrices(self):
        matrices = np.random.randint(
            low=0, high=self.max_counts, size=(self.n, 4))
        return (matrices,)

    def find_best(self, score, minimum=False):
        best_index = -1
        if minimum:
            direction = 1
            curr_score = float('inf')
        else:
            direction = -1
            curr_score = float('-inf')
        for idx, conf in self.iter_matrices():
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
        for idx, conf in self.iter_matrices():
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
             save_to=None, label=None, xlabel=None, ylabel=None, **kwargs):
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots()
        for (xs, ys), jitter_, marker_, s_, color_, label_, alpha_ in \
                izip_with_cycles(pairs, jitter, marker, s, color, label, alpha):
            if jitter_ is not None:
                xs = np.random.normal(xs, jitter_)
                ys = np.random.normal(ys, jitter_)
            ax.scatter(xs, ys, marker=marker_, s=s_, color=color_,
                       alpha=alpha_, label=label_, **kwargs)

        if label:
            legend = ax.legend(loc='upper left', markerscale=80, scatterpoints=1)
            for lbl in legend.get_texts():
                lbl.set_fontsize('small')

        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        if title is not None:
            ax.set_title(title)
        if save_to is None:
            fig.show()
        else:
            fig.savefig(save_to)
            plt.close(fig)
