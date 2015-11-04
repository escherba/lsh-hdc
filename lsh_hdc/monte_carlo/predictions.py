import numpy as np
import os
import warnings
from itertools import product, izip
from pymaptools.iter import izip_with_cycles, isiterable
from lsh_hdc.metrics import ClusteringMetrics, ConfusionMatrix2
from pymaptools.containers import labels_to_clusters
from pymaptools.sample import discrete_sample


def get_conf(obj):
    try:
        return obj.coassoc_
    except AttributeError:
        return obj


def simulate_clustering(galpha=2, gbeta=10, nclusters=20, pos_ratio=0.2,
                        err_pos=0.1, err_neg=0.02):

    csizes = map(int, np.random.gamma(galpha, gbeta, nclusters))
    num_pos = sum(csizes)
    if num_pos == 0:
        csizes.append(1)
        num_pos += 1
    num_neg = int(num_pos * ((1.0 - pos_ratio) / pos_ratio))

    # the larger the cluster, the more probable it is some unclustered
    # items belong to it
    probas = {}
    total_csizes = sum(csizes) + num_neg
    for idx, csize in enumerate([num_neg] + csizes):
        p = (csize / float(total_csizes))
        probas[idx] = p

    clusters = []
    for idx, csize in enumerate([num_neg] + csizes):
        prev_err_total = 1.0 - probas[idx]
        err_rate = err_pos if idx > 0 else err_neg
        err_mult = err_rate / prev_err_total
        cluster_probas = {cid: p * err_mult for cid, p in probas.iteritems()}
        cluster_probas[idx] = 1.0 - err_rate
        cluster = [discrete_sample(cluster_probas) for _ in xrange(csize)]
        if idx > 0:
            clusters.append(list(cluster))
        else:
            clusters.extend([[x] for x in cluster])

    return clusters


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
            self.show_record = self.show_cluster
        elif grid_type == 'matrices':
            self.grid = self.fill_matrices()
            self.grid_type = grid_type
            self.get_matrix = self.matrix_from_matrices
            self.show_record = self.show_matrix
        else:
            raise ValueError("Unknown grid_type selection '%s'" % grid_type)

    def show_matrix(self, idx, inverse=False):
        grid = self.grid
        return grid[0][idx]

    def show_cluster(self, idx, inverse=False):
        grid = self.grid
        if inverse:
            a, b = 1, 0
        else:
            a, b = 0, 1
        return labels_to_clusters(grid[a][idx], grid[b][idx])

    def best_clustering_by_score(self, score, flip_sign=False):
        idx, val = self.find_highest(score, flip_sign)
        return {"idx": idx,
                "found": "%s = %.4f" % (score, val),
                "result": self.show_cluster(idx),
                "inverse": self.show_cluster(idx, inverse=True)}

    def matrix_from_labels(self, *args):
        ltrue, lpred = args
        return ClusteringMetrics.from_labels(ltrue, lpred)

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
            tup = tuple(get_conf(matrix).to_ccw())
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

    def find_highest(self, score, flip_sign=False):
        best_index = -1
        if flip_sign:
            direction = 1
            curr_score = float('inf')
        else:
            direction = -1
            curr_score = float('-inf')
        for idx, conf in self.iter_matrices():
            new_score = conf.get_score(score)
            if cmp(curr_score, new_score) == direction:
                best_index = idx
                curr_score = new_score
        return (best_index, curr_score)

    def find_matching_matrix(self, matches):
        for idx, mx in self.iter_matrices():
            mx = get_conf(mx)
            if matches(mx):
                return idx, mx

    def compute(self, scores, score_dim=1, dtype=np.float16):
        result = {}
        if not isiterable(scores):
            scores = [scores]
        for score, dim in izip_with_cycles(scores, score_dim):
            result[score] = np.empty((self.n, dim), dtype=dtype)
        for idx, conf in self.iter_matrices():
            for score in scores:
                result[score][idx, :] = conf.get_score(score)
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
