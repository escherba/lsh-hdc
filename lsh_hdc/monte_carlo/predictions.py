import numpy as np
import os
import warnings
import random
from itertools import product, izip
from collections import defaultdict
from functools import partial
from pymaptools.iter import izip_with_cycles, isiterable
from lsh_hdc.metrics import ClusteringMetrics, ConfusionMatrix2
from lsh_hdc.ranking import RocCurve
from pymaptools.containers import labels_to_clusters, clusters_to_labels
from pymaptools.sample import discrete_sample


def get_conf(obj):
    try:
        return obj.pairwise_
    except AttributeError:
        return obj


def simulate_labeling(sample_size=2000, **kwargs):

    clusters = simulate_clustering(**kwargs)
    tuples = zip(*clusters_to_labels(clusters))
    random.shuffle(tuples)
    tuples = tuples[:sample_size]
    ltrue, lpred = zip(*tuples) or ([], [])
    return ltrue, lpred


def simulate_clustering(galpha=2, gbeta=10, nclusters=20, pos_ratio=0.2,
                        p_err=0.05, population_size=2000):
    csizes = map(int, np.random.gamma(galpha, gbeta, nclusters))
    num_pos = sum(csizes)
    if num_pos == 0:
        csizes.append(1)
        num_pos += 1
    num_neg = max(0, population_size - num_pos)
    expected_num_neg = num_pos * ((1.0 - pos_ratio) / pos_ratio)
    actual_neg_ratio = (num_neg - expected_num_neg) / float(expected_num_neg)
    if abs(actual_neg_ratio) > 0.2:
        word = "fewer" if actual_neg_ratio < 0.0 else "more"
        warnings.warn("{:.1%} {} negatives than expected. Got: {} (expected: {}. Recommended population_size: {})"
                      .format(abs(actual_neg_ratio), word, num_neg, int(expected_num_neg), int(expected_num_neg + num_pos)))

    # the larger the cluster, the more probable it is some unclustered
    # items belong to it
    dist_class_labels = {}
    total_csizes = sum(csizes) + num_neg
    for idx, csize in enumerate([num_neg] + csizes):
        p = (csize / float(total_csizes))
        dist_class_labels[idx] = p

    dist_err = {True: 1.0 - p_err, False: p_err}

    clusters = []

    # negative case first
    for _ in xrange(num_neg):
        no_error = discrete_sample(dist_err)
        class_label = 0 if no_error else discrete_sample(dist_class_labels)
        clusters.append([class_label])

    # positive cases
    for idx, csize in enumerate(csizes, start=1):
        cluster = []
        for _ in xrange(csize):
            no_error = discrete_sample(dist_err)
            class_label = idx if no_error else discrete_sample(dist_class_labels)
            cluster.append(class_label)
        clusters.append(cluster)

    idx = -1
    relabeled = []
    for cluster in clusters:
        relabeled_cluster = []
        for class_label in cluster:
            if class_label <= 0:
                class_label = idx
            relabeled_cluster.append(class_label)
            idx -= 1
        relabeled.append(relabeled_cluster)

    return relabeled


class Grid(object):

    def __init__(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

        self.max_classes = None
        self.max_counts = None
        self.n = None
        self.size = None

        self.grid = None
        self.grid_type = None
        self.get_matrix = None
        self.show_record = None

    @classmethod
    def with_clusters(cls, n=1000, size=200, max_classes=5, seed=None):
        obj = cls(seed=seed)

        obj.grid = obj.fill_clusters(max_classes=max_classes, size=size, n=n)
        obj.grid_type = 'clusters'
        obj.get_matrix = obj.matrix_from_labels
        obj.show_record = obj.show_cluster
        return obj

    @classmethod
    def with_sim_clusters(cls, n=1000, size=200, seed=None, **kwargs):
        obj = cls(seed=seed)

        obj.grid = obj.fill_sim_clusters(size=size, n=n, **kwargs)
        obj.grid_type = 'sim_clusters'
        obj.get_matrix = obj.matrix_from_labels
        obj.show_record = obj.show_cluster
        return obj

    @classmethod
    def with_matrices(cls, n=1000, max_counts=100, seed=None):
        obj = cls(seed=seed)

        obj.grid = obj.fill_matrices(max_counts=max_counts, n=n)
        obj.grid_type = 'matrices'
        obj.get_matrix = obj.matrix_from_matrices
        obj.show_record = obj.show_matrix
        return obj

    def show_matrix(self, idx, inverse=False):
        grid = self.grid
        return grid[0][idx]

    def show_cluster(self, idx, inverse=False):
        grid = self.grid
        a, b = (1, 0) if inverse else (0, 1)
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
        if self.grid_type in ['matrices']:
            for idx, tup in self.iter_grid():
                yield idx, self.matrix_from_matrices(*tup)
        elif self.grid_type in ['clusters', 'sim_clusters']:
            for idx, labels in self.iter_grid():
                yield idx, self.matrix_from_labels(*labels)

    def describe_matrices(self):
        for idx, matrix in self.iter_matrices():
            tup = tuple(get_conf(matrix).to_ccw())
            max_idx = tup.index(max(tup))
            if max_idx != 2:
                print idx, tup

    def fill_clusters(self, n=None, size=None, max_classes=None):
        if n is None:
            n = self.n
        else:
            self.n = n
        if size is None:
            size = self.size
        else:
            self.size = size
        if max_classes is None:
            max_classes = self.max_classes
        else:
            self.max_classes = max_classes

        classes = np.random.randint(
            low=0, high=max_classes, size=(n, size))
        clusters = np.random.randint(
            low=0, high=max_classes, size=(n, size))
        return classes, clusters

    def fill_sim_clusters(self, n=None, size=None, **kwargs):
        if n is None:
            n = self.n
        else:
            self.n = n
        if size is None:
            size = self.size
        else:
            self.size = size

        classes = np.empty((n, size), dtype=np.int64)
        clusters = np.empty((n, size), dtype=np.int64)
        for idx in xrange(n):
            ltrue, lpred = simulate_labeling(sample_size=size, **kwargs)
            classes[idx, :] = ltrue
            clusters[idx, :] = lpred
        return classes, clusters

    def fill_matrices(self, max_counts=None, n=None):
        if max_counts is None:
            max_counts = self.max_counts
        else:
            self.max_counts = max_counts
        if n is None:
            n = self.n
        else:
            self.n = n

        matrices = np.random.randint(
            low=0, high=max_counts, size=(n, 4))
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

    def compute(self, scores, dtype=np.float16):
        result = defaultdict(partial(np.empty, (self.n,), dtype=dtype))
        if not isiterable(scores):
            scores = [scores]
        for idx, conf in self.iter_matrices():
            for score in scores:
                score_arr = conf.get_score(score)
                if isiterable(score_arr):
                    for j, val in enumerate(score_arr):
                        result["%s-%d" % (score, j)][idx] = val
                else:
                    result[score][idx] = score_arr
        return result

    def compare(self, others, scores, dtype=np.float16, plot=False):
        result0 = self.compute(scores, dtype=dtype)

        if not isiterable(others):
            others = [others]

        result_grid = []
        for other in others:
            result1 = other.compute(scores, dtype=dtype)

            if plot:
                from matplotlib import pyplot as plt
                from palettable import colorbrewer
                colors = colorbrewer.get_map('Set1', 'qualitative', 9).mpl_colors

            result_row = {}
            for score_name, scores0 in result0.iteritems():
                scores1 = result1[score_name]
                rc = RocCurve.from_scores(scores0, scores1)
                auc_score = rc.auc_score()
                result_row[score_name] = auc_score
                if plot:
                    hmin = min(np.min(scores0), np.min(scores1))
                    hmax = max(np.max(scores0), np.max(scores1))
                    bins = np.linspace(hmin, hmax, 50)
                    plt.hist(scores0, bins, alpha=0.5, label='0', color=colors[0], edgecolor="none")
                    plt.hist(scores1, bins, alpha=0.5, label='1', color=colors[1], edgecolor="none")
                    plt.legend(loc='upper right')
                    plt.title("%s: AUC=%.4f" % (score_name, auc_score))
                    plt.show()
            result_grid.append(result_row)
        return result_grid

    def corrplot(self, compute_result, save_to):
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
