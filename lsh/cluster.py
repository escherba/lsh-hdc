from functools import partial
from pymaptools import UnionFind
from collections import defaultdict
from lsh import MinHashSignature, LSHC


class Cluster(object):
    """Clusters sets with Jaccard similarity above threshold with high
    probability.

    Algorithm based on Rajaraman, "Mining of Massive Datasets":
    1. Generate set signature
    2. Use LSH to map similar signatures to same buckets
    3. Use UnionFind to merge buckets containing same values
    """
    def __init__(self, signer=None, sketch_sim_fn=lambda x, y: True):
        self.union_find = UnionFind()
        self.signer = signer
        self.hash_map = defaultdict(list)
        self.sketch_sim_fn = sketch_sim_fn

    def add_set(self, s, label=None, sketch=None):
        # Set default label for this set
        if not label:
            label = s

        # Add to union-find structure
        uf = self.union_find
        uf.__getitem__(label)

        # Get signature vector and hash it
        keys = s \
            if self.signer is None \
            else self.signer.get_signature(s)
        label_lists = map(self.hash_map.__getitem__, keys)

        similar_to = partial(self.sketch_sim_fn, sketch)
        # Unite labels with same LSH keys
        for label_list in label_lists:
            if label_list:
                fst_label = label_list[0][0]
                good_lbl_count = \
                    len([x for x in label_list if similar_to(x[1])])
                if good_lbl_count > 0:
                    if label != fst_label:
                        label_list.append((label, sketch))
                        uf.union(fst_label, label)
            else:
                label_list.append((label, sketch))

    def get_clusters(self):
        """
        :return: a list of sets representing clusters
        :rtype: list
        """
        return self.union_find.sets()


class MinHashCluster(Cluster):
    def __init__(self, width=12, bandwidth=3, lsh_scheme="a0",
                 universe_size=None, kmin=1):
        """

        :param width: Number of bands
        :type width: int
        :param lsh_scheme: Adjusts number of combinatorial bands
        :type lsh_scheme: str
        :param bandwidth: Number of rows per band
        :type bandwidth: int
        :param universe_size: A prime number of size close to token universe
                              cardinality
        :type universe_size: long
        """
        signer = MinHashSignature(width,
                                  lsh_hasher=LSHC(bandwidth, width=width,
                                                  scheme=lsh_scheme),
                                  universe_size=universe_size,
                                  kmin=kmin)
        super(MinHashCluster, self).__init__(signer=signer)
