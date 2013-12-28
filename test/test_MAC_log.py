__author__ = 'escherba'

import unittest
import sys
import json
import operator
from utils import sort_by_length
from collections import defaultdict

import lsh
from utils import uniq_rev_index


class TestMacLog(unittest.TestCase):

    def test_mac_log(self):
        cluster_builder = lsh.Cluster(threshold=0.50)
        shingler = lsh.Shingler(4)

        posts_to_shingles = {}
        with open("data/detail.log.1") as mac_log:
            for line_num, line in enumerate(mac_log):
                if not line_num % 1000:
                    sys.stderr.write("Processing line " + str(line_num) + "\n")
                json_obj = json.loads(line)
                obj = json_obj.get("object", {})
                content = obj.get("content")
                post_id = obj.get("post_id")
                shingles = shingler.get_shingles(content)
                # TODO: no need for condition below
                if len(shingles) > 0:
                    cluster_builder.add_set(shingles, post_id)
                    posts_to_shingles[post_id] = shingles
                #if line_num > 1000:
                #    break

        sets = cluster_builder.get_sets()
        min_cluster_size = 4
        bnmi = cluster_builder.calculate_bnmi(sets, posts_to_shingles,
                                              min_cluster_size=min_cluster_size)
        cluster_sizes = map(len, filter(lambda x: len(x) > min_cluster_size, sets))
        num_clusters = len(cluster_sizes)
        points_in_clusters = sum(cluster_sizes)
        sys.stderr.write(json.dumps(
            {"num_clusters": num_clusters,
            "points_in_clusters": points_in_clusters,
            "bnmi": bnmi}) + "\n")

        # clusters: cluster_id -> [ post_ids ]
        clusters = dict(enumerate(sort_by_length(sets)))
        self.output_clusters(clusters)

    def output_clusters(self, clusters, min_cluster_size=2):

        # reverse_index: post_id -> cluster_id
        reverse_index = uniq_rev_index(clusters)

        out = defaultdict(list)

        with open("data/detail.log.1") as mac_log:
            for line_num, line in enumerate(mac_log):
                #if not line_num % 1000:
                #    print "Reading line " + str(line_num)
                json_obj = json.loads(line)
                obj = json_obj.get("object", {})
                content = obj.get("content")
                post_id = obj.get("post_id")
                try:
                    impermium = json_obj\
                        .get("impermium", [])[1]\
                        .get("4.0")
                except:
                    impermium = None
                cluster_id = reverse_index.get(post_id)
                if not cluster_id is None:
                    cluster = clusters.get(cluster_id)
                    if not cluster is None:
                        if len(cluster) >= min_cluster_size:
                            out[cluster_id].append({"cluster_id": cluster_id,
                                                    "post_id": post_id,
                                                    "content": content,
                                                    "impermium": impermium})
                #if line_num > 1000:
                #    break

        sorted_list = list({"cluster_id": k, "length": l, "posts": v} for k, v, l
                           in sorted(((k, v, len(v)) for k, v in out.items()),
                                     key=operator.itemgetter(2), reverse=True))
        print json.dumps(sorted_list)


if __name__ == '__main__':
    unittest.main()
