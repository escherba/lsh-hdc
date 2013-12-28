__author__ = 'escherba'

import unittest
import sys
import json
from lsh import Cluster, Shingler
from utils import uniq_rev_index, sort_by_length


class TestMacLog(unittest.TestCase):

    def test_mac_log(self):
        cluster = Cluster(threshold=0.50)
        shingler = Shingler(4)
        with open("data/detail.log.1") as mac_log:
            for line_num, line in enumerate(mac_log):
                if not line_num % 1000:
                    sys.stderr.write("Processing line " + str(line_num) + "\n")
                obj = json.loads(line).get("object", {})
                content = obj.get("content")
                post_id = obj.get("post_id")
                s = shingler.get_shingles(content)
                if len(s) > 0:
                    cluster.add_set(s, post_id)
                #if line_num > 10000:
                #    break

        # clusters: cluster_id -> [ post_ids ]
        clusters = dict(
            enumerate(
                sort_by_length(
                    cluster.get_sets())))
        self.output_pairs(clusters)

    def output_pairs(self, clusters):

        # reverse_index: post_id -> cluster_id
        reverse_index = uniq_rev_index(clusters)

        out = []
        with open("data/detail.log.1") as mac_log:
            for line_num, line in enumerate(mac_log):
                #if not line_num % 1000:
                #    print "Reading line " + str(line_num)
                obj = json.loads(line).get("object", {})
                content = obj.get("content")
                post_id = obj.get("post_id")
                cluster_label = reverse_index.get(post_id)
                if not cluster_label is None:
                    neighbors = clusters.get(cluster_label)
                    if not neighbors is None:
                        if len(neighbors) > 1:
                            out.append({"cluster_id": int(cluster_label), "content": content})

                #if line_num > 10000:
                #    break

        print json.dumps(out)
        #self.assertEqual(len(sets), 88, "failed at similarity level 25%")

if __name__ == '__main__':
    unittest.main()
