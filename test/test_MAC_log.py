__author__ = 'escherba'

import unittest
import json
from lsh import Cluster, word_shingle
from operator import itemgetter
import sys


class TestMacLog(unittest.TestCase):
    def test_mac_log(self):
        cluster = Cluster(threshold=0.50)
        with open("data/detail.log.1") as mac_log:
            for line_num, line in enumerate(mac_log):
                if not line_num % 1000:
                    sys.stderr.write("Processing line " + str(line_num) + "\n")
                obj = json.loads(line).get("object", {})
                content = obj.get("content")
                post_id = obj.get("post_id")
                s = word_shingle(content, 4)
                if len(s) > 0:
                    cluster.add_set(s, post_id)
                #if line_num > 10000:
                #    break

        sets = cluster.get_sets()
        clusters = dict((str(idx), val[0]) for idx, val
                        in enumerate(sorted([(s, len(s)) for s in sets],
                        key=itemgetter(1), reverse=True)))
        reverse_index = {}
        for cluster_label, post_ids in clusters.items():
            for post_id in post_ids:
                reverse_index[post_id] = cluster_label

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
