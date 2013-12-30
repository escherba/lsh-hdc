__author__ = 'escherba'

import argparse
import sys
import json
import operator
from utils import sort_by_length
from collections import defaultdict

import lsh
from utils import uniq_rev_index


class Options(argparse.Namespace):
    """Command-line option globals
    """
    file_path = "data/detail.log.1"
    width = 12
    bandwidth = 3
    shingle_size = 4
    quiet = False
    min_cluster = 4
    head = None

options = Options()


class TestMacLog():

    def test_mac_log(self):
        cluster_builder = lsh.Cluster(width=options.width,
                                      bandwidth=options.bandwidth)
        shingler = lsh.Shingler(options.shingle_size)

        posts_to_shingles = {}
        with open(options.file_path) as mac_log:
            for line_num, line in enumerate(mac_log):
                if (not options.quiet) and (not line_num % 10000):
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
                if (not options.head is None) and line_num > options.head:
                    break

        sets = cluster_builder.get_sets()
        bnmi = cluster_builder.get_uncertainty_index(sets, posts_to_shingles,
                                              min_cluster_size=options.min_cluster)
        cluster_sizes = map(len, filter(lambda x: len(x) > options.min_cluster, sets))
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

        with open("test/data/detail.log.1") as mac_log:
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
                if (not options.head is None) and line_num > options.head:
                    break

        sorted_list = list({"cluster_id": k, "length": l, "posts": v} for k, v, l
                           in sorted(((k, v, len(v)) for k, v in out.items()),
                                     key=operator.itemgetter(2), reverse=True))
        print json.dumps(sorted_list)


if __name__ == '__main__':
    """
    A sample Bash script illustrating how to run this, iterating over shingles of
    different sizes

    for i in 2 3 4 5 6 7 8
        do echo "$i"
        python process_MAC_log.py \
        --shingle_size $i \
        --quiet \
        --file data/detail.log.1 \
        | jq -c '.[].posts[] | select(.impermium.tag_details.bulk | length>0) | .post_id' \
        | wc -l
    done
    """
    parser = argparse.ArgumentParser(description='Perform clustering.')
    parser.add_argument('--file', type=str, dest='file_path', required=True,
                        help='Path to log file to process (required)')
    parser.add_argument('--head', type=int, dest='head', default=None,
                        help='how many lines from file to process (all if not set)', required=False)
    parser.add_argument('--shingle_size', type=int, dest='shingle_size', default=4,
                        help='shingle length (in tokens)', required=False)
    parser.add_argument('--min_cluster', type=int, dest='min_cluster', default=4,
                        help='minimum cluster size for quality evaluation', required=False)
    parser.add_argument('--width', type=int, dest='width', default=12,
                        help='length of signature array', required=False)
    parser.add_argument('--bandwidth', type=int, dest='bandwidth', default=3,
                        help='rows per band', required=False)
    parser.add_argument('--quiet', action='store_true',
                        help='whether to be quiet', required=False)
    options = parser.parse_args()

    o = TestMacLog()
    o.test_mac_log()
