__author__ = 'escherba'

import argparse
import sys
import json
import operator
from collections import defaultdict

from lsh import Cluster, WordShingler, gather_stats
from test.utils import uniq_rev_index, sort_by_length


class Options(argparse.Namespace):
    """Command-line option globals
    """
    file_path = "test/data/detail.log.1"
    bands = 4
    bandwidth = 3
    shingle_size = 4
    quiet = False
    user_id = False
    min_cluster = 4
    head = None

options = Options()


class TestMacLog():

    def test_mac_log(self):
        cluster_builder = Cluster(bands=options.bands,
                                  bandwidth=options.bandwidth)
        shingler = WordShingler(options.shingle_size)

        posts_to_shingles = {}
        data = {}
        with open(options.file_path) as mac_log:
            for line_num, line in enumerate(mac_log):
                if (not options.quiet) and (not line_num % 10000):
                    sys.stderr.write("Processing line " + str(line_num) + "\n")
                json_obj = json.loads(line)
                obj = json_obj["object"]
                content = obj["content"]
                post_id = obj["post_id"]
                data[post_id] = obj
                shingles = shingler.get_shingles(content)
                if options.user_id:
                    # optionally add user id as a shingle
                    shingles.add((obj["user_id"],))
                cluster_builder.add_set(shingles, post_id)
                posts_to_shingles[post_id] = shingles
                if (not options.head is None) and line_num > options.head:
                    break

        sets = cluster_builder.get_clusters()
        try:
            stats = gather_stats(sets,
                                 objects=data,
                                 shingles=posts_to_shingles,
                                 min_cluster_size=options.min_cluster)
        except ZeroDivisionError:
            stats = None
        cluster_sizes = map(len, filter(lambda x: len(x) > options.min_cluster, sets))
        num_clusters = len(cluster_sizes)
        points_in_clusters = sum(cluster_sizes)
        sys.stderr.write(json.dumps(
            {"num_clusters": num_clusters,
             "points_in_clusters": points_in_clusters,
             "stats": stats}) + "\n")

        # clusters: cluster_id -> [ post_ids ]
        clusters = dict(enumerate(sort_by_length(sets)))
        self.output_clusters(clusters)

    def output_clusters(self, clusters, min_cluster_size=2):

        # reverse_index: post_id -> cluster_id
        reverse_index = uniq_rev_index(clusters)

        out = defaultdict(list)

        with open(options.file_path) as mac_log:
            for line_num, line in enumerate(mac_log):
                #if not line_num % 1000:
                #    print "Reading line " + str(line_num)
                json_obj = json.loads(line)
                obj = json_obj["object"]
                content = obj["content"]
                post_id = obj["post_id"]
                try:
                    impermium = json_obj\
                        .get("impermium", [])[1]\
                        .get("4.0")
                except AttributeError:
                    # no impermium tags exist
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
    parser.add_argument('--bands', type=int, dest='bands', default=4,
                        help='number of bands', required=False)
    parser.add_argument('--bandwidth', type=int, dest='bandwidth', default=3,
                        help='rows per band', required=False)
    parser.add_argument('--quiet', action='store_true',
                        help='whether to be quiet', required=False)
    parser.add_argument('--user_id', action='store_true',
                        help='whether to be use user_id field', required=False)
    options = parser.parse_args()

    o = TestMacLog()
    o.test_mac_log()
