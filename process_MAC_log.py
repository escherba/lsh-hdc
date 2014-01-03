__author__ = 'escherba'

import argparse
import sys
import json
import operator
import calendar
import dateutil.parser as dateutil_parser
from collections import defaultdict, Counter
from functools import partial
from itertools import imap, chain
from math import log
from lsh import Cluster, WordShingler
from test.utils import uniq_rev_index, sort_by_length, JsonRepr


class Options(JsonRepr):
    """Command-line option globals
    """
    file_path = "test/data/detail.log.1"
    bands = 3
    bandwidth = 3
    shingle_size = 3
    quiet = False
    no_user_id = False
    min_cluster = 6
    head = None
    #timestamp = False
    #alias = False


def mac_gather_stats(clusters, objects=None, shingles=None):
    """

    :throws ZeroDivisionError:
    :returns: Theil uncertainty index (a homogeneity measure)
    :rtype: dict
    """
    def entropy(N, n):
        """

        :param N: sample count
        :param n: number of bits
        :return: (Information) entropy
        :rtype: float
        """
        n_ = float(n)
        if n_ > 0.0:
            ratio = n_ / float(N)
            return - ratio * log(ratio)
        else:
            return 0.0

    def average(l):
        """Find average
        :param l: a list of numbers
        :type l: list
        :returns: average
        :rtype: float
        """
        xs = list(l)
        return float(reduce(lambda x, y: x + y, xs)) / float(len(xs))

    def sumsq(l):
        """Sum of squares
        :param l: a list of numbers
        :type l: list
        :returns: sum of squares
        :rtype: float
        """
        xs = list(l)
        avg = average(xs)
        return sum((el - avg) ** 2 for el in xs)

    def explained_var(l):
        """Explained variance
        :param l: a list of list
        :type l: list
        :returns: explained variance
        :rtype: float
        """
        residual_var = sum(imap(sumsq, l))
        total_var = sumsq(chain.from_iterable(l))
        return 1.0 - residual_var / total_var

    result = {}

    post_count = 0
    numerator = 0.0
    multiverse = Counter()
    all_times = []
    cluster_count = len(clusters)
    tag_counter = Counter()

    for cluster_id, cluster in enumerate(clusters):
        universe = Counter()
        times = []
        for post_id in cluster:
            if not objects is None:
                json_obj = objects[post_id]
                try:
                    tags = json_obj[u'impermium'][1][u'4.0'][u'tags']
                except KeyError:
                    tags = []
                except TypeError:
                    tags = []
                for tag in tags:
                    tag_counter[tag] += 1
                timestamp = json_obj[u'object'][u'timestamp']
                t = dateutil_parser.parse(timestamp)
                times.append(calendar.timegm(t.utctimetuple()))
            if not shingles is None:
                universe.update(shingles[post_id])

        cluster_size = len(cluster)
        post_count += cluster_size
        if not objects is None:
            all_times.append(times)
        if not shingles is None:
            numerator += sum(imap(partial(entropy, cluster_size), universe.values()))
            multiverse.update(universe)

    if clusters and (not objects is None):
        result['time_coeff'] = explained_var(all_times)

    if clusters and (not shingles is None):
        denominator = float(cluster_count) * \
            sum(imap(partial(entropy, post_count), multiverse.values()))
        if numerator > 0.0:
            uncertainty_index = 1.0 - numerator / denominator
        else:
            uncertainty_index = 1.0
        result['uncertainty_index'] = uncertainty_index

    result['num_clusters'] = cluster_count
    result['num_comments_in_clusters'] = post_count
    result['num_tags'] = tag_counter
    return result


def mac_get_post_id(obj, n):
    return obj[u'post_id'] + '.' + str(n)


class TestMacLog:

    def __init__(self, options):
        self.options = options

    def test_mac_log(self):
        options = self.options
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
                obj = json_obj[u'object']
                content = obj[u'content']
                post_id = mac_get_post_id(obj, line_num)
                data[post_id] = json_obj
                shingles = shingler.get_shingles(content)

                # optionally add user_id as a shingle
                if not options.no_user_id:
                    shingles.add((obj[u'user_id'],))

                '''
                if options.timestamp:
                    shingles.add((obj[u'timestamp'],))

                if options.alias and u'alias' in obj:
                    shingles.add((obj[u'alias'],))
                '''

                cluster_builder.add_set(shingles, post_id)
                posts_to_shingles[post_id] = shingles
                if (not options.head is None) and line_num > options.head:
                    break

        sets = filter(lambda x: len(x) >= options.min_cluster,
                      cluster_builder.get_clusters())
        try:
            stats = mac_gather_stats(sets,
                                     objects=data,
                                     shingles=posts_to_shingles)
        except ZeroDivisionError:
            stats = None
        sys.stderr.write(json.dumps(
            {"options": options.as_dict(),
             "stats": stats}) + "\n")

        # clusters: cluster_id -> [ post_ids ]
        clusters = dict(enumerate(sort_by_length(sets)))
        self.output_clusters(clusters)

    def output_clusters(self, clusters, min_cluster_size=2):

        options = self.options

        # reverse_index: post_id -> cluster_id
        reverse_index = uniq_rev_index(clusters)

        out = defaultdict(list)

        with open(options.file_path) as mac_log:
            for line_num, line in enumerate(mac_log):
                #if not line_num % 1000:
                #    print "Reading line " + str(line_num)
                json_obj = json.loads(line)
                obj = json_obj[u'object']
                post_id = mac_get_post_id(obj, line_num)
                cluster_id = reverse_index.get(post_id)
                if not cluster_id is None:
                    cluster = clusters.get(cluster_id)
                    if not cluster is None:
                        if len(cluster) >= min_cluster_size:
                            out[cluster_id].append({"cluster_id": cluster_id,
                                                    "original": json_obj})
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
    parser.add_argument('--no_user_id', action='store_true',
                        help='do not use user_id field', required=False)
    #parser.add_argument('--timestamp', action='store_true',
    #                    help='use timestamp field', required=False)
    #parser.add_argument('--alias', action='store_true',
    #                    help='use alias field', required=False)

    options = Options()
    options.assign(parser.parse_args())

    o = TestMacLog(options)
    o.test_mac_log()
