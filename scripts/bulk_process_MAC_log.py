#!/usr/bin/env python

__author__ = 'escherba'

import argparse
import sys
import json
from datetime import timedelta, datetime
from collections import Counter
from itertools import islice

from lsh import WordShingler
from lsh.cluster import Cluster
from lsh.stats import UncertaintySummarizer, \
    MADSummarizer, MADRatioSummarizer, median, mad
from lsh.utils import sort_by_length, JsonRepr, read_json_file


class Options(JsonRepr):
    """Command-line option globals
    """
    file_path = None
    output_path = None
    width = 9
    bandwidth = 3
    shingle_size = 3
    quiet = False
    pretty = False
    extra = False
    min_cluster = 3
    head = None

    user_id = False
    timestamp = False
    alias = False
    enduser_ip = False


def mac_get_post_id(json_obj, n):
    """Get post_id field from a JSON object"""
    return json_obj[u'object'][u'post_id'] + '.' + str(n)


class MACShingler(WordShingler):

    def __init__(self, options):
        WordShingler.__init__(self, options.shingle_size)
        self.options = options

    def shingles_from_mac(self, mac_obj):
        obj = mac_obj[u'object']
        shingles = self.get_shingles(obj[u'content'])

        if self.options.user_id:
            shingles.add((obj[u'user_id'],))

        if self.options.timestamp:
            shingles.add((obj[u'timestamp'],))

        if self.options.alias and u'alias' in obj:
            shingles.add((obj[u'alias'],))

        if self.options.enduser_ip and u'enduser_ip' in obj:
            shingles.add((obj[u'enduser_ip'],))

        return shingles


def print_time_stats(**kwargs):

    def pretty_timestamp(ts):
        """

        :param ts:
        :type ts: int
        :rtype : str
        """
        return datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

    def pretty_timedelta(td):
        """

        :param td:
        :type td: int
        :rtype : str
        """
        return str(timedelta(seconds=td))

    cluster_size = kwargs['cluster_size']
    times = kwargs['times']
    o_med = int(median(times))
    o_mad = int(mad(times))
    if kwargs['pretty']:
        summary_obj = {"cluster_size": cluster_size,
                       "timestamps_median": pretty_timestamp(o_med),
                       "timestamps_mad": pretty_timedelta(o_mad),
                       "timestamps": map(pretty_timestamp, times),
                       }
    else:
        summary_obj = {"cluster_size": cluster_size,
                       "timestamps_median": o_med,
                       "timestamps_mad": o_mad,
                       "timestamps": times,
                       }
    print json.dumps(summary_obj)


def print_mac_stats(clusters, options=None):
    """Process a bunch of clusters and print some stats
    """

    cluster_count = 0
    post_count = 0
    shingler = MACShingler(options)
    usumm = UncertaintySummarizer()
    usersumm = UncertaintySummarizer()
    varsumm = MADSummarizer()
    madsumm = MADRatioSummarizer()
    unique_posts = dict()

    for cluster in islice(clusters, 0, options.head):
        posts = cluster[u'posts']
        cluster_size = len(posts)
        if cluster_size >= options.min_cluster:
            times = []
            universe = Counter()
            user_universe = Counter()
            cluster_count += 1
            for json_obj in posts:
                try:
                    tags = json_obj[u'impermium'][1][u'4.0'][u'tags']
                except (KeyError, TypeError):
                    tags = []
                obj = json_obj[u'object']
                unique_posts[obj[u'post_id']] = tags
                times.append(int(json_obj[u'ts']))
                universe.update(shingler.shingles_from_mac(json_obj))
                user_universe[obj[u'user_id']] += 1

            # for plotting time histograms
            if options.extra:
                print_time_stats(cluster_size=cluster_size,
                                 times=times,
                                 pretty=options.pretty)

            varsumm.add_object(times)
            madsumm.add_object(times)
            usumm.add_object(universe, cluster_size)
            usersumm.add_object(user_universe, cluster_size)
            post_count += cluster_size

    # in order to avoid counting non-unique post twice,
    # collect tags separately from the main loop
    tag_counter = Counter()
    for tags in unique_posts.values():
        tag_counter.update(tags)

    print json.dumps({
        'options': options.as_dict(),
        'stats': {
            'uncertainty_index': usumm.get_summary(),
            'num_clusters': cluster_count,
            'num_comments_in_clusters': len(unique_posts),
            'impermium_tags': tag_counter,
            'user_uncertainty': usersumm.get_summary(),
            'time_mad': varsumm.get_summary(),
            'time_madratio': madsumm.get_summary()
        }
    })


def cluster_from_mac_log(options):
    """Generate a list of clusters from a MAC log and summarize them
    """
    def output_clusters(unfiltered_sets, data):
        if options.output_path:
            fh = open(options.output_path, 'w')
        else:
            fh = None
        for cluster_id, cluster in enumerate(sort_by_length(unfiltered_sets)):
            parcel = {
                "cluster_id": cluster_id,
                "length": len(cluster),
                "posts": [data[pid] for pid in cluster]
                #"posts": {post_id: data[post_id] for post_id in cluster}
            }
            if fh:
                print >>fh, json.dumps(parcel)
            yield parcel

    cluster_builder = Cluster(width=options.width,
                              bandwidth=options.bandwidth)
    shingler = MACShingler(options)

    data = {}
    with open(options.file_path) as mac_log:
        for line_num, line in enumerate(islice(mac_log, 0, options.head)):
            if (not options.quiet) and (not line_num % 10000):
                sys.stderr.write("Processing line " + str(line_num) + "\n")
            json_obj = json.loads(line)
            post_id = mac_get_post_id(json_obj, line_num)
            cluster_builder.add_set(shingler.shingles_from_mac(json_obj), post_id)
            data[post_id] = json_obj

    clusters = cluster_builder.get_clusters()
    transformed_clusers = output_clusters(clusters, data)
    print_mac_stats(transformed_clusers,
                    options=options)


def get_options(args):
    """Merge options with defaults

    :param args: command-line arguments parsed by argparse
    :type args: argparse.Namespace
    :returns: an instance of Options class
    :rtype : instance
    """
    options = Options()
    options.assign(args)
    return options


def cluster_mac_log(args):
    """Cluster entrypoint

    Processes a MAC log
    :param args: command-line arguments parsed by argparse
    :type args: argparse.Namespace
    """
    cluster_from_mac_log(get_options(args))


def summarize_mac_log(args):
    """Summary entrypoint

    Summarize an intermediate
    :param args: command-line arguments parsed by argparse
    :type args: argparse.Namespace
    """
    options = get_options(args)
    print_mac_stats(read_json_file(options.file_path),
                    options=options)


if __name__ == '__main__':
    # A sample bash-script illustrating how to run this
    #
    # python process_MAC_log.py \
    #     --shingle_size 4 \
    #     --quiet \
    #     --file data/detail.log.1 > /dev/null
    #
    parser = argparse.ArgumentParser(description='Perform clustering.')

    # add common arguments up here
    parser.add_argument('--quiet', action='store_true',
                        help='whether to be quiet', required=False)
    parser.add_argument('--min_cluster', type=int, dest='min_cluster', default=3,
                        help='minimum cluster size for quality evaluation', required=False)
    parser.add_argument('--head', type=int, dest='head', default=None,
                        help='how many lines from file to process (all if not set)', required=False)
    parser.add_argument('--file', type=str, dest='file_path', required=True,
                        help='Path to log file to process (required)')
    parser.add_argument('--extra', action='store_true',
                        help='whether to show extra output', required=False)
    parser.add_argument('--pretty', action='store_true',
                        help='whether to show pretty output', required=False)

    # for specialized functionality, use subparsers
    subparsers = parser.add_subparsers()

    # subparser: cluster
    parser_cluster = subparsers.add_parser('cluster', help='cluster a MAC log file and produce an intermediate')
    parser_cluster.add_argument('--shingle_size', type=int, dest='shingle_size', default=3,
                                help='shingle length (in tokens)', required=False)
    parser_cluster.add_argument('--width', type=int, dest='bands', default=3,
                                help='feature vector length', required=False)
    parser_cluster.add_argument('--bandwidth', type=int, dest='bandwidth', default=3,
                                help='rows per band', required=False)

    parser_cluster.add_argument('--user_id', action='store_true',
                                help='include user_id field', required=False)
    parser_cluster.add_argument('--alias', action='store_true',
                                help='include alias field', required=False)
    parser_cluster.add_argument('--timestamp', action='store_true',
                                help='include timestamp field', required=False)
    parser_cluster.add_argument('--enduser_ip', action='store_true',
                                help='include end-user IP address', required=False)

    parser_cluster.add_argument('--output', type=str, dest='output_path', required=False,
                                help='Path to output')
    parser_cluster.set_defaults(func=cluster_mac_log)

    # subparser: summary
    parser_summary = subparsers.add_parser('summary', help='summary an intermediate')
    parser_summary.set_defaults(func=summarize_mac_log)

    # standard arg processing...
    args = parser.parse_args()
    args.func(args)
