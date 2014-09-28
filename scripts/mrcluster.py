import yaml
from lflearn.content_rules import ContentFilter
from lflearn.cluster.mrcluster_mixin import MRClusterMixin
from lflearn.feature_extract import HTMLNormalizer, RegexTokenizer
from pkg_resources import resource_filename
from operator import itemgetter
from lsh_hdc.cluster import HDClustering


with open(resource_filename(__name__, 'mac-a0.yaml'), 'r') as fh:
    mac_cfg = yaml.load(fh)

hdcluster = HDClustering(cfg=mac_cfg['model'],
                         content_filter=ContentFilter(),
                         get_body=itemgetter('content'),
                         get_label=itemgetter('post_id'),
                         get_prefix=itemgetter('user_id'),
                         opts=dict(
                             normalizer=HTMLNormalizer(),
                             tokenizer=RegexTokenizer()))


class MRClusterMixinHDC(MRClusterMixin):

    @property
    def hdc(self):
        return hdcluster
