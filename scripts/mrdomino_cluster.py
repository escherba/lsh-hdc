#!/usr/bin/env python2

from lflearn.cluster.mrcluster import MRClusterMixinHDC
from mrdomino import MRJob, MRStep, protocol as mr_protocol


class MRCluster(MRClusterMixinHDC, MRJob):

    INPUT_PROTOCOL = mr_protocol.JSONValueProtocol
    INTERNAL_PROTOCOL = mr_protocol.JSONProtocol
    OUTPUT_PROTOCOL = mr_protocol.JSONValueProtocol

    def steps(self):
        return [
            MRStep(mapper=self.lsh_mapper,
                   combiner=self.lsh_combiner,
                   reducer=self.lsh_reducer),
            MRStep(mapper=self.ab_mapper,
                   reducer=self.cc_reducer),
            MRStep(mapper=self.cc_mapper,
                   combiner=self.cc_combiner,
                   reducer=self.cc_reducer),
            MRStep(mapper=self.cc_mapper,
                   combiner=self.cc_combiner,
                   reducer=self.cc_reducer),
            MRStep(mapper=self.union_mapper,
                   reducer=self.union_reducer)
        ]

    def settings(self):
        return [
            '--step_config', ['8:8', '4:4', '12:4', '4:4', '4:4']
        ]


if __name__ == '__main__':
    MRCluster.run()
