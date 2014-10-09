#!/usr/bin/env python2

import sys
from lflearn.cluster.mrcluster import MRClusterMixinHDC
from mrdomino import MRJob, MRStep, MRSettings, protocol as mr_protocol


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
        return MRSettings(
            input_files=[sys.argv[1]],
            output_dir='out',
            tmp_dir='tmp',
            use_domino=True,
            n_concurrent_machines=2,
            n_shards_per_machine=3,
            step_config={
                0: dict(n_mappers=4, n_reducers=4),
                1: dict(n_mappers=2, n_reducers=2),
                2: dict(n_mappers=6, n_reducers=2),
                3: dict(n_mappers=2, n_reducers=2),
                4: dict(n_mappers=2, n_reducers=2),
            }
        )


if __name__ == '__main__':
    MRCluster.run()
