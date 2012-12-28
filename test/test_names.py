from ..lsh import Cluster, shingle

def test_names():
    """
    Should return 352 clusters of names.
    """
    names = open('data/perrys.csv', 'r').readlines()
    cluster = Cluster(threshold=0.75)
    for name in set(names):
        cluster.add_set(shingle(name, 3), name)
    assert len(cluster.get_sets()) == 352