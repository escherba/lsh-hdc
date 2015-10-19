CLUSTER_SIZES := $(shell for i in `seq 1 6`; do python -c "print 2 ** $$i"; done)
SEEDS := $(shell seq 1 5)
HASHES := metrohash md5
