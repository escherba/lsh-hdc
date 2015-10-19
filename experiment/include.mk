CLUSTER_SIZES := $(shell for i in `seq 1 6`; do python -c "print 2 ** $$i"; done)
SEEDS := $(shell seq 10 15)
HASHES := metrohash md5 builtin cityhash
