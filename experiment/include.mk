CLUSTER_SIZES := $(shell for i in {1..6}; do echo $$((2**$$i)); done)
SEEDS := $(shell echo {1..5})
HASHES := metrohash md5
