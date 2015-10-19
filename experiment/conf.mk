PYSCRIPT := lsh_hdc.study joint --pos_ratio 0.2 --p_err 0.1 --seq_len_min 12 --sim_size 60000
CLUSTER_SIZES := $(shell for i in `seq 1 6`; do python -c "print 2 ** $$i"; done)
SEEDS := $(shell seq 0 5)
HASHES := metrohash md5 builtin cityhash
