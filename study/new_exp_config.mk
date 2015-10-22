EXPERIMENT_ARGS := --p_err 0.1 --cluster_size 8 --seq_len_min 3

GROUP_FIELD := hashfun
GROUPS := metrohash md5 builtin cityhash

PARAM_FIELD := pos_ratio
PARAMS := $(shell for i in `seq 1 6`; do python -c "print 0.5 ** $$i"; done)
