EXPERIMENT_ARGS := --pos_ratio 0.05 --p_err 0.05 --cluster_size 8 --seq_len_mean 4

GROUP_FIELD := hashfun
GROUPS := metrohash md5 builtin cityhash

PARAM_FIELD := sim_size
PARAMS := $(shell for i in `seq 2 7`; do python -c "import math; print int(100 * 100 ** math.log($$i,3))"; done)

#PARAM_FIELD := seq_len_mean
#PARAMS := $(shell for i in `seq 0 6`; do python -c "print int(2 ** $$i)"; done)
