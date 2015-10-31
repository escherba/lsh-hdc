EXPERIMENT_ARGS := \
	--double_negs 0 --join_negs 1 --cluster_size 20 \
	--p_err 0.0 --pos_ratio 0.1

GROUP_FIELD := shingle_skip
GROUPS := 0 1 2

#GROUP_FIELD := hashfun
#GROUPS := metrohash md5 builtin cityhash

#PARAM_FIELD := sim_size
#PARAMS := $(shell for i in `seq 3 8`; do python -c "import math; print int(10 * 4 ** $$i)"; done)

PARAM_FIELD := doc_len_mean
PARAMS := $(shell for i in `seq 0 8`; do python -c "print int(2 ** $$i)"; done)

#PARAM_FIELD := cluster_size
#PARAMS := 2 3 $(shell for i in `seq 2 8`; do python -c "print int(2 ** $$i)"; done)

TRIAL_FIELD := seed
TRIALS := $(shell seq 10 15)
