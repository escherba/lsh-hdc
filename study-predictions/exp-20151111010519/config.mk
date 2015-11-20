EXP_MAPPER_ARGS := --pos_ratio 0.2
EXP_COMPUTE_METRICS := split_join_similarity_fadj split_join_similarity_nadj split_join_similarity

GROUP_FIELD := population_size
GROUPS := 2000 8000

STEPS=10
PARAM_FIELD := h1_err
PARAMS := $(shell python -c 'for i in xrange($(STEPS) + 1): print float(i) / $(STEPS)')
