EXP_MAPPER_ARGS := --pos_ratio 0.2
EXP_COMPUTE_METRICS := ari_similarity_m1 ari_similarity_m2r ari_similarity_m2c ari_similarity_m3

GROUP_FIELD := population_size
GROUPS := 2000 8000

STEPS=10
PARAM_FIELD := h1_err
PARAMS := $(shell python -c 'for i in xrange($(STEPS) + 1): print float(i) / $(STEPS)')
