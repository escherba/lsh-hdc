EXP_MAPPER_ARGS := --pos_ratio 0.2
EXP_COMPUTE_METRICS := assignment_score_nadjd assignment_score_nadj split_join_similarity_nadj

GROUP_FIELD := population_size
GROUPS := 2000 8000

STEPS=10
PARAM_FIELD := h1_err
PARAMS := $(shell python -c 'for i in xrange($(STEPS) + 1): print float(i) / $(STEPS)')
