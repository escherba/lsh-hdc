EXP_MAPPER_ARGS := --nclusters 20

GROUP_FIELD := population_size
GROUPS := 2000 8000 32000

STEPS=10
PARAM_FIELD := h1_err
PARAMS := $(shell python -c 'for i in xrange($(STEPS) + 1): print float(i) / $(STEPS)')
