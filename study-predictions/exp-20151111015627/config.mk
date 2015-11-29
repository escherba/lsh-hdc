EXP_MAPPER_ARGS := --population_size 8000

GROUP_FIELD := nclusters
GROUPS := 20 80 320

STEPS=10
PARAM_FIELD := h1_err
PARAMS := $(shell python -c 'for i in xrange($(STEPS) + 1): print float(i) / $(STEPS)')
