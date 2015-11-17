EXP_MAPPER_ARGS := --h0_err 0.9 --h1_err 0.5 --pos_ratio 0.2
EXP_REDUCER_ARGS := --legend_loc "lower right" --fig_format svg

GROUP_FIELD := population_size
GROUPS := 16000 32000

PARAM_FIELD := nclusters
#PARAMS := $(shell python -c 'import numpy; print u" ".join(u"%.3f" % x for x in numpy.linspace($(START), $(STOP), $(STEPS) + 1))')a
PARAMS := $(shell for i in `seq 0 8`; do python -c "print int(1 + 2 ** $$i)"; done)
