EXP_MAPPER_ARGS := --pos_ratio 0.2

GROUP_FIELD := population_size
GROUPS := 2000 8000

PARAM_FIELD := h1_err
PARAMS := $(shell for i in `seq 0 5`; do python -c "print round(float($$i) / 5, 2)"; done)
