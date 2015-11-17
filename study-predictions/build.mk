SHELL := /bin/bash

# disable built-in rules
.SUFFIXES:

# Include experiment
#
MKFILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
CURRENT_DIR := $(notdir $(patsubst %/,%,$(dir $(MKFILE_PATH))))

ifeq ($(EXPERIMENT),)
# EXPERIMENT has not been defined, so generate a name for
# one and include default experiment config file
EXPERIMENT := $(CURRENT_DIR)/exp-$(shell date +%Y%m%d%H%M%S)
include $(CURRENT_DIR)/new_exp_config.mk
else
# EXPERIMENT has been defined, include the make config from
# that directory only. It's likely that the following
# has been invoked:
#
#    EXPERIMENT=/my/custom/dir make experiment
#
include $(EXPERIMENT)/config.mk
endif

# custom defines

#null  :=
space := $(null) #
comma := ,

# cartesian products on space-separated lists
define prod2
	$(shell echo {$(subst $(space),$(comma),$1)}-{$(subst $(space),$(comma),$2)})
endef
define prod3
	$(shell echo {$(subst $(space),$(comma),$1)}-{$(subst $(space),$(comma),$2)}-{$(subst $(space),$(comma),$3)})
endef



# Study definition
COMPUTE_METRICS := $(EXP_COMPUTE_METRICS) \
	entropy_metrics mic_scores \
	adjusted_rand_index matthews_corr \
	informedness markedness \
	ochiai_coeff dice_coeff \
	assignment_score split_join_similarity talburt_wang_index \
	vi_similarity \
	mt_metrics bc_metrics

PLOT_METRICS := $(EXP_PLOT_METRICS) $(COMPUTE_METRICS) \
	entropy_metrics-2 \
	mt_metrics-2 \
	bc_metrics-2 \
	mic_scores-2

MAPPER := $(PYTHON) -m lsh_hdc.monte_carlo.predictions mapper \
	--sim_size 10000 \
	--metrics $(COMPUTE_METRICS) \
	$(EXP_MAPPER_ARGS)

REDUCER := $(PYTHON) -m lsh_hdc.monte_carlo.predictions reducer \
	--metrics $(PLOT_METRICS) \
	--group_by $(GROUP_FIELD) \
	--x_axis $(PARAM_FIELD) \
	$(EXP_REDUCER_ARGS)

MAPPER_FIELDS := $(GROUP_FIELD) $(PARAM_FIELD)
FIELD_PRODUCT := $(call prod2,$(GROUPS),$(PARAMS))




MAPPER_OUTPUT := $(addprefix $(EXPERIMENT)/,$(addsuffix .json,$(FIELD_PRODUCT)))
.INTERMEDIATE: $(MAPPER_OUTPUT)

# target is #1, param array is $2
define targs
	$(shell \
	vals=($$(echo $1 | tr "-" " ")); \
	i=0; for param in $2; do \
		printf " --$$param $${vals[$$i]} "; \
		i=$$(($$i+1)); \
	done)
endef

# Mapper
$(EXPERIMENT)/%.json: $(EXPERIMENT)/config.mk
	@mkdir -p $(@D)
	@$(MAPPER) --output $@ $(call targs,$*,$(MAPPER_FIELDS))

# Secondary files will be kept
.SECONDARY: $(addprefix $(EXPERIMENT)/,config.mk summary.ndjson summary.csv)

$(EXPERIMENT)/config.mk: $(CURRENT_DIR)/new_exp_config.mk
	@mkdir -p $(@D)
	@if [ ! -e "$@" ]; then cp -p $< $@; fi

$(EXPERIMENT)/summary.ndjson: $(MAPPER_OUTPUT)
	@mkdir -p $(@D)
	@cat $^ > $@


define archive
echo "archiving $(dir $1)";
if [ -e $1 ]; then \
    dname=`dirname $1`; \
	tar czf ` \
		i=0; while [ -e $$dname-$$i.tgz ]; do i=$$(($$i+1)); done; \
		echo $$dname-$$i.tgz; \
	` $(dir $1); \
fi;
endef

# Reducer
#
# if a previous version of the target already exists,
# archive the whole directory where the target lives.
$(EXPERIMENT)/summary.csv: $(EXPERIMENT)/summary.ndjson
	@mkdir -p $(@D)
	@$(call archive,$@)
	@echo "writing 'summary.csv' under $(@D)"
	@$(REDUCER) --input $< --output $(@D)

.PHONY: experiment
experiment: $(EXPERIMENT)/summary.csv
	@echo "done with $(<D)"
