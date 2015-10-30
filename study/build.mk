# disable built-in rules
.SUFFIXES:

# Experiment-independent defaults
METRICS := \
	homogeneity completeness nmi_score \
	adjusted_rand_index kappa0 kappa1 \
	mi_corr1 mi_corr0 mi_corr \
	adjusted_jaccard_coeff adjusted_sokal_sneath_coeff \
	adjusted_rogers_tanimoto_coeff adjusted_gower_legendre_coeff \
	informedness markedness matthews_corr \
	roc_max_info aul_score roc_auc \
	jaccard_coeff \
	time_cpu

SIMUL_CLUST_ANALY_ARGS := --label_negatives 1 --sim_size 100000 --metrics $(METRICS)

TRIAL_FIELD := seed
TRIALS := $(shell seq 10 15)

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

# create array of intermediate file names
TRIAL_RESULTS := $(shell \
	for group in $(GROUPS); do \
	for param in $(PARAMS); do \
	for trial in $(TRIALS); do \
		echo "$(EXPERIMENT)/$$group-$$param-$$trial.json"; \
	done; done; done)

.INTERMEDIATE: $(TRIAL_RESULTS)

$(EXPERIMENT)/%.json: $(EXPERIMENT)/config.mk
	@mkdir -p $(@D)
	@$(PYTHON) -m lsh_hdc.study simul_clust_analy \
		$(SIMUL_CLUST_ANALY_ARGS) $(EXPERIMENT_ARGS) \
		--$(GROUP_FIELD) $(word 1,$(subst -, ,$*)) \
		--$(PARAM_FIELD) $(word 2,$(subst -, ,$*)) \
		--$(TRIAL_FIELD) $(word 3,$(subst -, ,$*)) \
		--output $@

# Secondary files will be kept
.SECONDARY: $(addprefix $(EXPERIMENT)/,config.mk summary.ndjson summary.csv)

$(EXPERIMENT)/config.mk: $(CURRENT_DIR)/new_exp_config.mk
	@mkdir -p $(@D)
	@if [ ! -e "$@" ]; then cp -p $< $@; fi

$(EXPERIMENT)/summary.ndjson: $(TRIAL_RESULTS)
	@mkdir -p $(@D)
	@cat $^ > $@

# if a previous version of the target already exists,
# archive the whole directory where the target lives.
$(EXPERIMENT)/summary.csv: $(EXPERIMENT)/summary.ndjson
	@mkdir -p $(@D)
	@echo "archiving $(@D)"
	@if [ -e $@ ]; then \
		tar czf ` \
			i=0; while [ -e $(@D)-$$i.tgz ]; do i=$$(($$i+1)); done; \
			echo $(@D)-$$i.tgz; \
		` $(@D); \
	fi
	@echo "writing 'summary.csv' under $(@D)"
	@$(PYTHON) -m lsh_hdc.study summarize \
		--metrics $(METRICS) \
		--group_by $(GROUP_FIELD) \
		--x_axis $(PARAM_FIELD) \
		--fig_title "$(GROUP_FIELD)" \
		--input $< --output $(@D)

.PHONY: experiment
experiment: $(EXPERIMENT)/summary.csv
	@echo "done with $(<D)"
