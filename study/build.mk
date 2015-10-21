# disable built-in rules
.SUFFIXES:

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

FILENAMES := $(shell \
	for c in $(CLUSTER_SIZES); do \
	for h in $(HASHES); do \
	for s in $(SEEDS); do \
		printf "$(EXPERIMENT)/%s-%s-%s.json " $$c $$h $$s; \
	done; \
	done; \
	done)

# We delete $(FILENAMES) manually for cleaner console output,
# so let Make think they are secondary targets
.SECONDARY: $(FILENAMES)

$(EXPERIMENT)/%.json: $(EXPERIMENT)/config.mk
	@mkdir -p $(@D)
	@$(PYTHON) -m lsh_hdc.study joint $(BUILD_ARGS) --output $@ \
		--metrics nmi_score roc_auc time_cpu \
		--cluster_size $(word 1,$(subst -, ,$*)) \
		--hashfun $(word 2,$(subst -, ,$*)) \
		--seed $(word 3,$(subst -, ,$*))


# Secondary files will be kept
.SECONDARY: $(addprefix $(EXPERIMENT)/,config.mk summary.ndjson summary.csv)

$(EXPERIMENT)/config.mk: $(CURRENT_DIR)/new_exp_config.mk
	@mkdir -p $(@D)
	@echo "copying $< => $@"
	@if [ ! -e "$@" ]; then cp -p $< $@; fi

$(EXPERIMENT)/summary.ndjson: $(FILENAMES)
	@mkdir -p $(@D)
	@cat $^ > $@
	@rm -f $^

$(EXPERIMENT)/summary.csv: $(EXPERIMENT)/summary.ndjson
	@mkdir -p $(@D)
	@echo "archiving $(@D)"
	@# if a previous version of the target already exists,
	@# archive the whole directory where the target lives.
	@if [ -e $@ ]; then \
		tar czf ` \
			i=1; while [ -e $(@D)-$$i.tgz ]; do let i++; done; \
			echo $(@D)-$$i.tgz; \
		` $(@D); \
	fi
	@echo "writing 'summary.csv' under $(@D)"
	@$(PYTHON) -m lsh_hdc.study summary \
		--fig_title "$(BUILD_ARGS)" \
		--input $< \
		--output $(@D)

.PHONY: experiment
experiment: $(EXPERIMENT)/summary.csv
	@echo "done with $(<D)"
