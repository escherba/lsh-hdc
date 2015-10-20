# disable built-in rules
.SUFFIXES:

MKFILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
CURRENT_DIR := $(notdir $(patsubst %/,%,$(dir $(MKFILE_PATH))))

ifeq ($(EXPERIMENT),)
# EXPERIMENT has not been defined, so generate a name for
# one and include default experiment config file
EXPERIMENT := $(CURRENT_DIR)/exp-$(shell date +%Y%m%d%H%M%S)
include $(CURRENT_DIR)/config.mk
else
# EXPERIMENT has been defined, include the make config from
# that directory only. It's likely that the following
# has been invoked:
#
#    EXPERIMENT=/my/custom/dir make analysis
#
include $(EXPERIMENT)/config.mk
endif


FILENAMES := $(shell for csz in $(CLUSTER_SIZES); do for h in $(HASHES); do for s in $(SEEDS); do printf "$(EXPERIMENT)/%03d-%s-%d.json " $$csz $$h $$s; done; done; done)


# We delete $(FILENAMES) manually for cleaner console output,
# so let Make think they are secondary targets
.SECONDARY: $(FILENAMES)

$(EXPERIMENT)/%.json: $(EXPERIMENT)/config.mk
	@mkdir -p $(dir $@)
	@$(PYTHON) -m lsh_hdc.study joint $(BUILD_ARGS) --output $@ \
		--cluster_size $(word 1,$(subst -, ,$*)) \
		--hashfun $(word 2,$(subst -, ,$*)) \
		--seed $(word 3,$(subst -, ,$*))


# Secondary files will be kept
.SECONDARY: $(EXPERIMENT)/config.mk $(EXPERIMENT)/summary.ndjson $(EXPERIMENT)/summary.csv

$(EXPERIMENT)/%.mk: $(CURRENT_DIR)/%.mk
	@mkdir -p $(dir $@)
	@echo "copying $< => $@"
	@if [ ! -e "$@" ]; then cp -p $< $@; fi

$(EXPERIMENT)/summary.ndjson: $(FILENAMES)
	@mkdir -p $(dir $@)
	@cat $^ > $@
	@rm -f $^

$(EXPERIMENT)/summary.csv: $(EXPERIMENT)/summary.ndjson
	@mkdir -p $(dir $@)
	@echo "writing 'summary.csv' in $(EXPERIMENT)"
	@$(PYTHON) -m lsh_hdc.study summary \
		--title "$(BUILD_ARGS)" \
		--input $< \
		--output $(dir $@)
	@echo "archiving $(EXPERIMENT)"
	@tar czf ` \
		if [ -e $(EXPERIMENT).tgz ]; \
			then i=0; \
			while [ -e $(EXPERIMENT)-$$i.tgz ]; \
				do let i++; \
			done; \
			echo $(EXPERIMENT)-$$i.tgz; \
		else echo $(EXPERIMENT).tgz; \
		fi` $(EXPERIMENT)

.PHONY: analysis
analysis: $(EXPERIMENT)/summary.csv
	@echo "done with $(EXPERIMENT)"
