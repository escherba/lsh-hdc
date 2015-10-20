# disable built-in rules
.SUFFIXES:

MKFILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
CURRENT_DIR := $(notdir $(patsubst %/,%,$(dir $(MKFILE_PATH))))

ifeq ($(OUTPUT_DIR),)
# OUTPUT_DIR has not been defined, so generate a name for
# one and include default experiment config file
OUTPUT_DIR := $(CURRENT_DIR)/out-$(shell date +%Y%m%d%H%M%S)
include $(CURRENT_DIR)/config.mk
else
# OUTPUT_DIR has been defined, include the make config from
# that directory only. It's likely that the following
# has been invoked:
#
#    OUTPUT_DIR=/my/custom/dir make analysis
#
include $(OUTPUT_DIR)/config.mk
endif


FILENAMES := $(shell for csz in $(CLUSTER_SIZES); do for h in $(HASHES); do for s in $(SEEDS); do printf "$(OUTPUT_DIR)/%03d-%s-%d.json " $$csz $$h $$s; done; done; done)


# We delete $(FILENAMES) manually for cleaner console output,
# so let Make think they are secondary targets
.SECONDARY: $(FILENAMES)

$(OUTPUT_DIR)/%.json: $(OUTPUT_DIR)/config.mk
	@mkdir -p $(dir $@)
	@$(PYTHON) -m lsh_hdc.study joint $(BUILD_ARGS) --output $@ \
		--cluster_size $(word 1,$(subst -, ,$*)) \
		--hashfun $(word 2,$(subst -, ,$*)) \
		--seed $(word 3,$(subst -, ,$*))


# Secondary files will be kept
.SECONDARY: $(OUTPUT_DIR)/config.mk $(OUTPUT_DIR)/summary.ndjson $(OUTPUT_DIR)/summary.csv

$(OUTPUT_DIR)/%.mk: $(CURRENT_DIR)/%.mk
	@mkdir -p $(dir $@)
	@echo "copying $< => $@"
	@if [ ! -e "$@" ]; then cp -p $< $@; fi

$(OUTPUT_DIR)/summary.ndjson: $(FILENAMES)
	@mkdir -p $(dir $@)
	@cat $^ > $@
	@rm -f $^

$(OUTPUT_DIR)/summary.csv: $(OUTPUT_DIR)/summary.ndjson
	@mkdir -p $(dir $@)
	@echo "writing 'summary.csv' in $(OUTPUT_DIR)"
	@$(PYTHON) -m lsh_hdc.study summary \
		--title "$(BUILD_ARGS)" \
		--input $< \
		--output $(dir $@)
	@echo "archiving $(OUTPUT_DIR)"
	@tar czf ` \
		if [ -e $(OUTPUT_DIR).tgz ]; \
			then i=0; \
			while [ -e $(OUTPUT_DIR)-$$i.tgz ]; \
				do let i++; \
			done; \
			echo $(OUTPUT_DIR)-$$i.tgz; \
		else echo $(OUTPUT_DIR).tgz; \
		fi` $(OUTPUT_DIR)

.PHONY: analysis
analysis: $(OUTPUT_DIR)/summary.csv
	@echo "done building $(OUTPUT_DIR)"
