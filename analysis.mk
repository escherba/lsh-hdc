# disable built-in rules
.SUFFIXES:

include experiment/grid.mk
include experiment/conf.mk

# If OUTPUT_DIR is not set externally, define it. This
# lets us build make targets while passing a parameter:
#
#   OUTPUT_DIR=/my/custom/dir make analysis
#
ifeq ($(OUTPUT_DIR),)
OUTPUT_DIR := experiment/out-$(shell date +%Y%m%d%H%M%S)
endif

FILENAMES := $(shell for csz in $(CLUSTER_SIZES); do for h in $(HASHES); do for s in $(SEEDS); do printf "$(OUTPUT_DIR)/%03d-%s-%d.json " $$csz $$h $$s; done; done; done)


# Intermediate files will be deleted once done
.INTERMEDIATE: $(FILENAMES)

$(OUTPUT_DIR)/%.mk: experiment/%.mk
	@mkdir -p $(dir $@)
	if [ ! -f "$@" ]; then cp -p $< $@; fi

$(OUTPUT_DIR)/%.json: $(OUTPUT_DIR)/conf.mk
	@mkdir -p $(dir $@)
	$(PYTHON) -m lsh_hdc.study joint $(MISC_ARGS) --output $@ \
		--cluster_size $(word 1,$(subst -, ,$*)) \
		--hashfun $(word 2,$(subst -, ,$*)) \
		--seed $(word 3,$(subst -, ,$*))


# Secondary files will be kept
.SECONDARY: $(OUTPUT_DIR)/grid.mk $(OUTPUT_DIR)/conf.mk $(OUTPUT_DIR)/summary.ndjson $(OUTPUT_DIR)/summary.csv

$(OUTPUT_DIR)/summary.ndjson: $(FILENAMES)
	@mkdir -p $(dir $@)
	cat $^ > $@

$(OUTPUT_DIR)/summary.csv: $(OUTPUT_DIR)/summary.ndjson
	@mkdir -p $(dir $@)
	$(PYTHON) -m lsh_hdc.study summary --input $< --output $(dir $@)

.PHONY: analysis
analysis: $(OUTPUT_DIR)/summary.csv
	@echo "all done"
