# disable built-in rules
.SUFFIXES:

ifeq ($(OUTPUT_DIR),)
# OUTPUT_DIR has not been defined, so generate a name for
# one and include default experiment config file
OUTPUT_DIR := experiment/out-$(shell date +%Y%m%d%H%M%S)
include experiment/conf.mk
else
# OUTPUT_DIR has been defined, include the make config from
# that directory only. It's likely that the following
# has been invoked:
#
#    OUTPUT_DIR=/my/custom/dir make analysis
#
include $(OUTPUT_DIR)/conf.mk
endif


FILENAMES := $(shell for csz in $(CLUSTER_SIZES); do for h in $(HASHES); do for s in $(SEEDS); do printf "$(OUTPUT_DIR)/%03d-%s-%d.json " $$csz $$h $$s; done; done; done)


# Intermediate files will be deleted once make is done
.INTERMEDIATE: $(FILENAMES)

$(OUTPUT_DIR)/%.json: $(OUTPUT_DIR)/conf.mk
	@mkdir -p $(dir $@)
	$(PYTHON) -m $(PYSCRIPT) --output $@ \
		--cluster_size $(word 1,$(subst -, ,$*)) \
		--hashfun $(word 2,$(subst -, ,$*)) \
		--seed $(word 3,$(subst -, ,$*))


# Secondary files will be kept
.SECONDARY: $(OUTPUT_DIR)/conf.mk $(OUTPUT_DIR)/summary.ndjson $(OUTPUT_DIR)/summary.csv

$(OUTPUT_DIR)/%.mk: experiment/%.mk
	@mkdir -p $(dir $@)
	if [ ! -f "$@" ]; then cp -p $< $@; fi

$(OUTPUT_DIR)/summary.ndjson: $(FILENAMES)
	@mkdir -p $(dir $@)
	cat $^ > $@

$(OUTPUT_DIR)/summary.csv: $(OUTPUT_DIR)/summary.ndjson
	@mkdir -p $(dir $@)
	$(PYTHON) -m lsh_hdc.study summary --input $< --output $(dir $@)

.PHONY: analysis
analysis: $(OUTPUT_DIR)/summary.csv
	@echo "compressing $(OUTPUT_DIR)..."
	tar czf $(OUTPUT_DIR).tgz $(OUTPUT_DIR)
	@echo "all done"
