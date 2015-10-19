include experiment/include.mk
include experiment/misc.mk

OUTPUT_DIR := experiment/out

FILENAMES := $(shell for csz in $(CLUSTER_SIZES); do for h in $(HASHES); do for s in $(SEEDS); do printf "$(OUTPUT_DIR)/%03d-%s-%d.json " $$csz $$h $$s; done; done; done)


.INTERMEDIATE: $(FILENAMES)

$(OUTPUT_DIR)/%.json: experiment/misc.mk
	@mkdir -p $(dir $@)
	echo $(word 1,$(subst -, ,$*))
	$(PYTHON) -m lsh_hdc.study joint $(MISC_ARGS) --output $@ \
		--cluster_size $(word 1,$(subst -, ,$*)) \
		--hashfun $(word 2,$(subst -, ,$*)) \
		--seed $(word 3,$(subst -, ,$*))

.SECONDARY: $(OUTPUT_DIR)/summary.ndjson $(OUTPUT_DIR)/summary.csv

$(OUTPUT_DIR)/summary.ndjson: $(FILENAMES)
	@mkdir -p $(dir $@)
	cat $^ > $@

$(OUTPUT_DIR)/summary.csv: $(OUTPUT_DIR)/summary.ndjson
	@mkdir -p $(dir $@)
	$(PYTHON) -m lsh_hdc.study summary --output $(dir $@)

analysis: $(OUTPUT_DIR)/summary.csv
	@echo "all done"

analysis_clean:
	rm -rf $(OUTPUT_DIR)
