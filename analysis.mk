include experiment/include.mk
include experiment/misc.mk

OUTPUT_DIR := experiment/out

FILENAMES := $(shell for csz in $(CLUSTER_SIZES); do for h in $(HASHES); do for s in $(SEEDS); do printf "experiment/%03d-%s-%d.json " $$csz $$h $$s; done; done; done)


$(OUTPUT_DIR)/%.json: experiment/misc.mk
	@mkdir -p $(dir $@)
	$(PYTHON) -m lsh_hdc.study joint $(MISC_ARGS) --output $@ \
		--cluster_size $(word 1,$(subst -, ,$*)) \
		--hashfun $(word 2,$(subst -, ,$*)) \
		--seed $(word 3,$(subst -, ,$*))

$OUTPUT_DIR)/summary.json: $(FILENAMES) | experiment/include.mk
	@mkdir -p $(dir $@)
	cat $^ > $@

$(OUTPUT_DIR)/summary.csv: $(OUTPUT_DIR)/summary.json
	@mkdir -p $(dir $@)
	$(PYTHON) -m lsh_hdc.study summary --input $< --output $(dir $@)

analysis: $(OUTPUT_DIR)/summary.csv
	@echo "all done"
