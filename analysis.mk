include experiment/include.mk
include experiment/misc.mk


FILENAMES := $(shell for csz in $(CLUSTER_SIZES); do for h in $(HASHES); do for s in $(SEEDS); do printf "experiment/%03d-%s-%d.json " $$csz $$h $$s; done; done; done)


experiment/%.json: experiment/misc.mk
	$(PYTHON) -m lsh_hdc.study joint $(MISC_ARGS) --output $@ \
		--cluster_size $(word 1,$(subst -, ,$*)) \
		--hashfun $(word 2,$(subst -, ,$*)) \
		--seed $(word 3,$(subst -, ,$*))

experiment/summary.json: $(FILENAMES) | experiment/include.mk
	mkdir -p $(dir $@)
	cat $^ > $@

experiment/summary.csv: experiment/summary.json
	mkdir -p $(dir $@)
	$(PYTHON) -m lsh_hdc.study summary --input $< --output $(dir $@)

analysis: experiment/summary.csv
	@echo "all done"
