include experiment/include.mk
include experiment/misc.mk


FILENAMES := $(shell for csz in $(CLUSTER_SIZES); do for h in $(HASHES); do for s in $(SEEDS); do printf "experiment/%02d-%s-%d.json " $$csz $$h $$s; done; done; done)


experiment/%.json: experiment/misc.mk
	$(PYTHON) -m lsh_hdc.study --output $@ joint $(MISC_ARGS) \
		--cluster_size $(word 1,$(subst -, ,$*)) \
		--hashfun $(word 2,$(subst -, ,$*)) \
		--seed $(word 3,$(subst -, ,$*))


analysis: $(FILENAMES) | experiment/include.mk
	@echo "all done"
