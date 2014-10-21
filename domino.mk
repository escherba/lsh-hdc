MAC_LOG = data/2014-01-18.detail.sorted.10000.gz
MAC_OUT = out/reduce.out

test_mrdomino: dev
	$(PYTHON_TIMED) scripts/mrdomino_cluster.py \
		--use_domino \
		--n_concurrent_machines 4 \
		--out $(MAC_OUT) \
		$(MAC_LOG)
	$(PYTHON) -m lflearn.cluster.eval_clusters \
		--ground $(MAC_LOG) \
		--clusters $(MAC_OUT)
