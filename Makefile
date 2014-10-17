.PHONY: clean virtualenv upgrade test package dev

PYENV = . env/bin/activate;
PYTHON = $(PYENV) python
CUSTOM_PKG_REPO=http://packages.livefyre.com/buildout/packages/

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

package: env
	$(PYTHON) setup.py bdist_wheel
	$(PYTHON) setup.py sdist

test: dev
	$(PYTHON) `which nosetests` $(NOSEARGS)

dev: env requirements-tests.txt
	$(PYENV) pip install -e . -r requirements-tests.txt

clean:
	python setup.py clean
	rm -rf build dist
	rm -rf tmp/* out/*
	find . -type f -name "*.pyc" -exec rm {} \;

nuke: clean
	rm -rf *.egg *.egg-info env bin cover coverage.xml nosetests.xml

env virtualenv: env/bin/activate
env/bin/activate: requirements.txt setup.py
	test -f env/bin/activate || virtualenv --no-site-packages env
	$(PYENV) pip install -U wheel
	$(PYENV) pip install -U setuptools
	$(PYENV) pip install -e . -r requirements.txt
	touch env/bin/activate
