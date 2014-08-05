.PHONY: clean virtualenv upgrade test package dev eval_clusters roc

PYENV = . env/bin/activate;
PYTHON = $(PYENV) python
PYTHON_TIMED = $(PYENV) time python
MAC_LOG = tests/data/2014-01-14.detail.sorted
MAC_OUT = tests/out/$(shell basename $(MAC_LOG)).out

package: env
	$(PYTHON) setup.py bdist_egg
	$(PYTHON) setup.py sdist

test: env dev
	$(PYENV) nosetests $(NOSEARGS)

test_mr: tests/mr_cluster_mac_log.py mrjob.conf $(MAC_LOG) env dev
	mkdir -p tests/out
	$(PYTHON_TIMED) tests/mr_cluster_mac_log.py \
		-c mrjob.conf \
		-r local \
		$(MAC_LOG) > $(MAC_OUT)
	$(PYTHON) scripts/eval_clusters.py \
		--ground $(MAC_LOG) \
		--clusters $(MAC_OUT)

roc: scripts/eval_clusters.py
	$(PYTHON) scripts/eval_clusters.py \
		--clusters $(MAC_OUT) \
		--ground $(MAC_LOG)

dev: env/bin/activate dev_requirements.txt
	$(PYENV) pip install --process-dependency-links -e . -r dev_requirements.txt

clean:
	$(PYTHON) setup.py clean
	find . -type f -name "*.pyc" -exec rm {} \;

nuke: clean
	rm -rf *.egg *.egg-info env bin cover coverage.xml nosetests.xml

env virtualenv: env/bin/activate
env/bin/activate: requirements.txt setup.py
	test -d env || virtualenv --no-site-packages env
	ln -fs env/bin .
	$(PYENV) pip install --process-dependency-links -e . -r requirements.txt
	touch env/bin/activate

upgrade:
	test -d env || virtualenv --no-site-packages env
	ln -fs env/bin .
	$(PYENV) pip install --process-dependency-links -e . -r requirements.txt --upgrade
	touch env/bin/activate
