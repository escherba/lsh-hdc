.PHONY: clean virtualenv upgrade test package dev

PYENV = . env/bin/activate;
PYTHON = $(PYENV) python

package: env
	$(PYTHON) setup.py bdist_egg
	$(PYTHON) setup.py sdist

test: env dev
	$(PYENV) nosetests $(NOSEARGS)

test_mr: tests/mr_cluster_mac_log.py mrjob.conf tests/data/mac.json env dev 
	mkdir -p tests/out
	$(PYTHON) tests/mr_cluster_mac_log.py \
		-c mrjob.conf \
		tests/data/mac.json > tests/out/mr.out
	echo "Output written to tests/out/mr.out"

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
