.PHONY: clean virtualenv upgrade test package dev

PYENV = . env/bin/activate;
PYTHON = $(PYENV) python
CUSTOM_PKG_REPO=http://packages.livefyre.com/buildout/packages/

MAC_LOG = data/2014-01-18.detail.sorted.10000.gz

test_mrdomino: dev
	$(PYTHON_TIMED) scripts/mrdomino_cluster.py $(MAC_LOG_LARGE)
	$(PYTHON) -m lflearn.cluster.eval_clusters \
		--ground $(MAC_LOG) \
		--clusters out/reduce.out

package: env
	$(PYTHON) setup.py bdist_wheel
	$(PYTHON) setup.py sdist

test: dev
	$(PYENV) nosetests --with-doctest $(NOSEARGS)

dev: env requirements-tests.txt
	$(PYENV) pip install -e . -r requirements-tests.txt

clean:
	$(PYTHON) setup.py clean
	rm -rf build dist
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
