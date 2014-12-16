.PHONY: clean virtualenv upgrade test package dev

PYENV = . env/bin/activate;
PYTHON = $(PYENV) python
CUSTOM_PKG_REPO=http://packages.livefyre.com/buildout/packages/
EXTRAS_REQS := $(wildcard requirements-*.txt)

include domino.mk

package: env
	$(PYTHON) setup.py bdist_egg
	$(PYTHON) setup.py sdist

test: dev
	$(PYTHON) `which nosetests` $(NOSEARGS)

dev: env/make.dev
env/make.dev: $(EXTRAS_REQS) | env
	rm -rf env/build
	$(PYENV) for req in $?; do pip install -r $$req; done
	touch $@

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
	$(PYENV) pip install -U setuptools
	$(PYENV) pip install -e . -r requirements.txt
	touch env/bin/activate
