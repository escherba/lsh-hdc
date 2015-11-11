.PHONY: clean coverage develop env extras package release test virtualenv build_ext shell docs doc_sources

PYMODULE := lsh_hdc
EXTENSION_EXT := .cpp
PYPI_HOST := pypi
DISTRIBUTE := sdist bdist_wheel
SHELL_PRELOAD := $(PYMODULE)/workspace.py

SRC_ROOT := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
SHELL_PRELOAD := $(SRC_ROOT)/$(SHELL_PRELOAD)

EXTENSION_DEPS := $(shell find $(PYMODULE) -type f -name '*.pyx')
EXTENSION_INTS := $(patsubst %.pyx,%$(EXTENSION_EXT),$(EXTENSION_DEPS))
EXTENSION_LIBS := $(patsubst %$(EXTENSION_EXT),%.so,$(EXTENSION_INTS))

EXTRAS_REQS := $(wildcard extras-*-requirements.txt)

PYENV := . env/bin/activate;
PYTHON := $(PYENV) python
PIP := $(PYENV) pip


include study-preds/build.mk

doc_sources:
	sphinx-apidoc \
		-A "`$(PYTHON) setup.py --author`" \
		-H "`$(PYTHON) setup.py --name`" \
		-V "`$(PYTHON) setup.py --version`" \
		-f -e -d 4 -F -o docs $(PYMODULE)
	-git checkout docs/conf.py
	-git checkout docs/Makefile

docs: env build_ext
	$(PYENV) cd docs; make html; cd ..
	open docs/_build/html/index.html

package: env build_ext
	$(PYTHON) setup.py $(DISTRIBUTE)

release: env build_ext
	$(PYTHON) setup.py $(DISTRIBUTE) upload -r $(PYPI_HOST)

# if in local dev on Mac, `make coverage` will run tests and open
# coverage report in the browser
ifeq ($(shell uname -s), Darwin)
coverage: test
	open cover/index.html
endif

test: env build_ext
	# make sure package can be pip-installed from local directory
	$(PIP) install -e .
	# run tests
	$(PYENV) $(ENV_EXTRA) python `which nosetests` $(NOSEARGS)

shell: extras build_ext
	$(PYENV) PYTHONSTARTUP=$(SHELL_PRELOAD) ipython

extras: env/make.extras
env/make.extras: $(EXTRAS_REQS) | env
	$(PYENV) for req in $?; do pip install -r $$req; done
	touch $@

nuke: clean
	rm -rf *.egg *.egg-info env bin cover coverage.xml nosetests.xml

clean:
	-python setup.py clean
	rm -rf dist build
	rm -f $(EXTENSION_LIBS) $(EXTENSION_INTS)
	find . -path ./env -prune -o -type f -name "*.pyc" -exec rm -f {} \;

build_ext: env
	$(PYTHON) setup.py build_ext --inplace

$(EXTENSION_LIBS): build_ext
	@echo "done building $@"

develop: build_ext
	@echo "Installing for " `which pip`
	-pip uninstall --yes $(PYMODULE)
	pip install -e .

ifeq ($(PIP_SYSTEM_SITE_PACKAGES),1)
VENV_OPTS="--system-site-packages"
else
VENV_OPTS="--no-site-packages"
endif

env virtualenv: env/bin/activate
env/bin/activate: dev-requirements.txt requirements.txt | setup.py
	test -f $@ || virtualenv $(VENV_OPTS) env
	$(PYENV) easy_install -U pip
	$(PIP) install -U wheel cython
	$(PYENV) for reqfile in $^; do pip install -r $$reqfile; done
	touch $@
