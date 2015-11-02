.PHONY: clean coverage develop env extras package release test virtualenv build_ext shell docs

SRC_ROOT := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
PYMODULE := lsh_hdc
AUTHOR := Eugene Scherba
SHELL_PRELOAD := $(SRC_ROOT)/$(PYMODULE)/metrics.py
EXTENSION := $(PYMODULE)/ext.so
EXTENSION_INTERMEDIATE := $(PYMODULE)/ext.cpp
EXTENSION_DEPS := $(PYMODULE)/ext.pyx
PYPI_HOST := pypi
DISTRIBUTE := sdist bdist_wheel
EXTRAS_REQS := dev-requirements.txt $(wildcard extras-*-requirements.txt)

PYENV := . env/bin/activate;
PYTHON := $(PYENV) python
PIP := $(PYENV) pip


include study/build.mk

doc_sources:
	sphinx-apidoc -A "$(AUTHOR)" -e -F -o docs $(PYMODULE)
	git checkout docs/conf.py
	git checkout docs/Makefile

docs:
	cd docs; make html; cd ..
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
	$(PYENV) nosetests $(NOSEARGS)
	$(PYENV) py.test README.rst

shell: extras build_ext
	$(PYENV) PYTHONSTARTUP=$(SHELL_PRELOAD) ipython

extras: env/make.extras
env/make.extras: $(EXTRAS_REQS) | env
	rm -rf env/build
	$(PYENV) for req in $?; do pip install -r $$req; done
	touch $@

nuke: clean
	rm -rf *.egg *.egg-info env bin cover coverage.xml nosetests.xml

clean:
	python setup.py clean
	rm -rf dist build
	rm -f $(EXTENSION) $(EXTENSION_INTERMEDIATE)
	find . -path ./env -prune -o -type f -name "*.pyc" -or -name "*.so" -or -name "*.cpp" -exec rm -f {} \;

build_ext: $(EXTENSION)
	@echo "done building '$(EXTENSION)' extension"

$(EXTENSION): env $(EXTENSION_DEPS)
	$(PYTHON) setup.py build_ext --inplace

develop:
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
