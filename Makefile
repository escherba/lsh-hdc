.PHONY: clean test

PYENV = . env/bin/activate;
PYTHON = . env/bin/activate; python

package: env
	$(PYTHON) setup.py bdist_egg
	$(PYTHON) setup.py sdist
	$(PYTHON) setup.py bdist_wheel

test: env
	$(PYTHON) setup.py test

clean:
	$(PYTHON) setup.py clean
	find . -type f -name "*.pyc" -exec rm {} \;

virtualenv: env/bin/activate
env: env/bin/activate
env/bin/activate: requirements.txt setup.py
	test -d env || virtualenv --no-site-packages env
	$(PYENV) pip install -e . -r requirements.txt
	touch env/bin/activate

upgrade: env/bin/activate
	test -d env || virtualenv --no-site-packages env
	$(PYENV) pip install -e . -r requirements.txt --upgrade
	touch env/bin/activate
