LSH-HDC
=======

Locality-Sensitive Hashing for High-Dimensional Clustering

.. image:: https://circleci.com/gh/escherba/lsh-hdc.png?style=shield
    :target: https://circleci.com/gh/escherba/lsh-hdc
    :alt: Tests Status

Getting started
---------------

Clone the package and create virtualenv by running the appropriate make target:

.. code-block:: bash

    cd lsh-hdc/
    make env

Authors
-------

Some original parts specific to LSH were written by Mike Starr. The algorithms
were modified for the high-dimensional clustering application on short text
content by Eugene Scherba. The ``lsh_hdc.metrics`` and other submodules were
written by Eugene Scherba. Some methods are borrowed (with tweaks) from the
Scikit-Learn project, in which case the docstring for the methods notes this.

License
-------

The current project starting from version v0.2.0 is under BSD 3-clause license
due to inclusion of parts of Scikit-Learn code, which is under the same
license.  The original code by Mike Starr as well as the current package up
until version v0.1.5 is under MIT license.
