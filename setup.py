import re
from functools import partial
from setuptools import setup, find_packages
from pkg_resources import resource_string

get_resource = partial(resource_string, __name__)

# Regex groups: 0: URL part, 1: package name, 2: package version
find_egg = partial(
    re.search,
    re.compile(r'^(.+)#egg=([a-z0-9_.]+)-([a-z0-9_.-]+)$')
)


def process_reqs(reqs):
    """
    Add all egg-containing links to list #1 and egg information plus
    package names to list #2. Note: we rely on dependency links, support
    for which will be removed in future versions of pip.

    TODO: migrate to custom pip repo.
    """
    pkg_reqs = []
    dep_links = []
    for req in reqs:
        egg_info = find_egg(req)
        if egg_info is None:
            pkg_reqs.append(req)
        else:
            url, egg = egg_info.group(1, 2)
            pkg_reqs.append(egg)
            dep_links.append(req)
    return pkg_reqs, dep_links

requirements = get_resource('requirements.txt').splitlines()
dev_requirements = get_resource('dev_requirements.txt').splitlines()

install_requires, dep_links1 = process_reqs(requirements)
tests_require, dep_links2 = process_reqs(dev_requirements)
dependency_links = dep_links1 + dep_links2


setup(
    name="lsh-hdc",
    version="0.0.20",
    author="Eugene Scherba",
    author_email="escherba@livefyre.com",
    description=("Algorithms for locality-sensitive hashing on text data"),
    url='https://github.com/escherba/lsh-hdc',
    packages=find_packages(exclude=['tests', 'scripts']),
    long_description="LSH algo that uses MinHash signatures",
    install_requires=install_requires,
    dependency_links=dependency_links,
    tests_require=tests_require,
    test_suite='nose.collector',
    classifiers=[
    ],
)
