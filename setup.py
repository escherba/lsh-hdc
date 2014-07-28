import re
from functools import partial
from setuptools import setup, find_packages
from pkg_resources import resource_string

requirements = resource_string(
    __name__, 'requirements.txt').splitlines()
dev_requirements = resource_string(
    __name__, 'dev_requirements.txt').splitlines()

#EGG_FRAGMENT = re.compile(r'(.+)#egg=(\w+)((-|>=|==)?[\d\w.]+)?$')
EGG_FRAGMENT = re.compile(r'(.+)#egg=([a-z0-9_.]+)-([a-z0-9_.-]+)$')
find_egg = partial(re.search, EGG_FRAGMENT)

pkg_names = []
dep_links = []
for req in requirements:
    egg_info = find_egg(req)
    if egg_info is None:
        pkg_names.append(req)
    else:
        url, egg = egg_info.group(1, 2)
        pkg_names.append(egg)
        dep_links.append(req)

tests_require = filter(lambda r: find_egg(r) is None, dev_requirements)

print "install_requires: " + str(pkg_names)
print "dependency_links: " + str(dep_links)

setup(
    name="lsh-hdc",
    version="0.0.19",
    author="Eugene Scherba",
    author_email="escherba@livefyre.com",
    description=("Algorithms for locality-sensitive hashing on text data"),
    url='https://github.com/escherba/lsh-hdc',
    packages=find_packages(exclude=['tests', 'scripts']),
    long_description="LSH algo that uses MinHash signatures",
    install_requires=pkg_names,
    dependency_links=dep_links,
    tests_require=tests_require,
    test_suite='nose.collector',
    classifiers=[
    ],
)
