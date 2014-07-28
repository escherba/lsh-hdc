import re
from functools import partial
from setuptools import setup, find_packages
from pkg_resources import resource_string

requirements = resource_string(
    __name__, 'requirements.txt').splitlines()
dev_requirements = resource_string(
    __name__, 'dev_requirements.txt').splitlines()

# regex for finding URLs in strings
GRUBER_URLINTEXT_PAT = re.compile(ur'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?\xab\xbb\u201c\u201d\u2018\u2019]))')
contains_url = partial(re.findall, GRUBER_URLINTEXT_PAT)
EGG_PATTERN = re.compile(ur'(.+)#egg=(\w+((-|>=|==)?[\d\w.]+)?)$')
find_eggs = partial(re.search, EGG_PATTERN)

pkg_names = []
dep_links = []
for req in requirements:
    if contains_url(req):
        url, egg = find_eggs(req).groups()[0:1]
        pkg_names.append(egg)
        dep_links.append(url)
    else:
        pkg_names.append(req)

tests_require = filter(lambda r: not contains_url(r), dev_requirements)

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
