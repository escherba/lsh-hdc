import re
from setuptools import setup, find_packages
from pkg_resources import resource_string

SKIP_RE = re.compile(r'^\s*--find-links')

# Regex groups: 0: URL part, 1: package name, 2: package version
EGG_RE = re.compile(r'^(.+)#egg=([a-z0-9_.]+)-([a-z0-9_.-]+)$')

# Regex groups: 0: URL part, 1: package name, 2: branch name
URL_RE = re.compile(r'^\s*(https?://[\w\.]+.*/([^\/]+)/archive/)([^\/]+).zip$')


def process_reqs(reqs):
    pkg_reqs = []
    for req in reqs:
        if re.match(SKIP_RE, req) is not None:
            continue
        # add packages of form:
        # git+https://github.com/Livefyre/pymaptools#egg=pymaptools-0.0.3
        egg_info = re.search(EGG_RE, req)
        if egg_info is not None:
            _, egg = egg_info.group(1, 2)
            pkg_reqs.append(egg)
            continue
        # add packages of form:
        # https://github.com/escherba/matplotlib/archive/qs_fix_build.zip
        zip_info = re.search(URL_RE, req)
        if zip_info is not None:
            _, pkg = zip_info.group(1, 2)
            pkg_reqs.append(pkg)
            continue
        pkg_reqs.append(req)
    return pkg_reqs

INSTALL_REQUIRES = process_reqs(
    resource_string(__name__, 'requirements.txt').splitlines())
TESTS_REQUIRE = process_reqs(
    resource_string(__name__, 'requirements-tests.txt').splitlines())

setup(
    name="lsh-hdc",
    version="0.1.2",
    author="Eugene Scherba",
    license="MIT",
    author_email="escherba@livefyre.com",
    description=("Algorithms for locality-sensitive hashing on text data"),
    url='https://github.com/escherba/lsh-hdc',
    packages=find_packages(exclude=['tests', 'scripts']),
    long_description="LSH algo that uses MinHash signatures",
    install_requires=INSTALL_REQUIRES,
    tests_require=TESTS_REQUIRE,
    data_files=[("", ["LICENSE"])],
    test_suite='nose.collector',
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 2.7',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Text Processing :: Filters',
    ],
)
