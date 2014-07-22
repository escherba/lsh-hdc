from setuptools import setup, find_packages
from pkg_resources import resource_string


tests_require = [
    'nose>=1.0',
    'coverage',
    'nosexcover',
    'mock>=1.0'
]

setup(
    name="lsh-hdc",
    version="0.0.1",
    author="Eugene Scherba",
    author_email="escherba@livefyre.com",
    url='https://github.com/escherba/lsh-hdc',
    keywords="LSH-based high-dimensional clustering",
    packages=find_packages(exclude=['tests', 'scripts']),
    license='LICENSE',
    setup_requires=tests_require,
    extras_require={
        'plot': [
            'matplotlib>=1.3.1'
        ],
        'dev': [
            'ipython>=2.1.0'
        ] + tests_require
    },
    test_suite='nose.collector',
    tests_require=tests_require,
    description="High-dimensional clustering using locality-sensitive hashing",
    long_description=resource_string(__name__, 'README.md')
)
