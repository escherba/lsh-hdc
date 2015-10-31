import numpy as np


def _parse_version(version_string):
    version = []
    for x in version_string.split('.'):
        try:
            version.append(int(x))
        except ValueError:
            # x may be of the form dev-1ea1592
            version.append(x)
    return tuple(version)


np_version = _parse_version(np.__version__)


if np_version < (1, 6, 2):
    # Allow bincount to accept empty arrays
    # https://github.com/numpy/numpy/commit/40f0844846a9d7665616b142407a3d74cb65a040
    def bincount(x, weights=None, minlength=None):
        if len(x) > 0:
            return np.bincount(x, weights, minlength)
        else:
            if minlength is None:
                minlength = 0
            minlength = np.asscalar(np.asarray(minlength, dtype=np.intp))
            return np.zeros(minlength, dtype=np.intp)

else:
    from numpy import bincount
