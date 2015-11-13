def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('fent', parent_package, top_path)
    config.add_extension(
        name='_fent',
        sources=['_fent.pyf', '_fent.f90'],
    )
    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())