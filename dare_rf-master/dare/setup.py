import os

import numpy
from numpy.distutils.misc_util import Configuration
from numpy.distutils.core import setup
from Cython.Build import cythonize


def configuration(parent_package='', top_path=None):
    config = Configuration('dare', parent_name=parent_package, top_path=top_path)

    libraries = []
    if os.name == 'posix':
        libraries.append('m')

    config.add_extension("_config",
                         sources=["_config.pyx"],
                         include_dirs=[numpy.get_include()],
                         libraries=libraries,
                         extra_compile_args=["-O3"])
    config.add_extension("_manager",
                         sources=["_manager.pyx"],
                         include_dirs=[numpy.get_include()],
                         libraries=libraries,
                         extra_compile_args=["-O3"])
    config.add_extension("_tree",
                         sources=["_tree.pyx"],
                         include_dirs=[numpy.get_include()],
                         libraries=libraries,
                         extra_compile_args=["-O3"])
    config.add_extension("_splitter",
                         sources=["_splitter.pyx"],
                         include_dirs=[numpy.get_include()],
                         libraries=libraries,
                         extra_compile_args=["-O3"])
    config.add_extension("_remover",
                         sources=["_remover.pyx"],
                         include_dirs=[numpy.get_include()],
                         libraries=libraries,
                         extra_compile_args=["-O3"])
    config.add_extension("_simulator",
                         sources=["_simulator.pyx"],
                         include_dirs=[numpy.get_include()],
                         libraries=libraries,
                         extra_compile_args=["-O3"])
    config.add_extension("_utils",
                         sources=["_utils.pyx"],
                         include_dirs=[numpy.get_include()],
                         libraries=libraries,
                         extra_compile_args=["-O3"])
    config.add_extension("_argsort",
                         sources=["_argsort.pyx"],
                         include_dirs=[numpy.get_include()],
                         libraries=libraries,
                         extra_compile_args=["-O3"])

    config.ext_modules = cythonize(
        config.ext_modules,
        compiler_directives={'language_level': 3},
        annotate=True
    )

    return config


if __name__ == "__main__":
    setup(**configuration(top_path='').todict())
