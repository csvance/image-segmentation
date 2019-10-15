from distutils.core import setup
from Cython.Build import cythonize

setup(name="metri", ext_modules=cythonize('metrics.pyx'),)