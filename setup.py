from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(name='Time-Warp',
      ext_modules=cythonize("timewarp/timewarp.pyx"),
      include_dirs=[numpy.get_include()])
