from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

setup(name='adapu-publication-rng-repair', version='0.1.0',
      ext_modules=cythonize([Extension('decisionstump', ['src/decisionstump.pyx'],
                                      include_dirs=[np.get_include()])], language_level=3))
