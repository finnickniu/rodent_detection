from distutils.core import setup
from Cython.Build import cythonize

setup(name='detect',
     ext_modules=cythonize('detect.py'))