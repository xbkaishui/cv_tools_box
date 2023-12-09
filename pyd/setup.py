from distutils.core import setup
from Cython.Build import cythonize

setup(
    name='addpyd',
    ext_modules=cythonize("pydtest/pydt.py"),
)
