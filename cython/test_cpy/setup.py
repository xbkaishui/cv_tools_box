from setuptools import setup
from Cython.Build import cythonize

setup(
    name='foo2',
    version='1.0',
    ext_modules = cythonize("helloworld.pyx")
)