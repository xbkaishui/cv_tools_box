from distutils.core import setup
from distutils.extension import Extension
from setuptools import find_packages, setup

setup(name='foo',
      version='1.0',
      ext_modules=[Extension('foo', ['./src/foo.c'])],
      package_dir={"": "src"},
      packages=find_packages(where="src", include=["cyp", "cyp.*"]),
      )