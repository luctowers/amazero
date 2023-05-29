#!/usr/bin/env python3

from distutils.core import setup, Extension
import numpy as np
boardops = Extension('amazero.boardops', sources = ['amazero/boardops.c'], include_dirs=[np.get_include()])
setup(name = 'amazero', version = '0.1', packages=['amazero'], ext_modules = [boardops])
