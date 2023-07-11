from setuptools import setup
from Cython.Build import cythonize
#from Cython.Compiler.Main import default_options
#import numpy as np
#
#default_options['compile_time_env'] = {'times': np.array}

setup(
    ext_modules = cythonize("basicrta/cygibbs.pyx",
    compiler_directives={'language_level' : "3"}) 
)

