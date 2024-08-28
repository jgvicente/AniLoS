from setuptools import setup
from Cython.Build import cythonize
import numpy
from distutils.extension import Extension

hyp_extensions = Extension(
    name = "anilos.hybess",
    sources= ["./src/anilos/hybess_src/hybess.c"],
    include_dirs=[numpy.get_include()]
)
    
setup(
    ext_modules = [hyp_extensions]
)

# hyp_extensions = Extension(
#     name = "anilos.hybess",
#     sources=["./src/anilos/hybess_src/hybess.pyx"],
#     include_dirs=[numpy.get_include()]
# )
# setup(
#     ext_modules=cythonize([hyp_extensions], compiler_directives={'boundscheck': False,
#     								  'wraparound': False,
#     								  'cdivision': True,
#     								  'language_level': '3' })
# )
