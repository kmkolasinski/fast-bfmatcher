import numpy
from setuptools import setup, Extension
from fast_bfmatcher.version import __version__


USE_CYTHON = True
try:
    # https://cython.readthedocs.io/en/latest/src/userguide/source_files_and_compilation.html
    from Cython.Build import cythonize
except ModuleNotFoundError:
    USE_CYTHON = False


EXT = ".pyx" if USE_CYTHON else ".c"
PACKAGE_NAME = "fast_bfmatcher"

extensions = [
    Extension(
        f"{PACKAGE_NAME}.matching_ops",
        [f"{PACKAGE_NAME}/fast_ops.c", f"{PACKAGE_NAME}/matching_ops" + EXT],
        libraries=["blas"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=[
            "-Ofast",
            "-march=native",
            "-msse3",
            "-finline-functions",
            "-fopt-info-vec-optimized",
        ],
    )
]

if USE_CYTHON:
    from Cython.Build import cythonize
    extensions = cythonize(extensions)


setup(
    name=PACKAGE_NAME,
    version=__version__,
    description="Faster implementation of opencv bruteforce cross check matcher",
    url="https://github.com/kmkolasinski/fast-bfmatcher",
    author="Krzysztof Kolasinski",
    author_email="kmkolasinski@gmail.com",
    license="MIT",
    packages=["fast_bfmatcher"],
    ext_modules=extensions,
)
