from setuptools import Extension, dist, setup

from fast_bfmatcher.version import __version__

dist.Distribution().fetch_build_eggs(["Cython", "numpy"])

try:
    import numpy

    # https://cython.readthedocs.io/en/latest/src/userguide/source_files_and_compilation.html
    from Cython.Build import cythonize

    USE_CYTHON = True
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
    include_package_data=True,
    zip_safe=False,
    install_requires=["numpy>=1.19.5"],
    ext_modules=extensions,
)
