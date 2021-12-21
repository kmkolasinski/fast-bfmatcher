from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
from setuptools.command.build_ext import build_ext
import numpy

from fast_bfmatcher.version import __version__

PACKAGE_NAME = "fast_bfmatcher"

setup(
    name=PACKAGE_NAME,
    version=__version__,
    description="...",
    url="...",
    author="Krzysztof Kolasinski",
    author_email="kmkolasinski@gmail.com",
    license="MIT License",
    packages=find_packages(exclude=["tests"]),
    package_dir={PACKAGE_NAME: PACKAGE_NAME},
    cmdclass={"build_ext": build_ext},
    install_requires=["cython"],
    setup_requires=["cython"],
    include_dirs=[numpy.get_include(), f"./{PACKAGE_NAME}"],
    ext_modules=cythonize(
        [
            Extension(
                f"{PACKAGE_NAME}.matching_ops",
                [f"{PACKAGE_NAME}/fast_ops.c", f"{PACKAGE_NAME}/matching_ops.pyx"],
                libraries=["blas"],
                extra_compile_args=[
                    "-Ofast",
                    "-march=native",
                    "-msse3",
                    "-finline-functions",
                    # "-fopt-info-vec-optimized",
                ],
            )
        ],
        # annotate=True,
    ),
)
