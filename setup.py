from setuptools import Extension, dist, setup

from fast_bfmatcher.version import __version__

dist.Distribution().fetch_build_eggs(["cython", "numpy"])


def get_include_dir():

    import numpy

    return [numpy.get_include()]


PACKAGE_NAME = "fast_bfmatcher"

extensions = [
    Extension(
        f"{PACKAGE_NAME}.matching_ops",
        [f"{PACKAGE_NAME}/fast_ops.c", f"{PACKAGE_NAME}/matching_ops.pyx"],
        libraries=["blas"],
        include_dirs=get_include_dir(),
        extra_compile_args=[
            "-Ofast",
            "-march=native",
            "-msse3",
            "-finline-functions",
            "-fopt-info-vec-optimized",
        ],
    )
]


def get_extension_modules():

    from Cython.Build import cythonize

    return cythonize(extensions)


setup(
    name=PACKAGE_NAME,
    version=__version__,
    description="Faster implementation of opencv bruteforce cross check matcher",
    url="https://github.com/kmkolasinski/fast-bfmatcher",
    author="Krzysztof Kolasinski",
    author_email="kmkolasinski@gmail.com",
    license="MIT",
    packages=[PACKAGE_NAME],
    include_package_data=True,
    zip_safe=False,
    setup_requires=["cython", "numpy"],
    install_requires=["numpy>=1.19.5"],
    package_data={PACKAGE_NAME: ["matching_ops.pyx", "fast_ops.c", "fast_ops.h"]},
    ext_modules=get_extension_modules(),
)
