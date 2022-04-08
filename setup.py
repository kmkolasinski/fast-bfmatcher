import subprocess
from distutils import log
from distutils.errors import DistutilsSetupError
from pathlib import Path
from typing import Union

from setuptools import Extension, dist, setup
from setuptools.command.build_ext import build_ext

dist.Distribution().fetch_build_eggs(["cython", "numpy"])

BLIS_PATH = Path("build/blis")
BLIS_VERSION = "0.9.0"
PACKAGE_NAME = "fast_bfmatcher"
VERSION = "1.5.0"


def run_command(command: str, sources_path: Union[str, Path] = "."):

    sources_path = str(sources_path)
    log.info(f'>> Executing command: "{command}" in directory: {sources_path}\n')

    command = [w.strip() for w in command.split(" ")]
    result = subprocess.run(
        command,
        cwd=sources_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    stdout, stderr = result.stdout.decode(), result.stderr.decode()

    if result.returncode != 0:
        status = stdout or stderr
        raise DistutilsSetupError(
            f"An error occurred while running the command: {command}.\nStatus: {status}"
        )


class compile_blis_and_build_ext(build_ext):
    def build_extension(self, ext: Extension):
        print(f">> Building extension: {ext.name}")

        if not BLIS_PATH.exists():
            Path("build").mkdir(exist_ok=True)
            # clone blis at selected version
            run_command(
                f"git clone --depth 1 --branch {BLIS_VERSION} https://github.com/flame/blis.git",
                "build",
            )
        else:
            log.info(f'>> BLIS_PATH="{BLIS_PATH}" exists, skipping download')

        blis_library_list = list(BLIS_PATH.glob("lib/*/libblis.a"))
        blis_include_list = list(BLIS_PATH.glob("include/*/blis.h"))

        if len(blis_library_list) == 0:
            log.info(">> Building BLIS: https://github.com/flame/blis#getting-started")
            run_command("./configure -t openmp auto", BLIS_PATH)
            run_command("make -j 4", BLIS_PATH)

            blis_library_list = list(BLIS_PATH.glob("lib/*/libblis.a"))
            blis_include_list = list(BLIS_PATH.glob("include/*/blis.h"))

        if len(blis_library_list) == 1:
            blis_library = str(blis_library_list[0])
            blis_include = str(Path(blis_include_list[0]).parent)
        else:
            raise DistutilsSetupError(
                f"Configuration error: expected single BLIS library, got: {blis_library_list}"
            )

        log.info(
            f">> BLIS library has been already built, using library from path: {blis_library}, "
            f"BLIS include path is: {blis_include}"
        )

        ext.include_dirs.append(blis_include)
        ext.extra_link_args.append(blis_library)
        super(compile_blis_and_build_ext, self).build_extension(ext)


def get_include_dir():
    import numpy

    return [numpy.get_include()]


extensions = [
    Extension(
        f"{PACKAGE_NAME}.matching_ops",
        [f"{PACKAGE_NAME}/fast_ops.c", f"{PACKAGE_NAME}/matching_ops.pyx"],
        include_dirs=get_include_dir(),
        extra_compile_args=[
            "-Ofast",
            "-march=native",
            "-msse3",
            "-finline-functions",
            "-Wno-unused-function",
            "-Wfatal-errors",
            "-fPIC",
            "-std=c99",
            "-fopenmp",
            "-D_POSIX_C_SOURCE=200112L",
            "-fopt-info-vec-optimized",
        ],
        extra_link_args=[
            "-lm",
            "-fopenmp",
            "-lrt",
        ],
    )
]


def get_extension_modules():
    from Cython.Build import cythonize

    return cythonize(extensions)


setup(
    name=PACKAGE_NAME,
    version=VERSION,
    description="Faster implementation of OpenCV BFMatcher matcher",
    url="https://github.com/kmkolasinski/fast-bfmatcher",
    author="Krzysztof Kolasinski",
    author_email="kmkolasinski@gmail.com",
    license="MIT",
    packages=["fast_bfmatcher", "fast_bfmatcher.extra"],
    include_package_data=True,
    zip_safe=False,
    setup_requires=["cython", "numpy"],
    install_requires=["numpy>=1.16.0"],
    package_data={PACKAGE_NAME: ["matching_ops.pyx", "fast_ops.c", "fast_ops.h"]},
    ext_modules=get_extension_modules(),
    cmdclass={"build_ext": compile_blis_and_build_ext},
)
