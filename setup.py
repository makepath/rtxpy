import os
import re
import sys
import platform
import subprocess

from packaging.version import Version
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from shutil import copyfile

class CMakeExtension(Extension):
    def __init__(self, name, moduleName, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)
        self.moduleName = moduleName

class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: " +
                ", ".join(e.name for e in self.extensions))

        if platform.system() == "Windows":
            cmake_version = Version(re.search(r'version\s*([\d.]+)',
                                    out.decode()).group(1))
            if cmake_version < Version('3.10.0'):
                raise RuntimeError("CMake >= 3.10.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(
            os.path.dirname(self.get_ext_fullpath(ext.name)))
        extdir += '/' + ext.moduleName
        if platform.system() == "Windows":
            extdir = extdir.replace("/","\\")
        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir]
        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]
        conda_build = os.environ.get("CONDA_BUILD", 0)

        if platform.system() == "Windows":
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(
                cfg.upper(),
                extdir)]
            # CMake lets you override the generator, as is done in conda build.
            # If using NMake for generator or building on conda, do not use
            # arch specifier as not supported.
            cmake_generator = os.environ.get("CMAKE_GENERATOR", "")
            if sys.maxsize > 2**32 and not (
                    cmake_generator.startswith("NMake") or conda_build):
                cmake_args += ['-A', 'x64']
            if not conda_build:
                build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j2']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get('CXXFLAGS', ''),
            self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args,
                              cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args,
                              cwd=self.build_temp)

        if platform.system() == "Windows":
            src = self.build_temp + "\\" + cfg + "\\" + ext.moduleName + ".dll"
            dst = extdir + "\\" + ext.moduleName + ".dll"
            copyfile(src, dst)

setup(
    version='0.0.3',
    ext_modules=[CMakeExtension('crtx','rtxpy')],
    cmdclass=dict(build_ext=CMakeBuild),
)
