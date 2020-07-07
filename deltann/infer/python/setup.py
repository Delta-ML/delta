import os
import sys
import re
import platform
import subprocess

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion

#if "DELTA_BUILD_DIR" in os.environ:
#    DELTA_BUILD_DIR = os.environ['DELTA_BUILD_DIR']
#else:
#    raise RuntimeError("You must set the env variable DELTA_BUILD_DIR, which is build dir of the delta infer with python.")


def load_readme(path):
  return open(os.path.join(os.path.dirname(__file__), "README.md")).read()


class DeltaExportExtension(Extension):

  def __init__(self, name, sourcedir=''):
    Extension.__init__(self, name, sources=[])
    self.sourcedir = os.path.abspath(sourcedir)


class DeltaExtBuild(build_ext):

  def run(self):
    try:
      out = subprocess.check_output(['cmake', '--version'])
    except OSError:
      raise RuntimeError(
          "CMake must be installed to build the following extensions: " +
          ", ".join(e.name for e in self.extensions))

    if platform.system() == "Windows":
      cmake_version = LooseVersion(
          re.search(r'version\s*([\d.]+)', out.decode()).group(1))
      if cmake_version < '3.1.0':
        raise RuntimeError("CMake >= 3.1.0 is required on Windows")

    for ext in self.extensions:
      self.build_extension(ext)

  def build_extension(self, ext):
    extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
    cmake_args = [
        '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
        '-DPYTHON_EXECUTABLE=' + sys.executable,
        '-DBUILD_DELTA_INFER_PYTHON=ON', '-DCMAKE_INSTALL_RPATH=$ORIGIN'
    ]

    BuildWithDebug = 'ON' if self.debug else 'OFF'
    BuildMode = 'Debug' if self.debug else 'Release'
    build_args = ['--config', BuildMode]

    if platform.system() == "Windows":
      cmake_args += [
          '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(
              BuildMode.upper(), extdir)
      ]
      if sys.maxsize > 2**32:
        cmake_args += ['-A', 'x64']
      build_args += ['--', '/m']
    else:
      #cmake_args += ['-DBUILD_DEBUG=' + BuildWithDebug]
      cmake_args += ['-DBUILD_DEBUG=' + "ON"]
      build_args += ['--', '-j8']

    env = os.environ.copy()
    env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(
        env.get('CXXFLAGS', ''), self.distribution.get_version())
    if not os.path.exists(self.build_temp):
      os.makedirs(self.build_temp)
    subprocess.check_call(
        ['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
    subprocess.check_call(
        ['cmake', '--build', '.'] + build_args, cwd=self.build_temp)


setup(
    name='delta_infer',
    version='0.0.1',
    author='Speech-HPC',
    author_email='cuichaowen@didiglobal.com',
    description='Delta inference python api setup',
    long_description=load_readme("./README.txt"),
    packages=find_packages(),
    include_package_data=True,
    ext_modules=[DeltaExportExtension('export_py', '..')],
    cmdclass=dict(build_ext=DeltaExtBuild),
    entry_points={
        'console_scripts': ['visual = delta_infer.visual_pattern:command',]
    },
    install_requires=[
        "absl-py >= 0.8.0",
        "netron >= 3.5.9",
    ],
    zip_safe=False,
)
