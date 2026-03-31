import os
import sys
import subprocess
from pathlib import Path
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def build_extension(self, ext):
        import torch
        
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        
        abi_flag = int(getattr(torch._C, "_GLIBCXX_USE_CXX11_ABI", 1))

        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}',
            f'-DPYTHON_EXECUTABLE={sys.executable}',
            f'-DCMAKE_PREFIX_PATH={torch.utils.cmake_prefix_path}',
            f'-D_GLIBCXX_USE_CXX11_ABI={abi_flag}',
        ]
        
        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]
        
        if sys.platform.startswith('linux'):
            try:
                num_cores = os.cpu_count() or 1
                build_args += ['-j', str(num_cores)]
            except:
                pass
        
        cmake_args += [f'-DCMAKE_BUILD_TYPE={cfg}']
        
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        
        subprocess.check_call(
            ['cmake', ext.sourcedir] + cmake_args,
            cwd=self.build_temp
        )
        
        subprocess.check_call(
            ['cmake', '--build', '.'] + build_args,
            cwd=self.build_temp
        )

setup(
    name='rxmesh-torch',
    version='0.1.0',
    author='will-zzy',
    description='PyTorch binding for RXMesh',
    long_description='',
    ext_modules=[CMakeExtension('rxmesh_torch_ops', sourcedir='..')],
    cmdclass={'build_ext': CMakeBuild},
    zip_safe=False,
    python_requires='>=3.7',
    install_requires=[
        'torch>=1.12.0',
        'numpy',
    ],
)
