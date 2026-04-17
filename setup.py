import os
import subprocess
import setuptools
import torch

from torch.utils.cpp_extension import BuildExtension
from setuptools import Extension


if __name__ == '__main__':
    cxx_flags = ['-O3', '-Wno-deprecated-declarations', '-Wno-unused-variable', '-Wno-sign-compare', '-Wno-reorder', '-Wno-attributes']
    sources = ['csrc/deep_ep.cpp', 'csrc/sycl/intranode.cpp', 'csrc/sycl/layout.cpp']
    include_dirs = ['csrc/']

    include_dirs.extend(torch.utils.cpp_extension.include_paths())
    # Summary
    print('Build summary:')
    print(f' > Sources: {sources}')
    print(f' > Includes: {include_dirs}')

    ext_modules = []

    sycl_compiler = os.getenv('SYCL_CXX', 'icpx')

    # Check if SYCL compiler exists
    try:
        subprocess.run([sycl_compiler, '--version'], check=True, capture_output=True)
        
        sycl_compile_args = ['-fsycl', '-O3', '-DUSE_XPU', '-fsycl-default-sub-group-size=32']
        sycl_link_args = ['-fsycl', '-lze_loader', '-lmpi']
        
        # Add Intel GPU specific optimization flags
        # Read target device from XPU_AOT_TARGETS env var, default to 'bmg'
        aot_target = os.getenv('XPU_AOT_TARGETS', 'bmg')
        sycl_compile_args.extend(['-fsycl-targets=spir64_gen', '-Xs', f'-device {aot_target}'])
        sycl_link_args.extend(['-fsycl-targets=spir64_gen', '-Xs', f'-device {aot_target}'])
        
        # Add common compile flags (warning suppression only, -O3 already in sycl_compile_args)
        xpu_cxx_flags = [flag for flag in cxx_flags if flag.startswith('-Wno-')]
        sycl_compile_args.extend(xpu_cxx_flags)
        sycl_compile_args.extend(['-DDISABLE_NVSHMEM', '-std=c++17'])
        
        torch_lib_path = os.path.join(os.path.dirname(torch.__file__), 'lib')
        
        sycl_extension = Extension(
            name='deep_ep_cpp',
            sources=sources,
            include_dirs=include_dirs,
            library_dirs=[torch_lib_path],
            extra_compile_args=sycl_compile_args,
            extra_link_args=sycl_link_args,
            libraries=['torch_cpu', 'torch', 'torch_python', 'c10'],
            language='c++'
        )
        
        # Override compiler for this extension
        os.environ['CXX'] = sycl_compiler
        ext_modules.append(sycl_extension)
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise SystemExit(
            f"Error: SYCL compiler '{sycl_compiler}' not found. "
            "Please install a SYCL compiler (e.g., icpx from Intel oneAPI toolkit) "
            "or set SYCL_CXX to point to a valid SYCL compiler."
        )
    # noinspection PyBroadException
    try:
        cmd = ['git', 'rev-parse', '--short', 'HEAD']
        revision = '+' + subprocess.check_output(cmd).decode('ascii').rstrip()
    except Exception:
        revision = ''

    setuptools.setup(name='deep_ep_xpu',
                     version='0.0.1' + revision,
                     packages=setuptools.find_packages(include=['deep_ep_xpu']),
                     ext_modules=ext_modules,
                     cmdclass={'build_ext': BuildExtension})
