from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='emd_ext',
    ext_modules=[
        CUDAExtension('emd_ext', [
            'emd.cpp',
            'emd_cuda.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })