import setuptools
import numpy as np

numpy_include_path = np.get_include()

setuptools.setup(
    name="SoDiferent",
    packages=setuptools.find_packages(),
    include_dirs=[numpy_include_path], 
    define_macros=[
        ("NDEBUG",1),
    ],
    ext_modules=[
        setuptools.Extension("SoDiferent.NumericRk", sources=["src/NumericRk.cxx"],language="c++",extra_compile_args=["-std=c++17"],extra_link_args=["-lstdc++"]),
    ]
)
