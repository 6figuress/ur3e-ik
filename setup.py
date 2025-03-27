import os

from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup

# Force using GCC
os.environ["CC"] = "gcc"
os.environ["CXX"] = "g++"

setup(
    name="ur_ikfast",
    version="0.1.0",
    license="MIT License",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    ext_modules=cythonize(
        [
			Extension(
                "ur3e_pen_final_ikfast",
                ["ur3e_pen_final/ur3e_pen_ikfast.pyx", "ur3e_pen_final/ikfast_wrapper.cpp"],
                language="c++",
                libraries=["lapack"],
            ),
			Extension(
                "ur3e_pen_final_2_ikfast",
                ["ur3e_pen_final_2/ur3e_pen_ikfast.pyx", "ur3e_pen_final_2/ikfast_wrapper.cpp"],
                language="c++",
                libraries=["lapack"],
            ),
        ]
    ),
)
