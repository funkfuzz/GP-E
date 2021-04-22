from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setup(
        name='gazepoint_detection',
        version='0.1',
        description='Gazepoint Detection algorithm',

        author='funkfuzz',
        long_description=long_description,
        long_description_content_type="text/markdown",

        packages=find_packages(exclude=[]),
        python_requires='>=3.6',
        install_requires=[
            'coloredlogs',
            'h5py',
            'numpy',
            'opencv-python',
            'pandas',
            'ujson',
            'dlib',
            # choose the most appropriate for you tensorflow version 

        ],
)


# list of all packages installed that worked for me on my machine

# _tflow_select             2.1.0                       gpu
# absl-py                   0.12.0           py36haa95532_0
# astor                     0.8.1            py36haa95532_0
# blas                      1.0                         mkl
# ca-certificates           2021.1.19            haa95532_1
# certifi                   2020.12.5        py36haa95532_0
# cffi                      1.14.5           py36hcd4344a_0
# colorama                  0.4.4              pyhd3eb1b0_0
# coloredlogs               15.0             py36haa95532_0
# coverage                  5.5              py36h2bbff1b_2
# cudatoolkit               10.0.130                      0
# cudnn                     7.6.5                cuda10.0_0
# cython                    0.29.22          py36hd77b12b_0
# dlib                      19.21.1                  pypi_0    pypi
# freetype                  2.10.4               hd328e21_0
# gast                      0.4.0                      py_0
# grpcio                    1.36.1           py36hc60d5dd_1
# h5py                      2.8.0            py36hf7173ca_2
# hdf5                      1.8.20               hac2f561_1
# humanfriendly             9.1              py36haa95532_0
# icc_rt                    2019.0.0             h0cc432a_1
# importlib-metadata        3.7.3            py36haa95532_1
# intel-openmp              2020.2                      254
# jpeg                      9b                   hb83a4c4_2
# keras-applications        1.0.8                      py_1
# keras-preprocessing       1.1.2              pyhd3eb1b0_0
# libopencv                 3.4.2                h20b85fd_0
# libpng                    1.6.37               h2a8f88b_0
# libprotobuf               3.14.0               h23ce68f_0
# libtiff                   4.2.0                hd0e1b90_0
# lz4-c                     1.9.3                h2bbff1b_0
# markdown                  3.3.4            py36haa95532_0
# mkl                       2020.2                      256
# mkl-service               2.3.0            py36h196d8e1_0
# mkl_fft                   1.3.0            py36h46781fe_0
# mkl_random                1.1.1            py36h47e9c7a_0
# ninja                     1.10.2           py36h6d14046_0
# numpy                     1.19.2           py36hadc3359_0
# numpy-base                1.19.2           py36ha3acd2a_0
# olefile                   0.46                     py36_0
# opencv                    3.4.2            py36h40b0b35_0
# opencv-python             4.5.1.48                 pypi_0    pypi
# openssl                   1.1.1k               h2bbff1b_0
# pandas                    1.1.3            py36ha925a31_0
# pillow                    8.1.2            py36h4fa10fc_0
# pip                       21.0.1           py36haa95532_0
# protobuf                  3.14.0           py36hd77b12b_1
# py-opencv                 3.4.2            py36hc319ecb_0
# pycparser                 2.20                       py_2
# pyreadline                2.1                      py36_1
# python                    3.6.13               h3758d61_0
# python-dateutil           2.8.1              pyhd3eb1b0_0
# pytorch                   1.2.0           py3.6_cuda100_cudnn7_1    pytorch
# pytz                      2021.1             pyhd3eb1b0_0
# scipy                     1.5.2            py36h9439919_0
# setuptools                52.0.0           py36haa95532_0
# six                       1.15.0           py36haa95532_0
# sqlite                    3.35.2               h2bbff1b_0
# tensorboard               1.14.0           py36he3c9ec2_0
# tensorflow                1.14.0          gpu_py36h305fd99_0
# tensorflow-base           1.14.0          gpu_py36h55fc52a_0
# tensorflow-estimator      1.14.0                     py_0
# tensorflow-gpu            1.14.0               h0d30ee6_0
# termcolor                 1.1.0            py36haa95532_1
# tk                        8.6.10               he774522_0
# torchvision               0.4.0                py36_cu100    pytorch
# typing_extensions         3.7.4.3            pyha847dfd_0
# ujson                     4.0.2                    pypi_0    pypi
# vc                        14.2                 h21ff451_1
# vs2015_runtime            14.27.29016          h5e58377_2
# werkzeug                  1.0.1              pyhd3eb1b0_0
# wheel                     0.36.2             pyhd3eb1b0_0
# wincertstore              0.2              py36h7fe50ca_0
# wrapt                     1.12.1           py36he774522_1
# xz                        5.2.5                h62dcd97_0
# zipp                      3.4.1              pyhd3eb1b0_0
# zlib                      1.2.11               h62dcd97_4
# zstd                      1.4.5                h04227a9_0
