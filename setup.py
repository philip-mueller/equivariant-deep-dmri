from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name='equideepdmri',
    version='0.1.0',
    author='Philip Mueller',
    author_email='philip.jan.mueller@gmail.com',
    description='Rotationally and translationally equivariant layers and networks for deep learning on diffusion MRI scans',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/philip-mueller/equivariant-deep-dmri',
    license='MIT',
    packages=find_packages(include=['equideepdmri.*']),
    install_requires=['e3nn==0.0.0', 'numpy', 'torch-sparse==0.6.5'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
