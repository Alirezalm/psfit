from setuptools import setup, find_packages

setup(
    name="psfit",
    version="0.1.1",
    author="Alireza Olama",
    author_email="alireza.lm69@gmail.com",
    description="Parallel Sparse Fitting Toolbox for Distributed Sparse Model Training.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/alirezalm/psfit",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        "numpy",
        "torch",
        "ray",
        "scikit-learn"
    ],
)
