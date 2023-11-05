import setuptools 

with open("README.md", "r") as fh:
    long_description = fh.read()
    
setuptools.setup(
    name="tinygrad",
    version="1.0.0",
    author="Vivek Chauhan",
    author_email="vivekchauhan0395@gmail.com",
    description="A Tinygrad is a lightweight deep learning framework for educational purposes.",
    packages= setuptools.find_packages(),
    url="https://github.com/VivekChauhan05/Tinygrad",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache License 2.0",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)