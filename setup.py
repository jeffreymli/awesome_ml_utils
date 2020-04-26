import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="awesome-ml-utils", # Replace with your own username
    version="0.0.1",
    author="Jeff Li",
    author_email="dj.jeffmli@gmail.com",
    description="A package of useful utility functions when building machine learning models!",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://https://github.com/jeffreymli/awesome_ml_utils",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)