import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="rs_metrics",
    version="0.4.0",
    author="Darel",
    author_email="darel142857@gmail.com",
    description="Metrics for recommender systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/darel13712/rs_metrics",
    packages=setuptools.find_packages(),
    install_requires=[
        'pandas',
        'numpy',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
