import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="QiskitExtension",
    version="0.1.4",
    author="HSIEH, LI-YU",
    author_email="cjh9027@smail.nchu.edu.tw",
    description="A Qiskit visualize extension",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    url="https://github.com/Slope86/QiskitExtension",
    packages=setuptools.find_packages(exclude=["experiments", "tests"]),
    package_data={"": ["*.ini"]},
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "qiskit[visualization] == 1.0.0",
        "qiskit-aer == 0.13.3"
    ],
    python_requires=">=3.10",
)
