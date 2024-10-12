from setuptools import setup, find_packages

setup(
    # Metadata
    name="modular_genai",
    version="0.1.0-alpha",
    author="Abdullah Vanlioglu",
    author_email="a.vanli2019@gmail.com",
    url="",
    description="A modular GenAI library",
    long_description=("Clean and modular components for GenAI research written in"
                      "Pytorch, and JAX."),
    license="MIT",

    # Package Info
    packages=["modular_genai"],
    install_requires=[
    ],
    zip_safe=False
)