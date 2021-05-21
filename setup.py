from setuptools import setup, find_packages

setup(
    name="retinal",
    version="0.1",
    author="Bingyuan Liu",
    description="Application in multiple retinal tasks",
    packages=find_packages(),
    python_requries=">=3.8",
    install_requires=[
        # Please install pytorch-related libraries and opencv by yourself based on your environment
    ],
)
