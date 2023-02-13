"""Example setup file
"""
from setuptools import setup, find_packages

setup(
    name='example_package',
    version='0.0.0.1',
    author='UCSD Engineers for Exploration',
    author_email='e4e@eng.ucsd.edu',
    entry_points={
        'console_scripts': [
            'ExamplePythonConsoleScript = example_package.example_module:exampleEntryPoint'
        ]
    },
    packages=find_packages(),
    install_requires=[]
)
