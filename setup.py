from setuptools import setup, find_packages
import glob
import re

def requires():
    """ gets packages from requirements.txt """
    with open('requirements.txt') as infile:
        return infile.read().splitlines()

## Auto-update ipyrad version from git repo tag
# Fetch version from git tags, and write to version.py.
# Also, when git is not available (PyPi package), use stored version.py.
INITFILE = "PTA/__init__.py"
CUR_VERSION = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                    open(INITFILE, "r").read(),
                    re.M).group(1)

setup(
    name="PTA",
    version=CUR_VERSION,
    url="https://github.com/isaacovercast/PTA",
    author="Isaac Overcast",
    author_email="isaac.overcast@gmail.com",
    description="Phylogeographic Temporal Analysis",
    long_description=open('README.md').read(),
    packages=find_packages(),    
    install_requires=requires(),
    entry_points={
            'console_scripts': [
                'PTA = PTA.__main__:main',
            ],
    },
    license='GPL',
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
    ],
)
