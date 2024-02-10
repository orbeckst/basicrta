<<<<<<< HEAD
"""Bayesian single-cutoff residence time analysis"""

# Add imports here
from .functions import * 


# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
=======
"""
basicrta
A package to extract binding kinetics from molecular dynamics simulations
"""

# Add imports here
from importlib.metadata import version

__version__ = version("basicrta")
>>>>>>> basicrta-kit/master
