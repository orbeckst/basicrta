import os

on_rtd = os.environ.get("READTHEDOCS") == "True"
if on_rtd:
    html_theme = "default"
else:
    html_theme = "nature"

extensions = [
   'sphinx.ext.duration',
   'sphinx.ext.doctest',
   'sphinx.ext.autodoc',
   'sphinx.ext.autosummary',
]
