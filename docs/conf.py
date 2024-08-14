# Configuration file for the Sphinx documentation builder.

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = 'LLM-MRI'
copyright = '2024, Luiz Costa, Mateus Figenio, André Santanchè, Luiz Gomes-Jr'
author = 'Luiz Costa, Mateus Figenio, André Santanchè, Luiz Gomes-Jr'
release = '01.0'

extensions = [
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.autodoc",
    'sphinx_rtd_theme',
]

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

