# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'AdaptiveResonanceLib'
copyright = '2024, Niklas Melton'
author = 'Niklas Melton'
release = '0.1.2'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc', 'autoapi.extension', 'sphinx.ext.napoleon', 'myst_parser']

templates_path = ['_templates']
exclude_patterns = ['artlib/experimental/*']

autoapi_type = 'python'
autoapi_dirs = ['../../artlib']  # Adjust this to point to your source code directory

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']


