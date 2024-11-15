# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

def skip(app, what, name, obj, would_skip, options):
    if name == "__init__":
        return False
    return would_skip

def setup(app):
    app.connect("autodoc-skip-member", skip)

project = 'AdaptiveResonanceLib'
copyright = '2024, Niklas Melton'
author = 'Niklas Melton'
release = '0.1.2'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'autoapi.extension',
    'sphinx.ext.napoleon',
    'myst_parser',
    'sphinx.ext.intersphinx',
    'sphinxcontrib.bibtex',
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

templates_path = ['_templates']
exclude_patterns = ['artlib/experimental/*', '../../artlib/experimental/*']

autoapi_type = 'python'
autoapi_dirs = ['../../artlib']  # Adjust this to point to your source code directory
autoapi_ignore = ['*/experimental', '*/experimental/*']
# autoapi_python_class_content = 'both'
# autoclass_content = 'both'

bibtex_bibfiles = ['artlib_citation.bib', '../../references.bib']

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'sklearn': ('https://scikit-learn.org/stable/', None)
}

suppress_warnings = ['ref.duplicate', 'duplicate.object', 'myst.duplicate_def', 'ref.python']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['../_static']





