# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import os
import re
import sys
sys.path.insert(0, os.path.abspath('..'))

# Single source of truth for the version
with open(os.path.join(os.path.dirname(__file__), '..', 'ribonet', 'version.py')) as fh:
    _version = re.search(r'__version__\s*=\s*"([^"]+)"', fh.read()).group(1)

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'RiboNet'
copyright = '2026, Nedialkova Lab, Gagneur Lab'
author = 'Xavier Hernandez-Alias, Danny Nedialkova, Julien Gagneur'
release = _version
version = _version

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx_rtd_theme',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = []
html_theme_options = {
    'logo_only': False,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'style_nav_header_background': '#2980B9',
    # Toc options
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}

# -- Extension configuration -------------------------------------------------

# Napoleon settings for Google/NumPy style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}
autodoc_typehints = 'description'
# Heavy dependencies are mocked so that the API reference builds without installing them
autodoc_mock_imports = [
    'torch', 'transformers', 'multimolecule', 'wandb', 'pysam', 'gffutils', 'h5py',
    'hdf5plugin', 'numba', 'sklearn', 'scipy', 'seaborn', 'logomaker', 'leidenalg',
    'igraph', 'Bio', 'tqdm',
]
autodoc_typehints_description_target = 'documented'

# Autosummary settings
autosummary_generate = True

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
    'biopython': ('https://biopython.org/docs/latest/', None),
}

# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    'papersize': 'letterpaper',
    'pointsize': '11pt',
    'preamble': '',
    'figure_align': 'htbp',
    'fncychap': '',  # disable fncychap to avoid missing package dependency
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass [howto, manual, or own class]).
latex_documents = [
    ('index', 'RiboNet.tex', 'RiboNet Documentation',
     'Nedialkova Lab, Gagneur Lab', 'manual'),
]

# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    ('index', 'ribonet', 'RiboNet Documentation',
     [author], 1)
]

# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    ('index', 'RiboNet', 'RiboNet Documentation',
     author, 'RiboNet', 'Deep learning analysis toolkit for ribosome profiling.',
     'Miscellaneous'),
]