# RiboNet Documentation

This directory contains the source files for RiboNet's documentation, built with [Sphinx](https://www.sphinx-doc.org/) and hosted on [Read the Docs](https://readthedocs.org/).

## Building Documentation Locally

### Prerequisites

Install documentation dependencies:

```bash
pip install -r requirements.txt
```

### Build HTML Documentation

```bash
cd docs
make html
```

The built documentation will be in `_build/html/`. Open `_build/html/index.html` in your browser.

### Clean Build Files

```bash
cd docs
make clean
```

## Documentation Structure

- `index.rst` - Main documentation page with table of contents
- `installation.rst` - Installation instructions
- `quickstart.rst` - Quick start guide
- `preprocessing.rst` - Preprocessing documentation
- `training.rst` - Training documentation
- `interpretation.rst` - Interpretation documentation
- `inputs.rst` - Input file format specifications
- `outputs.rst` - Output file descriptions
- `faq.rst` - Frequently asked questions
- `conf.py` - Sphinx configuration
- `img/` - Images used in documentation

## Contributing to Documentation

1. Edit `.rst` files using reStructuredText syntax
2. Build locally to preview changes: `make html`
3. Check for warnings and errors
4. Commit and push changes
5. Documentation will be automatically built and deployed on Read the Docs

## reStructuredText Syntax

### Headings

```rst
Chapter Title
=============

Section Title
-------------

Subsection Title
~~~~~~~~~~~~~~~~

Subsubsection Title
^^^^^^^^^^^^^^^^^^^
```

### Code Blocks

```rst
.. code-block:: bash

   ribonet preprocess --help
```

### Links

```rst
External link: `RiboNet <https://github.com/nedialkova-lab/ribonet>`_
Internal link: :doc:`installation`
```

### Admonitions

```rst
.. note::
   This is a note.

.. warning::
   This is a warning.

.. tip::
   This is a tip.
```

## Read the Docs Configuration

Documentation is automatically built on Read the Docs when changes are pushed to the repository. Configuration is in `.readthedocs.yml` at the repository root.

## Questions?

For documentation-related questions, please open an issue on GitHub: https://github.com/nedialkova-lab/ribonet/issues