Installation
============

System Requirements
-------------------

- Linux (or macOS for CPU-only steps)
- NVIDIA GPU with CUDA support, and a recent NVIDIA driver
- Python >= 3.9

.. warning::
   RiboNet is designed for GPU acceleration and is **not practical to run on CPU** due to excessively long training and inference times. Ensure your system has a compatible NVIDIA GPU before installation.

All dependencies, including PyTorch, are installed automatically. Pre-trained language models are downloaded automatically on first use.

Option 1: Bioconda (Recommended)
--------------------------------

The Bioconda package installs RiboNet together with all its dependencies, including PyTorch with GPU support and the MEME suite (used to annotate discovered motifs):

.. code-block:: bash

   conda create -n ribonet -c conda-forge -c bioconda ribonet
   conda activate ribonet

Conda automatically selects the CUDA build of PyTorch when it detects an NVIDIA driver. If you install RiboNet on a machine without GPU (e.g. the login node of a cluster) and run it on GPU nodes, request the CUDA build explicitly:

.. code-block:: bash

   CONDA_OVERRIDE_CUDA=12.6 conda create -n ribonet -c conda-forge -c bioconda ribonet "pytorch=*=cuda*"

To install into an existing environment, use ``conda install -c conda-forge -c bioconda ribonet``.

Option 2: Pip
-------------

RiboNet is also available on PyPI. On Linux, the PyTorch wheels installed by pip already include CUDA support:

.. code-block:: bash

   # Create and activate an environment (conda, venv, ...)
   conda create -n ribonet python=3.11
   conda activate ribonet

   pip install ribonet

Motif annotation in ``ribonet report`` uses ``tomtom`` from the MEME suite, which cannot be installed with pip. Without it, motifs are still discovered but not matched to known RNA-binding protein motifs. To enable it, install MEME in the same environment:

.. code-block:: bash

   conda install -c conda-forge -c bioconda meme

.. note::
   The default PyTorch wheels require a recent NVIDIA driver. If ``torch.cuda.is_available()`` returns ``False`` on an older driver, install a PyTorch build for your CUDA version first, as described at https://pytorch.org/get-started/locally (e.g. ``pip install torch --index-url https://download.pytorch.org/whl/cu121``), and then ``pip install ribonet``.

Optional components
~~~~~~~~~~~~~~~~~~~

- RiNALMo language model embeddings (``--lm rinalmo``) require the ``multimolecule`` package: ``pip install "ribonet[rinalmo]"`` (or ``pip install multimolecule`` in a conda environment).

Option 3: Development Installation
----------------------------------

For development or to use the latest unreleased version:

.. code-block:: bash

   git clone https://github.com/nedialkova-lab/RiboNet
   cd RiboNet

   conda create -n ribonet -c conda-forge -c bioconda python=3.11 meme
   conda activate ribonet
   pip install -e ".[test]"

Verify the Installation
-----------------------

Check that RiboNet is installed and that PyTorch detects your GPU:

.. code-block:: bash

   ribonet --version
   python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

If ``CUDA available: False``, check your NVIDIA driver (``nvidia-smi``) and that you are on a GPU node.

Then run the test dataset, as described in :doc:`quickstart`.

Troubleshooting
---------------

CUDA not available
~~~~~~~~~~~~~~~~~~

If PyTorch cannot detect your GPU:

1. Check that a GPU is visible: ``nvidia-smi``.
2. Check that PyTorch was installed with CUDA support: ``python -c "import torch; print(torch.version.cuda)"`` should not print ``None``.
3. With conda, reinstall the CUDA build: ``conda install -c conda-forge "pytorch=*=cuda*"``.
4. With pip, reinstall PyTorch for your CUDA version (see note above).
