.. RiboNet documentation master file

RiboNet Documentation
=====================

.. image:: img/ribonet.png
   :width: 500
   :align: center

:Author: Xavier Hernandez-Alias

:Version: |release|

**Deep learning analysis toolkit for ribosome profiling**

RiboNet is an automated analysis pipeline for the quantification, prediction, and interpretation of ribosome occupancy from ribosome profiling data.

.. warning::
   **GPU Required:** RiboNet requires an NVIDIA GPU with CUDA support for training and inference. Running on CPU will lead to excessively long computation times.

Key Features
------------

- Estimate A-site offsets using `RiboA <https://github.com/obrien-lab/aip_web_docker>`__
- Extract ribosome density profiles at codon or nucleotide resolution
- Train deep neural network models to predict ribosome counts and distribution
- Leverage pre-trained language models (`RiNALMo <https://github.com/lbcb-sci/RiNALMo>`__, `SpeciesLM <https://github.com/gagneurlab/SpeciesLM>`__)
- Perform in silico mutagenesis to identify cis-regulatory sequences
- Discover regulatory motifs using `TF-MoDISco <https://github.com/kundajelab/tfmodisco>`__
- Generate comprehensive reports with visualizations

Citation
--------

If you use RiboNet in your research, please cite:

   Hernandez-Alias X, Kaya S, Doychev K, Schroeder F, Hermann B, Madran B, Majidian S, Gagneur J, Nedialkova DD (2026). Metabolic transitions reprogram translation through sequence-dependent regulation. *bioRxiv*. doi: `10.64898/2026.11.15.766763 <https://doi.org/10.64898/2026.11.15.766763>`__

A BibTeX entry is available in the :doc:`FAQ <faq>`.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart
   inputs
   outputs

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   preprocessing
   training
   interpretation

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api
   faq
