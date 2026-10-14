Frequently Asked Questions
==========================

Do I need a GPU?
~~~~~~~~~~~~~~~~

**Yes, absolutely.** RiboNet requires an NVIDIA GPU with CUDA support. Training and inference on CPU would take excessively long (days to weeks instead of minutes or hours). Only preprocessing and parts of the report run on CPU. See :doc:`installation` for GPU requirements.

Do I need matched RNA-seq data?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Yes, RNA expression data is required for normalization of ribosome counts to the baseline RNA abundance. One RNA-seq sample per condition is sufficient.

What if I don't have UTR annotations?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can skip specifying UTRs, this will include fixed-length sequences upstream and downstream of the CDS (defined by ``--maxutr5length`` and ``--maxutr3length``, default 500 nt).

How long does preprocessing take?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Typical times on a 16-core CPU:

- **Small dataset** (yeast): 30-60 minutes
- **Large dataset** (human): 2-4 hours

Use ``--threads`` to parallelize. Multiprocessing will be used for RiboA and generating inputs files.

Which language model should I use?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here are some recommendations:

**For codon-resolution models:**

- Use RiNALMo or SpeciesLM (if your organism is in the fungi database).
- For SpeciesLM proxy, use the closest related species as proxy if exact match unavailable. E.g. for *S. cerevisiae* use *Kazachstania africana* or *Candida glabrata*.

**For nucleotide-resolution models:**

- Use RiNALMo. SpeciesLM can be also used, but it was trained to be anchored at the start codon instead of full transcripts.

In case of doubt, you can use the standalone RiboNet, which achieves top performance and is the fastest option. However, it can overfit on small datasets and low-coverage regions, which can lead to poorer UTR interpretation.

Should I use codon or nucleotide resolution?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Codon resolution** is the default and recommended for most studies of mRNA translation regulation. It is faster and resource efficient for limited GPU memory.

**Nucleotide resolution** should be used if you want to capture regulatory features at single-nucleotide level, such as frameshifting or RNA structures.

How long does training take?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Approximate times on NVIDIA A40:

- **3000 genes, codon resolution**: 20-30 minutes
- **3000 genes, nucleotide resolution**: 2-3 hours

If using LLM embeddings, training time will increase slightly.

What's a good model performance?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

RiboNet performance should always be evaluated in comparison to the variability of biological replicates. In general:

- **Spearman correlation of distribution profiles**: should correlate on par or slightly better than replicate-level correlations. This performance will also depend on the codon vs nucleotide resolution, since the second is inherently more noisy.
- **Pearson correlation of total counts**: should correlate better than the baseline correlation against RNA abundance, usually achieving R > 0.8.
- **Explained variance across transcripts and conditions(%)**: it typically predicts >50% of the total variance.
- **Normalized codon occupancy**: qualitatively, predictions of slow vs fast codons should match with true occupancies.

Which scoring should I use for interpretation?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**In silico mutagenesis (ISM)** - Recommended:

- More accurate and interpretable
- Captures non-linear effects
- Slow and computationally expensive, because it requires a separate forward pass for each mutation of each position of each transcript.

**Other** - Not yet implemented.

How many gene batches should I use for interpretation?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Depends on your computational resources. If using a **HPC cluster** with several GPUs, using more batches will allow you to parallelize the scoring across multiple nodes, reducing total runtime.

More batches = better parallelization but more job management overhead.

How long does interpretation take?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**ISM** (*in silico* mutagenesis): For a batch of 500 genes, 5-10 hours on A40 (``ribonet score_batch``).

After scoring, 1-3 hours for generating motifs and report (``ribonet report``). Time will depend on the number of conditions, and whether ``--save_distribution`` is enabled.

Where is my data stored?
~~~~~~~~~~~~~~~~~~~~~~~~~

All outputs are in the project directory:

.. code-block:: text

   <project_dir>/
   ├── riboA/              # RiboA outputs
   ├── preprocessing/      # Training data
   ├── training/           # Training outputs, model, and report
   └── interpretation/     # Attribution scores, motifs, and report

See :doc:`outputs` for complete structure.

How much disk space do I need?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Approximate requirements:

- **Preprocessing**: 200 MB per sample (it will depend on the number of genes after filtering)
- **Model weights**: 4 MB
- **Interpretation**: 1-5 GB if the default ``--save_distribution`` false is used. Otherwise, it depends on the resolution:
   - Codon resolution: ~50 GB per batch of 500 genes
   - Nucleotide resolution: ~300 GB per batch of 500 genes

Getting Help
------------

1. **Check documentation**: Read :doc:`index`, :doc:`quickstart`, specific sections
2. **GitHub Issues**: https://github.com/nedialkova-lab/RiboNet/issues
3. **Email**: See README for contact information

What information should I include when reporting issues?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Please provide:

1. RiboNet version: ``ribonet --version``
2. Python version: ``python --version``
3. PyTorch and CUDA versions: ``python -c "import torch; print(torch.__version__, torch.version.cuda)"``
4. Full error message or log file
5. Command that failed
6. Other information to reproduce the error

How do I cite RiboNet?
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   RiboNet manuscript in preparation.
   
   For now, please cite:
   - RiboNet GitHub: https://github.com/nedialkova-lab/RiboNet
