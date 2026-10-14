Interpretation
==============

The interpretation step performs *in silico* mutagenesis and motif discovery to identify *cis*-regulatory sequences that control ribosome occupancy.

Overview
--------

RiboNet interpretation consists of three steps:

1. **Initialize interpretation**: Set up batches for distributed scoring
2. **Score batches**: Perform *in silico* mutagenesis (ISM) or gradient-based attribution
3. **Generate report**: Discover motifs using TF-MoDISco and create visualizations

Workflow
--------

Step 1: Initialize interpretation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   ribonet interpret \
     --project_dir my_project \
     --n_batches 10 \
     --scoring ISM

This configures scoring and splits genes into batches as specified in ``interpretation/batches.tsv``. With ``--n_batches 1``, all genes are scored directly and Step 2 can be skipped.

Step 2: Score batches
~~~~~~~~~~~~~~~~~~~~~

Score each batch independently (can be parallelized):

.. code-block:: bash

   # Score batch 0
   ribonet score_batch --project_dir my_project --batch 0
   
   # Score batch 1
   ribonet score_batch --project_dir my_project --batch 1
   
   # ... continue for all batches

On HPC clusters, use job arrays. In Step 1, an example job array command is created for SLURM (``ribonet_array.sbatch``). Adapt the command for your cluster's job scheduler and configuration (e.g. queues, GPU nodes, conda environment).

.. code-block:: bash

   # SLURM example
   sbatch ribonet_array.sbatch

.. warning::
   **GPU Required:** This step requires an NVIDIA GPU with CUDA support for scoring.

Step 3: Generate report
~~~~~~~~~~~~~~~~~~~~~~~

After all batches are scored:

.. code-block:: bash

   ribonet report \
     --project_dir my_project \
     --control WT \
     --threads 8

This runs TF-MoDISco motif discovery and generates the interpretation report.

.. warning::
   **GPU Required:** Unless ``--save_distribution`` is enabled during ``ribonet interpret``, this step requires an NVIDIA GPU with CUDA support for scoring. Otherwise, it can be run on CPU.

Initialize Interpretation: ``ribonet interpret``
------------------------------------------------

``--project_dir``
~~~~~~~~~~~~~~~~~

Path to project directory with trained model.

``--n_batches``
~~~~~~~~~~~~~~~

**Required**

Number of batches to split genes into for scoring parallelisation. With ``1``, scoring runs immediately; with more batches, each batch is scored with ``ribonet score_batch``.

.. code-block:: bash

   ribonet interpret --n_batches 20 ...

``--scoring``
~~~~~~~~~~~~~

**Default:** ``ISM``

**Choices:** ``ISM``, other options will be added in the future

Attribution method:

- ``ISM``: *In silico* mutagenesis (systematic mutations)

**In silico mutagenesis (ISM):**

For each position, mutate to all possible nucleotides and measure impact on predictions.

.. code-block:: text

   Original: ATCG...
   Mutant 1: TTCG... → Δ prediction = prediction_original - prediction_mutant
   Mutant 2: CTCG... → Δ prediction = prediction_original - prediction_mutant
   Mutant 3: GTCG... → Δ prediction = prediction_original - prediction_mutant

   dcounts = max(Δ prediction for each mutation)
   ddcounts = mean((Δ prediction in Condition N) - (Δ prediction in Control) for each mutation)

``--save_distribution``
~~~~~~~~~~~~~~~~~~~~~~~

If disabled (default), only the scores of sequences on counts are saved. Downstream analysis of the distribution will require partial re-scoring sequences on the distribution, which is slower and requires GPU.

If enabled, both the scores on counts and distribution are saved, so that downstream analysis of the distribution is faster and can run on CPU-only. However, this will significantly increase the disk space required for storing the scores.

.. code-block:: bash

   ribonet interpret --save_distribution ...

``--batch_size``
~~~~~~~~~~~~~~~~

**Default:** ``None``

The inference batch size used within RiboNet. If not specified, it will be set to four times the training batch size. Reduce if you encounter GPU memory issues during scoring, or increase if you have sufficient GPU memory for faster scoring.

Score Batches: ``ribonet score_batch``
--------------------------------------

``--project_dir``
~~~~~~~~~~~~~~~~~

Path to project directory.

``--batch``
~~~~~~~~~~~

Batch number to score (0-indexed).

Must be < ``n_batches`` specified in ``interpret``.

.. code-block:: bash

   ribonet score_batch --project_dir my_project --batch 0

Generate Report: ``ribonet report``
-----------------------------------

``--project_dir``
~~~~~~~~~~~~~~~~~

Path to project directory.

``--control``
~~~~~~~~~~~~~

**Default:** None (it will compare to the average across conditions)

Name of control condition for differential analysis. Must match a condition name from the sample file.

.. code-block:: bash

   ribonet report --control WT ...

``--threads``
~~~~~~~~~~~~~

**Default:** ``1``

Number of CPU threads for multiprocessing.

.. code-block:: bash

   ribonet report --threads 16 ...

``--motif_db``
~~~~~~~~~~~~~~

**Default:** oRNAment and AtTract databases are used by default for annotation.

Path to motif database for annotation, in MEME format. Motif matching uses ``tomtom`` from the `MEME suite <https://meme-suite.org>`__, which is installed with the Bioconda package. If ``tomtom`` is not available (e.g. pip installation), motifs are discovered but not annotated; install it with ``conda install -c bioconda meme``.

.. code-block:: bash

   ribonet report --motif_db rbp_motifs.meme ...

``--diffsites``
~~~~~~~~~~~~~~~

**Default:** 5000

Number of top differentially occupied sites to use for distribution analysis. Around these sites, a window of size ``--pwind`` will be analyzed for motif enrichment.

``--pwind``
~~~~~~~~~~~

**Default:** 30 (nucleotides)

Window size around differentially occupied sites to analyze for motif enrichment.

``--max_seqlets_per_metacluster``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Default:** 1000

Maximum number of seqlets per metacluster passed to TF-MoDISco. The seqlets with the highest importance are kept. Increase to discover rarer motifs at the cost of runtime and memory.

``--filename``
~~~~~~~~~~~~~~

**Default:** ``interpretation.pdf``

Filename for the interpretation report PDF.

Motif Discovery
---------------

TF-MoDISco
~~~~~~~~~~

RiboNet uses TF-MoDISco (Transcription Factor Motif Discovery from Importance Scores) to discover regulatory motifs. Full description of TF-MoDISco is available in the `original repository <https://github.com/kundajelab/tfmodisco>`__.

1. **Extract seqlets**: High-importance subsequences from importance scores
2. **Cluster seqlets**: Group similar sequence patterns
3. **Generate motifs**: Create position weight matrices (PWMs) for each cluster

From the original implementation, the built-in code has been adapted to:
- Disable the reverse complement step, since ribosome occupancy is strand-specific.
- Sort seqlets by importance score to prioritize the most important seqlets (``--max_seqlets_per_metacluster``, default 1000).
- Custom default motif database uses oRNAment and AtTract RBP motif databases motif matching.
- Use colorblind-safe palette for logos and visualizations.

Counts
~~~~~~

Importance scores on counts is computed in automatically-detected sets of conditions. Two types of scores are analyzed:

- **dcounts**: Importance scores on counts for each condition
- **ddcounts**: Differential importance scores on counts between each condition and the control

Motif discovery is performed separately for each region (5' UTR, CDS, 3' UTR), since their importance scores can have very different scales.

Distribution
~~~~~~~~~~~~

To detect regulatory patterns of ribosomal elongation:
1. Identify top differentially occupied sites between conditions.
2. Extract importance scores in a window around these sites.
3. Plot aggregated importance scores to visualize patterns of ribosome stalling or acceleration.
4. Run TF-MoDISco on these scores to discover motifs associated with differential ribosome occupancy.

If more specific sets of conditions or experimental designs are required, the ``RiboInterpretation`` class contains methods for personalized analysis of the scores (``generate_report``, ``extract_counts_motifs``, ``extract_distribution_motifs``). Refer to the help and docstrings for more details.

.. code-block:: python

   from ribonet.interpretation import RiboInterpretation

   # Load the interpretation object
   interpretation = RiboInterpretation(project_dir)
   # Define control condition for each condition to analyse within the set
   # The set of cond0, cond1, and cond2 will be analyzed together, and the importance scores of cond1 and cond2 will be compared to cond0
   ctrl_cond = {"cond1": "cond0", "cond2": "cond0"}
   interpretation.generate_report(ctrl_cond, threads=8, filename="custom.pdf")

Output Files
------------

See :doc:`outputs` for detailed descriptions of interpretation output files:

- ``full_contrasts/batch_*/``: Importance scores per batch
- ``counts/*_cond1_modisco.npz``: Attribution scores for condition 1 per region
- ``counts/cond1_cond2/dcounts/*_modisco_results.h5``: TF-MoDISco results on dcounts per region
- ``counts/cond1_cond2/dcounts/*_modisco_report/``: TF-MoDISco report on dcounts per region
- ``counts/cond1_cond2/ddcounts/*_modisco_results.h5``: TF-MoDISco results on ddcounts per region
- ``counts/cond1_cond2/ddcounts/*_modisco_report/``: TF-MoDISco report on ddcounts per region
- ``distribution/cond1_cond2_vs_control/diffsites.csv``: Top differentially occupied sites
- ``distribution/cond1_cond2_vs_control/modisco_results.h5``: TF-MoDISco results on distribution around differentially occupied sites
- ``distribution/cond1_cond2_vs_control/modisco_report/``: TF-MoDISco report on distribution around differentially occupied sites
- ``interpretation.pdf``: Comprehensive interpretation report

Interactively working with results
----------------------------------

Inspect discovered motifs
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from ribonet.interpretation import RiboInterpretation

   # Load the interpretation object
   interpretation = RiboInterpretation(project_dir)
   # Load motifs for a specific contrast and region
   motif_df = interpretation.get_genes_pattern(pattern="pattern_0", typepat="pos_patterns", \
                                               arr="dcounts", conds="cond1_cond2", region="5UTR")

Visualize counts scores of a specific gene
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from ribonet.interpretation import RiboInterpretation

   # Load the interpretation object
   interpretation = RiboInterpretation(project_dir)
   # Plot logos
   interpretation.plot_logo(gene="YAL001C", conds=["cond1", "cond2"], \
                            ctrl_cond="cond0", ntstart=0, ntend=100)


Visualize counts and distribution scores of a specific gene and condition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from ribonet.interpretation import RiboInterpretation

   # Load the interpretation object
   interpretation = RiboInterpretation(project_dir)
   # Plot matrix heatmap
   interpretation.plot_matrix(gene="YAL001C", cond="cond1", \
                              ctrl_cond="cond0", ntstart=0, ntend=100)

.. warning::
   **GPU Required:** Unless ``--save_distribution`` is enabled during ``ribonet interpret``, this step requires an NVIDIA GPU with CUDA support for scoring. Otherwise, it can be run on CPU.

Troubleshooting
---------------

Multiprocessing errors
~~~~~~~~~~~~~~~~~~~~~~

**Symptoms:** RiboNet stops with "A worker process terminated abruptly, most likely because it ran out of memory", or the job is killed by the scheduler for exceeding its memory.

Some steps in RiboNet preprocessing and interpretation run in parallel worker processes (``--threads``), and each worker needs its own memory (e.g. RiboA uses several GB per large BAM file). If a worker is killed, RiboNet stops with the error above instead of waiting. In that case:

- Reduce the number of threads: ``--threads``
- Ensure your system has sufficient RAM memory
- Monitor system resources during execution to identify bottlenecks: ``htop``

Few motifs discovered
~~~~~~~~~~~~~~~~~~~~~

**Solutions:**

- Check model performance (low metrics may indicate poor predictions)
- Check importance scores manually for a few genes

GPU out of memory
~~~~~~~~~~~~~~~~~

**Solutions:**

- Reduce batch size. Check ``inference_batch_size`` in the config file for current batch size
- Reduce model complexity by using a codon-resolution model, applying UTR pooling, or decreasing hidden dimensions
- Use a GPU with more memory
