Preprocessing
=============

The preprocessing step processes ribosome profiling BAM files and prepares data for model training.

Overview
--------

The ``ribonet preprocess`` command performs the following tasks:

1. Estimates A-site offsets using RiboA
2. Extracts ribosome density profiles from BAM files
3. Normalizes counts across samples
4. Filters transcripts based on expression and quality criteria
5. Prepares input features and output labels for training

Basic Usage
-----------

.. code-block:: bash

   ribonet preprocess \
     --sample_file samples.txt \
     --rna_counts rna_counts.txt \
     --genome_fasta genome.fa \
     --gff annotation.gff \
     --project_dir my_project \
     --threads 8

Required Arguments
------------------

``--sample_file``
~~~~~~~~~~~~~~~~~

Path to tab-separated file describing ribosome profiling samples. See :doc:`inputs` for format details.

``--rna_counts``
~~~~~~~~~~~~~~~~

Path to tab-separated file with RNA expression counts. See :doc:`inputs` for format details.

``--genome_fasta``
~~~~~~~~~~~~~~~~~~

Path to genome FASTA file.

``--gff``
~~~~~~~~~

Path to genome annotation file (GFF or GTF format).

``--project_dir``
~~~~~~~~~~~~~~~~~

Output directory for all analysis results. Will be created if it doesn't exist.

Optional Arguments
------------------

``--maxlength``
~~~~~~~~~~~~~~~

**Default:** ``4000``

Maximum transcript length (in nucleotides). Transcripts longer than this will be excluded.

.. code-block:: bash

   ribonet preprocess --maxlength 5000 ...

``--mindensity``
~~~~~~~~~~~~~~~~

**Default:** ``0.1``

Minimum mean ribosome density (reads per codon). Transcripts with lower density will be excluded.

.. code-block:: bash

   ribonet preprocess --mindensity 0.2 ...

``--maxutr5length``
~~~~~~~~~~~~~~~~~~~

**Default:** ``500`` (nucleotides)

Maximum 5' UTR length to include. Longer 5' UTRs will be truncated.

.. code-block:: bash

   ribonet preprocess --maxutr5length 300 ...

``--maxutr3length``
~~~~~~~~~~~~~~~~~~~

**Default:** ``500`` (nucleotides)

Maximum 3' UTR length to include. Longer 3' UTRs will be truncated.

.. code-block:: bash

   ribonet preprocess --maxutr3length 300 ...

``--utr5annot``
~~~~~~~~~~~~~~~

**Default:** None

Path to 5' UTR annotation file in FASTA format (SGD style). See :doc:`inputs` for format details. If not provided, RiboNet will extract this information from the GTF/GFF file if available. Otherwise, it will take fixed sequences upstream of the start codon, specified as ``maxutr5length``.

``--utr3annot``
~~~~~~~~~~~~~~~

**Default:** None

Path to 3' UTR annotation file in FASTA format (SGD style). See :doc:`inputs` for format details. If not provided, RiboNet will extract this information from the GTF/GFF file if available. Otherwise, it will take fixed sequences downstream of the stop codon, specified as ``maxutr3length``.

``--spikein``
~~~~~~~~~~~~~

**Default:** None

Comma-separated list of spike-in gene names or path to text file with one gene name per line, to use for normalization. The name of spike-ins should be included in the input bam files and RNA counts file.

.. code-block:: bash

   ribonet preprocess --spikein Fluc,Rluc ...

``--spikein_absolute_ribosomes``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Default:** None

If not provided, spike-ins are used to calibrate the relative abundance of ribosome profiling counts (i.e. keep ribosome counts / spike-in counts constant). If the absolute number of ribosomes per transcript of the spike-ins is known (e.g. `RiboCalibration <https://www.nature.com/articles/s41467-024-51258-0>`__), this parameter requires the total number of ribosomes per transcript in spike-in genes to perform absolute normalization, e.g. Fluc trisomes, 3, and Rluc disomes, 2, sum up to 5.

.. code-block:: bash

   ribonet preprocess --spikein Fluc,Rluc --spikein_absolute_ribosomes 5 ...

``--threads``
~~~~~~~~~~~~~

**Default:** ``1``

Number of CPU threads for parallel processing. Make sure to have enough memory when increasing this.

.. code-block:: bash

   ribonet preprocess --threads 16 ...

A-site Offset Estimation
------------------------

RiboNet uses `RiboA <https://github.com/obrien-lab/aip_web_docker>`__ to estimate A-site offsets for each read length and sample.

How it works
~~~~~~~~~~~~

1. Determines optimal codon A-site offset for each read length and 5' end frame (typically 15-18 nt from 5' end). It uses a Integer Programming method that maximizes the number A-site counts between the second and the stop codon of transcripts.
2. RiboA version used here was modified to calibrate for nucleotide resolution. For each of the read lengths:

   a. It positions the most abundant frame at the first position of the codon (i.e. 0 offset).
   b. Based on the offset for the most abundant frame, it sets the offset of the two other frames to the closest offset within the selected codon in step 1 (0, 1, or 2).

3. Selected offsets are then applied to all transcripts to extract P-site centered ribosome density profiles.

An offset is only estimated for a read length and frame if at least 10 genes have more than 0.5 reads per codon of that length and frame. Otherwise, the offset is reported as ``NA`` in ``riboA/<sample_id>/A-site_offsets.tab`` and those reads are not used. The estimated offsets are also printed in the log.

Manual offset specification
~~~~~~~~~~~~~~~~~~~~~~~~~~~

If ``--fixed_5offsets`` is provided, RiboA will just apply the specified A site offsets without performing estimation. One offset must be given for every read length in the sample file (``readlength-offset`` pairs separated by commas).

.. code-block:: bash

   ribonet preprocess --fixed_5offsets 27-15,28-15,29-16,30-16 ...

Output Files
------------

See :doc:`outputs` for detailed descriptions of all preprocessing output files:

- ``riboA/QC_report.pdf``: PDF report with QC plots and summary statistics
- ``riboA/asite_occupancy.tsv``: Per-sample A-site codon occupancy table (samples × codons)
- ``riboA/<sample_id>/``: RiboA outputs for each sample, including estimated A-site offsets and P-site profiles
- ``preprocessing/x_inputs.parquet``: Input features
- ``preprocessing/y_outputs.parquet``: Ribosome distribution profiles

.. note::
   In case the bam files listed in the sample file do not have index files (.bai or .csi), RiboNet will automatically create them in the same directory as the bam files.

Troubleshooting
---------------

Few transcripts passing filters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem:** Very few transcripts remain after filtering.

**Solutions:**

- Lower ``--mindensity`` threshold
- Increase ``--maxlength``
- Check that read lengths in sample file match actual data

A-site offset estimation fails
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem:** RiboA cannot estimate offsets.

**Solutions:**

- Check the offsets estimations in the RiboA output files to see if they are reasonable (e.g. 12-15 nt from 5' end).
- Check the QC plots in the ``QC_report.pdf`` report to see if there is strong 3-nt periodicity and good coverage of coding regions.
- RiboA estimation may fail if there are too few transcripts with sufficient coverage. In that case, try determining offsets manually (``--fixed_5offsets``).
- Check that BAM files are properly mapped
- Verify that GFF/GTF file contains correct CDS annotations
- Ensure read lengths specified in sample file are present in BAM
- Check library quality (should have strong 3-nt periodicity)

Multiprocessing errors
~~~~~~~~~~~~~~~~~~~~~~

**Symptoms:** RiboNet stops with "A worker process terminated abruptly, most likely because it ran out of memory", or the job is killed by the scheduler for exceeding its memory.

Some steps in RiboNet preprocessing and interpretation run in parallel worker processes (``--threads``), and each worker needs its own memory (e.g. RiboA uses several GB per large BAM file). If a worker is killed, RiboNet stops with the error above instead of waiting. In that case:

- Reduce the number of threads: ``--threads``
- Ensure your system has sufficient RAM memory
- Monitor system resources during execution to identify bottlenecks: ``htop``
