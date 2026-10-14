Quick Start
===========

This guide will walk you through a basic RiboNet analysis workflow.

Overview
--------

RiboNet analysis consists of three main steps:

1. **Preprocess**: Process ribosome profiling BAM files and extract all transcript inputs and ribosome occupancies
2. **Train**: Train a deep learning model to predict ribosome counts and distribution
3. **Interpret**: Compute importance scores and discover regulatory motifs

Test Your Installation
----------------------

The ``tests`` folder of the `GitHub repository <https://github.com/nedialkova-lab/RiboNet/tree/main/tests>`__ contains a small synthetic dataset to check that the full pipeline works, and to see what each input file looks like:

- 2 conditions (``WT`` and ``KD``) with 2 replicates each
- short (20-23 nt) and long (27-32 nt) ribosome footprints from the same BAM files
- 561 genes of *S. cerevisiae* chromosomes II and III (genome and annotation from SGD R64-2-1)

Ribosome footprints and RNA-seq counts are simulated with ``tests/generate_test_data.py``, mimicking typical yeast ribosome profiling data. In the ``KD`` condition, decoding of AAA, CAA and GAA codons is slower (as upon loss of the mcm5s2U tRNA modification), so the trained model should predict higher A-site occupancy of these codons (mainly AAA and CAA) in ``KD``, as shown in the codon occupancy plot of ``training/training.pdf``.

Run the test on a GPU node (~1 h on an NVIDIA A40):

.. code-block:: bash

   git clone https://github.com/nedialkova-lab/RiboNet
   bash RiboNet/tests/run_test.sh my_test_output 8   # output folder, number of threads

``run_test.sh`` runs the commands below on the test data, and ends with ``TEST PASSED`` if all reports were generated:

.. code-block:: bash

   cd RiboNet/tests
   ribonet preprocess --sample_file data/sample_file.txt --rna_counts data/rna_counts.txt \
       --genome_fasta data/genome.fa --gff data/annotation.gff3 --project_dir my_test_output \
       --threads 8 --maxutr5length 100 --maxutr3length 100 --maxlength 1200
   ribonet train --project_dir my_test_output --resolution codon
   ribonet interpret --project_dir my_test_output --n_batches 1
   ribonet report --project_dir my_test_output --control WT --threads 8

The ``--maxutr5length``, ``--maxutr3length`` and ``--maxlength`` options only keep short transcripts for training, so that the test runs fast.

Quick checks of the installation that run on CPU in ~1 min are also available with ``pytest RiboNet/tests`` (requires ``pip install pytest``).

Example Workflow
----------------

Step 1: Prepare Input Files
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a tab-separated sample file (``samples.txt``):

.. code-block:: text

   sample_id	condition	readlength	bam
   WT_rep1	WT	28-30	/path/to/WT_rep1.bam
   WT_rep2	WT	28-30	/path/to/WT_rep2.bam
   MUT_rep1	MUT	28-30	/path/to/MUT_rep1.bam
   MUT_rep2	MUT	28-30	/path/to/MUT_rep2.bam

Create an RNA counts file (``rna_counts.txt``):

.. code-block:: text

   gene_id	Length	WT MUT
   YAL001C	3483  150	160
   YAL002W	3825  89	88

You'll also need:

- Genome FASTA file (``genome.fa``)
- Genome annotation GFF/GTF file (``annotation.gff``)

Step 2: Preprocess Data
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   ribonet preprocess \
     --sample_file samples.txt \
     --rna_counts rna_counts.txt \
     --genome_fasta genome.fa \
     --gff annotation.gff \
     --project_dir my_ribonet_project \
     --threads 8

Step 3: Train Model
~~~~~~~~~~~~~~~~~~~

**For codon-resolution models** (recommended):

.. code-block:: bash

   ribonet train \
     --project_dir my_ribonet_project

**For nucleotide-resolution models**:

.. code-block:: bash

   ribonet train \
     --project_dir my_ribonet_project \
     --resolution nucleotide

**Key parameters:**

- ``--resolution``: ``codon`` or ``nucleotide``
- ``--lm``, ``--lm5``, ``--lm3``: Pre-trained language model(s)
- ``--species_proxy5``, ``--species_proxy3``: Species proxy if using SpeciesLM
- ``--epoch``: Number of training epochs
- ``--batch_size``: Batch size

Step 4: Interpret Results
~~~~~~~~~~~~~~~~~~~~~~~~~~

Initialize interpretation and distribute scoring jobs (with ``--n_batches 1``, all genes are scored directly and the ``score_batch`` step is not needed):

.. code-block:: bash

   ribonet interpret \
     --project_dir my_ribonet_project \
     --n_batches 10 \
     --scoring ISM

Score individual batches (can be parallelized):

.. code-block:: bash

   # Score batch 0
   ribonet score_batch \
     --project_dir my_ribonet_project \
     --batch 0

   # Score batch 1
   ribonet score_batch \
     --project_dir my_ribonet_project \
     --batch 1
   
   # ... repeat for all batches

Generate final report with motif discovery:

.. code-block:: bash

   ribonet report \
     --project_dir my_ribonet_project \
     --control WT \
     --threads 8

**Key parameters:**

- ``--n_batches``: Number of batches for distributed scoring
- ``--scoring``: Scoring method (``ISM`` for in silico mutagenesis)
- ``--save_distribution``: Saves importance score distributions for all genes (can be large)
- ``--control``: Control condition name for differential analysis
- ``--threads``: Number of threads for motif discovery

Expected Runtime
----------------

Approximate runtimes on NVIDIA A40 GPU:

- **Preprocessing**: 30 minutes - 2 hours (depends on number of samples and transcripts)
- **Training**: 20 minutes - 2 hours (depends on the resolution, model architecture, and number of epochs)
- **Interpretation**: 4-12 hours (depends on number of batches and scoring method)

Output Files
------------

After completing the workflow, your project directory will contain:

.. code-block:: text

   my_ribonet_project/
   ├── config_ribonet.json            # Model configuration
   ├── riboA/                         # RiboA outputs per sample
   │   ├── counts.rpf.txt             # Counts per gene for each bam file
   │   └── QC_report.pdf              # Preprocessing report
   ├── preprocessing/                 # Preprocessed data
   │   ├── x_inputs.parquet           # Input features
   │   └── y_outputs.parquet          # Output labels
   ├── training/                      # Trained model
   │   ├── ribonet.pt                 # Model weights
   │   ├── model_prediction.parquet   # Test set predictions
   │   ├── model_trainingset.parquet  # Training set predictions
   │   └── training.pdf               # Training report
   └── interpretation/                # Interpretation results
       ├── batches.tsv                # Gene batches for scoring
       ├── full_contrasts/            # Scoring results
       ├── counts/                    # Discovered features in counts
       ├── distribution/              # Discovered features in distribution
       └── interpretation.pdf         # Interpretation report with motifs and feature importance

See :doc:`outputs` for detailed descriptions of all output files.
