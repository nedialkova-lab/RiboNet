Output files
============

RiboNet generates outputs in three stages: preprocessing, training, and interpretation. All outputs are organized in the project directory specified with ``--project_dir``.

.. note::
   In case the bam files listed in the sample file do not have index files (.bai or .csi), RiboNet will automatically create them in the same directory as the bam files.

Project directory structure
---------------------------

.. code-block:: text

   my_project/
   ├── config_ribonet.json
   ├── ribonet.log
   ├── riboA/
   │   ├── QC_report.pdf
   │   ├── asite_occupancy.tsv
   │   ├── qc_plots/
   │   ├── sample1/
   │   ├── ...
   │   ├── sampleN/
   │   │   ├── aip.log
   │   │   ├── A-site_offsets.tab
   │   │   ├── Perc_of_genes_for_all_offsets.tab
   │   │   ├── P-site_profiles_codon.tab
   │   │   ├── P-site_profiles_nucleotide.tab
   │   │   └── P-site_profiles_nucleotide_extend2001nt.tab
   │   └── counts.rpf.txt
   ├── preprocessing/
   │   ├── x_inputs.parquet
   │   ├── y_outputs.parquet
   │   └── sample_file.txt
   ├── training/
   │   ├── ribonet.pt
   │   ├── model_prediction.parquet
   │   ├── model_trainingset.parquet
   │   └── training.pdf
   └── interpretation/
       ├── batches.tsv
       ├── full_contrasts/
       │   ├── contrasts_batch.parquet      (with --n_batches 1)
       │   ├── contrasts_batch_0.parquet    (with --n_batches > 1)
       │   ├── contrasts_batch_1.parquet
       │   └── ...
       ├── counts/
       │   ├── 5UTR_<cond>_<readlengths>_modisco.npz
       │   ├── CDS_<cond>_<readlengths>_modisco.npz
       │   ├── 3UTR_<cond>_<readlengths>_modisco.npz
       │   ├── start_stop_metagene.pdf
       │   └── <cond1>-<cond2>-…/
       │       ├── dcounts/
       │       │   ├── 5UTR_modisco_results.h5
       │       │   ├── 5UTR_modisco_report/
       │       │   ├── 5UTR_seqlet_positions.pdf
       │       │   ├── CDS_modisco_results.h5
       │       │   ├── CDS_modisco_report/
       │       │   ├── CDS_seqlet_positions.pdf
       │       │   ├── 3UTR_modisco_results.h5
       │       │   ├── 3UTR_modisco_report/
       │       │   └── 3UTR_seqlet_positions.pdf
       │       └── ddcounts/
       │           └── (same as dcounts/)
       ├── distribution/
       │   └── <cond1>-<cond2>-…_vs_<ctrl>/
       │       ├── diffsites.csv
       │       ├── sites_scores_pwind<pwind>.npz
       │       ├── diffsites_histogram.pdf
       │       ├── diffsites_average_logo.pdf
       │       ├── diffsites_average_logo.xlsx
       │       ├── modisco_results.h5
       │       └── modisco_report/
       └── interpretation.pdf

RiboA Outputs
-------------

Located in ``<project_dir>/riboA/``

It contains the outputs of RiboA for each sample:

- ``A-site_offsets.tab``: Estimated A-site offsets for each read length and frame
- ``Perc_of_genes_for_all_offsets.tab``: Percentage of genes maximizing the A-site counts at each offset
- ``P-site_profiles_codon.tab``: P-site profiles at codon resolution for each gene
- ``P-site_profiles_nucleotide.tab``: P-site profiles at nucleotide resolution for each gene
- ``P-site_profiles_nucleotide_extend2001nt.tab``: Same as above but with 2001 nt extension on both sides of the CDS

QC_report.pdf
~~~~~~~~~~~~~

**Format:** PDF

**Description:** QC data report with:

- Read length distribution per sample
- 5' end metagene profiles per read length
- A-site codon occupancy heatmap (log2 occupancy scores per sample and codon)
- Individual plots are also saved in the ``qc_plots/`` subdirectory for easier access.

asite_occupancy.tsv
~~~~~~~~~~~~~~~~~~~

**Format:** Tab-separated values (TSV)

**Description:** Per-sample A-site codon occupancy table. For each sample the
mean ribosome occupancy score is computed as the per-position count divided by
the per-gene/sample mean, averaged over all CDS positions that carry a given
A-site codon. The first and last 15 nucleotides of each CDS are excluded.
Stop codons are omitted. Values represent raw (non-log) occupancy scores;
a value of 1 corresponds to average occupancy.

**Rows:** individual samples (``sample_id`` from the sample file)

**Columns:** 61 amino-acid-encoding codons (sorted alphabetically)

The log2-transformed heatmap of this table is also saved in
``riboA/qc_plots/codon_occupancy.pdf`` and embedded in ``QC_report.pdf``.

Preprocessing Outputs
---------------------

Located in ``<project_dir>/preprocessing/``

x_inputs.parquet
~~~~~~~~~~~~~~~~

**Format:** Parquet (Apache Arrow)

**Description:** Input features for model training

**Columns:**

- Gene identifier (index)
- ``utr5``: 5' UTR sequence (padded with ``X`` to ``--maxutr5length``), followed by the start codon
- ``cds``: CDS sequence
- ``utr3``: stop codon followed by the 3' UTR sequence (padded with ``X`` to ``--maxutr3length``)
- ``utr5_len``: Length of 5' UTR, capped at ``--maxutr5length``
- ``cds_len``: Length of CDS
- ``utr3_len``: Length of 3' UTR, capped at ``--maxutr3length``
- ``longest_utr5_annot``: Length of longest annotated 5' UTR
- ``longest_utr3_annot``: Length of longest annotated 3' UTR
- ``rna``: RNA expression levels per condition (TPMs, corrected by spike-ins if provided)

**Example:**

.. code-block:: python

   import pandas as pd
   x_inputs = pd.read_parquet("preprocessing/x_inputs.parquet")
   print(x_inputs.head())

y_outputs.parquet
~~~~~~~~~~~~~~~~~

**Format:** Parquet (Apache Arrow)

**Description:** Ribosome density profiles for model training

**Columns:**

- Gene identifier (index)
- Multi-level columns indexed by ``(region, condition)`` where:
  - Condition: e.g. ``WT``, ``mutant1``, etc.
  - Regions: ``utr5``, ``cds``, ``utr3``

Each cell contains a list of arrays of ribosome counts at each nucleotide position along the transcript. Each item in the list corresponds to one biological replicate.

**Example:**

.. code-block:: python

   import pandas as pd
   y_outputs = pd.read_parquet("preprocessing/y_outputs.parquet")
   # Access CDS ribosome counts for condition "WT"
   cds_counts = y_outputs[("cds", "WT_27-30")].iloc[0]

sample_file.txt
~~~~~~~~~~~~~~~

**Format:** Tab-separated

**Description:** Sample metadata file provided and written during preprocessing. Used by
subsequent training and interpretation steps.

**Columns:**

- ``sample_id``: Unique sample identifier
- ``condition``: Experimental condition name
- ``readlength``: Read length used for A-site offset determination
- ``bam``: Path to the BAM file
- ``track``: Combined ``condition_readlength`` label used internally

Training Outputs
----------------

Located in ``<project_dir>/training/``

ribonet.pt
~~~~~~~~~~

**Format:** PyTorch model state dict

**Description:** Best-performing RiboNet model weights saved during training
(selected by lowest validation loss).

config_ribonet.json
~~~~~~~~~~~~~~~~~~~

**Format:** JSON

**Description:** Saved in the project directory (not in ``training/``). Configuration file containing all parameters used for training and interpretation. Useful for reproducibility and record-keeping.

model_prediction.parquet
~~~~~~~~~~~~~~~~~~~~~~~~

**Format:** Parquet (Apache Arrow)

**Description:** Model predictions on the held-out test set. One row per gene;
multi-level columns indexed by ``(condition, info)`` where ``info`` is either
``profile_pred`` (predicted ribosome density profile as an array) or
``count_pred`` (predicted total ribosome count as a scalar).

**Example:**

.. code-block:: python

   import pandas as pd
   preds = pd.read_parquet("training/model_prediction.parquet")
   # Predicted ribosome count for condition "WT"
   count_pred = preds[("WT", "count_pred")]

model_trainingset.parquet
~~~~~~~~~~~~~~~~~~~~~~~~~

**Format:** Parquet (Apache Arrow)

**Description:** Same structure as ``model_prediction.parquet`` but for
the training set. Useful for assessing overfitting and generating
training-set visualisations.

training.pdf
~~~~~~~~~~~~

**Format:** PDF

**Description:** Training evaluation report with:

- Percentage of explained variance for counts prediction across genes and conditions.
- Count prediction correlation plots (predicted vs. observed per condition).
- Distribution correlation plots (predicted vs. observed ribosome density profiles in coding sequences).
- Normalized codon occupancy plots for predicted and observed distributions in coding sequences.

Interpretation Outputs
----------------------

Located in ``<project_dir>/interpretation/``

batches.tsv
~~~~~~~~~~~

**Format:** Tab-separated

**Description:** Batch assignment file created when genes are split into
multiple batches for parallel scoring. Maps each gene to a ``batch_id``.

full_contrasts/
~~~~~~~~~~~~~~~

**Directory containing per-batch Parquet files of raw importance scores:**

.. code-block:: text

   full_contrasts/
   ├── contrasts_batch_0.parquet
   ├── contrasts_batch_1.parquet
   └── ...

With ``--n_batches 1``, a single ``contrasts_batch.parquet`` file is written.

Each file contains per-gene, per-condition, per-position contrast scores for the full transcript. Files can be filtered by gene when loading:

.. code-block:: python

   import pandas as pd
   batch = pd.read_parquet(
       "interpretation/full_contrasts/contrasts_batch_0.parquet",
       filters=[("gene", "==", "YAL001C")]
   )

counts/
~~~~~~~

**Per-condition NPZ files** (``{region}_{condition}_{readlengths}_modisco.npz``):

Intermediate files used as input to TF-MoDISco. Arrays:

- ``ohe``: One-hot encoded sequences (n_genes, length, 4)
- ``dcounts``: Absolute count attribution scores (n_genes, length, 4)
- ``ddcounts``: Differential count attribution scores (n_genes, length, 4)

**dcounts/ and ddcounts/ subdirectories:**

TF-MoDISco results for absolute (``dcounts``) and differential
(``ddcounts``) count attributions, grouped by the set of conditions
analysed together.

- ``{region}_modisco_results.h5``: TF-MoDISco discovered motifs (HDF5 format)
- ``{region}_modisco_report/``: HTML motif report with sequence logos
  (``motifs.html``, ``trimmed_logos/``)

distribution/
~~~~~~~~~~~~~

**diffsites.csv:**

List of genomic positions (gene, position) identified as differentially
translated between the compared conditions and the control.

**sites_scores_pwind<pwind>.npz:**

Intermediate files used as input to TF-MoDISco and distribution logos. Arrays:

- ``ohe``: One-hot encoded sequences (n_diffsites, pwind, 4)
- ``scores``: Attribution scores per condition (n_diffsites, pwind, 4)

**diffsites_histogram.pdf:**

Histogram of the percentage of differential sites per position compared to the total of genes of at least that length.

**diffsites_average_logo.pdf:**

Average sequence logo around differential sites, computed by averaging the attribution scores across all differential sites. Source data is also saved in ``diffsites_average_logo.xlsx``.

**modisco_results.h5:**

TF-MoDISco motifs discovered from importance scores around differential
ribosome distribution sites.

**modisco_report/:**

HTML report with motif sequence logos (``motifs.html``, ``trimmed_logos/``),
seqlet position density along the transcript
(``seqlet_transcript_sites.pdf``), and seqlet window positions
(``seqlet_window_positions.pdf``).

interpretation.pdf
~~~~~~~~~~~~~~~~~~

**Format:** PDF

**Description:** Comprehensive interpretation report with:

- Discovered motif logos for counts and distribution analyses
- Seqlet position plots
- Differential site summaries
- Known motif matches (if a motif database was provided)
