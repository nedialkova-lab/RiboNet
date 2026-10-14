Input Files
===========

RiboNet requires several input files. This page describes the required format for each file type.

.. tip::
   The ``tests/data`` folder of the `GitHub repository <https://github.com/nedialkova-lab/RiboNet/tree/main/tests>`__ contains a complete, small example of every input file (see :doc:`quickstart`).

Sample File
-----------

**Required for:** ``ribonet preprocess``

A tab-separated file describing ribosome profiling samples.

Format
~~~~~~

.. code-block:: text

   sample_id	condition	readlength	bam
   WT_rep1	WT	27-30	/path/to/WT_rep1.bam
   WT_rep2	WT	27-30	/path/to/WT_rep2.bam
   MUT_rep1	MUT	27-30	/path/to/MUT_rep1.bam

**Columns:**

- ``sample_id``: Unique identifier for each sample (no spaces)
- ``condition``: Experimental condition name (e.g., WT, treated, knockout). The condition name should match the column names in the RNA counts file.
- ``readlength``: Dash-separated range of read lengths to include (e.g., ``27-30``)
- ``bam``: Full path to BAM file containing ribosome profiling reads

Requirements
~~~~~~~~~~~~

- Must be tab-separated
- Must include header row with exact column names
- Read lengths should match common ribosome footprint sizes (typically 20-23 nt for short footprints, 27-32 nt for long footprints). One same bam file can be listed more than once with different read length ranges if needed.
- Multiple replicates per condition are recommended

Example
~~~~~~~

.. code-block:: text

   sample_id	condition	readlength	bam
   Glucose_long1	Glucose	27-30	/data/riboseq/glucose_rep1.bam
   Glucose_long2	Glucose	27-30	/data/riboseq/glucose_rep2.bam
   Glucose_long3	Glucose	27-30	/data/riboseq/glucose_rep3.bam
   Galactose_long1	Galactose	27-30	/data/riboseq/galactose_rep1.bam
   Galactose_long2	Galactose	27-30	/data/riboseq/galactose_rep2.bam
   Galactose_long3	Galactose	27-30	/data/riboseq/galactose_rep3.bam
   Glucose_short1	Glucose	20-22	/data/riboseq/glucose_rep1.bam
   Glucose_short2	Glucose	20-22	/data/riboseq/glucose_rep2.bam
   Glucose_short3	Glucose	20-22	/data/riboseq/glucose_rep3.bam
   Galactose_short1	Galactose	20-22	/data/riboseq/galactose_rep1.bam
   Galactose_short2	Galactose	20-22	/data/riboseq/galactose_rep2.bam
   Galactose_short3	Galactose	20-22	/data/riboseq/galactose_rep3.bam

RNA Counts File
---------------

**Required for:** ``ribonet preprocess``

A tab-separated file containing RNA expression counts for normalization (e.g. from ``featureCounts`` or ``htseq-count``).

Format
~~~~~~

.. code-block:: text

   gene_id	Length	condition1	condition2	condition3
   YAL001C	3483  150	145	160
   YAL002W	3825  89	92	88

**Columns:**

- ``gene_id``: Gene identifier matching the GFF/GTF file
- ``Length``: Gene length (e.g. CDS or transcript length), used to compute TPMs
- At least one column per condition with raw counts. If duplicated conditions exist, the counts will be depth-normalized and averaged across samples of the same condition.
- Condition names should match those in the sample file

Requirements
~~~~~~~~~~~~

- Must be tab-separated
- Gene IDs must match those in the GFF/GTF annotation file
- Spike-ins should be included if used for normalization

Example
~~~~~~~

.. code-block:: text

   gene_id	Length   Glucose	Galactose
   YAL001C	3483  150	145
   YAL002W	3825  89	92
   YAL003W	621  45	43
   YAL005C	1929  205	198

Genome FASTA File
-----------------

**Required for:** ``ribonet preprocess``

A FASTA file containing the reference genome sequence.

Format
~~~~~~

Standard FASTA format with sequence headers matching chromosome/contig names in the GFF/GTF file.

.. code-block:: text

   >chr1
   ATCGATCGATCG...
   >chr2
   GCTAGCTAGCTA...

GFF/GTF Annotation File
-----------------------

**Required for:** ``ribonet preprocess``

A GFF/GTF file describing gene structure and features.

Format
~~~~~~

Standard GFF3 (``.gff``, ``.gff3``) or GTF (``.gtf``) format, with ``CDS`` features linked to their gene (directly or through a transcript feature):

.. code-block:: text

   chr1	source	gene	1000	5000	.	+	.	ID=gene1;Name=YAL001C
   chr1	source	mRNA	1000	5000	.	+	.	ID=transcript1;Parent=gene1
   chr1	source	exon	1000	5000	.	+	.	Parent=transcript1
   chr1	source	CDS	1101	4900	.	+	0	Parent=transcript1

Gene names are taken from the ``gene_name``, ``gene_id``, ``Name`` or ``ID`` attributes (in this order of priority) and must match the RNA counts file. If ``--utr5annot``/``--utr3annot`` are not given, UTRs are defined by the transcript (``mRNA``/``transcript``) boundaries around the CDS. If a gene has several coding isoforms, the first one in genomic order is used, so annotations with one canonical transcript per gene are recommended.

Optional Input Files
--------------------

UTR Annotation Files
~~~~~~~~~~~~~~~~~~~~

**Optional for:** ``ribonet preprocess``

5' and 3' UTR annotations for each transcript. If not provided, RiboNet will take fixed upstream and downstream sequences around the CDS, specified as ``maxutr5length`` and ``maxutr3length``.

**5' UTR annotation** (``--utr5annot``)

**3' UTR annotation** (``--utr3annot``)

Format should be a fasta file with UTR sequences (e.g. http://sgd-archive.yeastgenome.org/sequence/S288C_reference/SGD_all_ORFs_5prime_UTRs.fsa.zip). If not provided, RiboNet will extract UTR lengths from the GFF/GTF file if available. Otherwise, it will take fixed sequences upstream and downstream of the CDS, specified as ``maxutr5length`` and ``maxutr3length``.

If multiple isoforms exist for a gene, the longest UTR is used.

Data Quality Guidelines
-----------------------

For best results with RiboNet:

Ribosome Profiling Data
~~~~~~~~~~~~~~~~~~~~~~~~

- **Read depth**: ≥10 million mapped reads to CDS per sample
- **Replicates**: ≥2 biological replicates per condition
- **Library quality**: Strong 3-nt periodicity in CDS regions
- **Alignment**: Mapped to genome with splice-aware aligner.

RNA-seq Data
~~~~~~~~~~~~

- **Read depth**: ≥5 million mapped reads per sample
- **Matching samples**: One sample per condition, matching the ribosome profiling conditions

Genome Annotation
~~~~~~~~~~~~~~~~~

- **UTR annotation**: Include 5' and 3' UTR boundaries if available
- **Gene models**: Use primary/canonical transcripts for each gene if multiple isoforms exist
