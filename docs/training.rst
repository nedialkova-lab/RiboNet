Training
========

The training step trains a deep neural network model to predict ribosome counts and distribution along transcripts from sequence.

Overview
--------

The ``ribonet train`` command performs the following tasks:

1. Loads preprocessed data from the project directory
2. Extracts sequence embeddings using pre-trained language models (if specified)
3. Trains a deep learning model to predict ribosome counts and distribution
4. Evaluates model performance on test set
5. Saves trained model and generates training report

Basic Usage
-----------

**For codon-resolution models:**

.. code-block:: bash

   ribonet train \
     --project_dir my_project \
     --resolution codon

   ribonet train \
     --project_dir my_project \
     --resolution codon \
     --lm5 specieslm5 \
     --lm3 specieslm3 \
     --species_proxy5 kazachstania_africana_cbs_2517_gca_000304475 \
     --species_proxy3 candida_glabrata

**For nucleotide-resolution models:**

.. code-block:: bash

   ribonet train \
     --project_dir my_project \
     --resolution nucleotide

   ribonet train \
     --project_dir my_project \
     --resolution nucleotide \
     --lm rinalmo

Required Arguments
------------------

``--resolution``
~~~~~~~~~~~~~~~~

**Default:** ``codon``

**Choices:** ``codon``, ``nucleotide``

Resolution for ribosome density profiles:

- ``codon``: Aggregate counts per codon (recommended unless looking for nucleotide-level features like ribosome frameshifting)
- ``nucleotide``: Single-nucleotide resolution (higher computation requirements). It can run annotation-free, i.e. without the model knowing the CDS start and end.

.. code-block:: bash

   # Codon resolution (default)
   ribonet train --resolution codon ...
   
   # Nucleotide resolution
   ribonet train --resolution nucleotide ...

``--project_dir``
~~~~~~~~~~~~~~~~~

Path to project directory created by ``ribonet preprocess``. Must contain the ``preprocessing/`` subdirectory with the preprocessed data. Training outputs are written to ``training/``.

Language Model Arguments
------------------------

For codon-resolution models
~~~~~~~~~~~~~~~~~~~~~~~~~~~

``--lm5``
^^^^^^^^^

**Choices:** ``specieslm5``, ``rinalmo``

Language model for the 5' region (5' UTR + start of CDS). If omitted, no language model is used (one-hot encoding + residual neural network).

- ``specieslm5``: SpeciesLM model pre-trained on upstream sequences from start codon of fungi
- ``rinalmo``: RiNALMo pre-trained RNA model (requires ``pip install multimolecule``)

``--lm3``
^^^^^^^^^

**Choices:** ``specieslm3``, ``rinalmo``

Language model for the 3' region (end of CDS + 3' UTR). If omitted, no language model is used (one-hot encoding + residual neural network).

- ``specieslm3``: SpeciesLM model pre-trained on downstream sequences from stop codon of fungi
- ``rinalmo``: RiNALMo pre-trained RNA model (requires ``pip install multimolecule``)

``--species_proxy5``
^^^^^^^^^^^^^^^^^^^^

Species name for SpeciesLM 5' embeddings. Required if ``--lm5`` is ``specieslm5``.

``--species_proxy3``
^^^^^^^^^^^^^^^^^^^^

Species name for SpeciesLM 3' embeddings. Required if ``--lm3`` is ``specieslm3``.

``--model_dir5`` and ``--model_dir3``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Default:** ``none``

Directories to load the 5' and 3' language models from (or to save them to after the first download). If not specified, models are downloaded if not previously cached.

``--utr_pool``
^^^^^^^^^^^^^^

**Default:** ``0.3``

Pooling factor for UTR regions in codon-resolution models. UTRs are condensed by pooling adjacent nucleotides together to reduce sequence length and model complexity. A value of 0.3 means that UTRs will be pooled to 30% of their original length.

.. code-block:: bash

   ribonet train --utr_pool 0.5 ...

For nucleotide-resolution models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``--lm``
^^^^^^^^

**Choices:** ``rinalmo``

RNA language model for the entire transcript. If omitted, no language model is used (one-hot encoding).

- ``rinalmo``: RiNALMo pre-trained RNA model (requires ``pip install multimolecule``)

``--cache_dir``
^^^^^^^^^^^^^^^

**Default:** Hugging Face default cache (``~/.cache/huggingface``)

Directory for caching language model weights (both resolutions).

.. code-block:: bash

   ribonet train --cache_dir /scratch/cache ...

``--model_dir``
^^^^^^^^^^^^^^^

**Default:** ``none``

Directory to load the language model from (or to save it to after the first download). If not specified, models will be downloaded if not previously cached.


Training Hyperparameters
------------------------

``--epoch``
~~~~~~~~~~~

**Default:** ``50``

Maximum number of training epochs. Early stopping is applied if validation loss does not improve for 10 consecutive epochs.

.. code-block:: bash

   ribonet train --epoch 20 ...

``--warmup``
~~~~~~~~~~~~

**Default:** ``5``

Number of warmup epochs with a lower increasing learning rate (starting at 0.1% of main learning rate) to stabilize training.


``--batch_size``
~~~~~~~~~~~~~~~~

**Default:** ``32``

Batch size for training. Reduce if GPU memory is limited.

.. code-block:: bash

   ribonet train --batch_size 16 ...

``--learning``
~~~~~~~~~~~~~~

**Default:** ``0.0005``

Learning rate for AdamW optimizer.

.. code-block:: bash

   ribonet train --learning 0.0001 ...

``--weight_decay``
~~~~~~~~~~~~~~~~~~

**Default:** ``0.01``

Weight decay (AdamW regularization).

.. code-block:: bash

   ribonet train --weight_decay 0.0005 ...

``--alpha`` and ``--beta``
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Default:** alpha = ``0.1`` (counts) and beta = ``1.0`` (profile) for codon resolution, alpha = ``0.1`` and beta = ``4.5`` for nucleotide resolution

Loss weights for count and distribution prediction. See the loss function section below for details.

.. code-block:: bash

   ribonet train --alpha 1.0 --beta 0.5 ...

``--split``
~~~~~~~~~~~~~~

**Default:** ``0.7``

Fraction of transcripts to use for training. The remaining transcripts are split into validation and test sets (15% each). Adjust to use more or less data for training.

The transcripts are assigned randomly, and the assignment is saved to ``training/split.tsv``. Running ``ribonet train`` again on a project with a trained model does not retrain and keeps its saved split, so that the stored predictions and the evaluation stay consistent. To train a new model with a new random split, use a new project directory.

.. code-block:: bash

   ribonet train --split 0.8 ...

Other Arguments
---------------

``--ablate_utr5`` and ``--ablate_utr3``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Default:** off

Ablation controls: zero out the 5' UTR (or 3' UTR) embedding, so that the model only uses the remaining sequence regions. Useful to quantify the contribution of each UTR to the predictions.

``--id``
~~~~~~~~

Optional descriptive identifier for the training run, stored in ``config_ribonet.json``.

``--verbose``
~~~~~~~~~~~~~

**Default:** ``1``

Verbosity level: ``0`` for warnings only, ``1`` for informative messages. Available for all subcommands.

Model Architecture Details
--------------------------

Codon-resolution model
~~~~~~~~~~~~~~~~~~~~~~

**Input:**

- 5' UTR sequence (language model embeddings or nucleotide one-hot)
- CDS sequence (codons one-hot)
- 3' UTR sequence (language model embeddings or nucleotide one-hot)
- RNA expression levels per condition
   
**Architecture:**

- UTRs are processed through a residual neural network if no language model is used, else a weighted average of the language model embeddings layers is used (with learnable weights) to get a single embedding per position. Hidden dimensions are downprojected with a dropout and linear layer.
- CDS is processed through a separate 5-layer convolutional block to extract local features.
- After concatenating 5' UTR, CDS, and 3' UTR and adding a positional encoding, transformer encoder layer captures long-range interactions.
- Separate heads for count and distribution prediction, one for each condition. Count head uses global average pooling followed by fully connected layers. Distribution head uses a convolutional layer followed by a softmax to predict probability per position.
   
**Output:**

- Total ribosome counts per condition
- Ribosome probability per transcript position

Nucleotide-resolution model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Input:**

- Full transcript sequence (language model embeddings or nucleotide one-hot)
- RNA expression levels per condition
   
**Architecture:**

- Sequence is processed through a residual neural network if no language model is used, else a weighted average of the language model embeddings layers is used (with learnable weights) to get a single embedding per position. Hidden dimensions are downprojected with a dropout and linear layer.
- After adding a positional encoding and an annotation encoding (optional, indicating UTR, CDS and coding frame), transformer encoder layer captures long-range interactions.
- Attention mechanisms
- Separate heads for count and distribution prediction, one for each condition. Count head uses global average pooling followed by fully connected layers. Distribution head uses a convolutional layer followed by a softmax to predict probability per position.
   
**Output:**

- Total ribosome counts per condition
- Ribosome probability per transcript position

Loss Function
-------------

RiboNet uses a combined loss function:

.. math::

   L_{total} = \alpha \cdot L_{count} + \beta \cdot L_{dist}

Where:
- :math:`L_{count}`: Mean squared error for total count prediction
- :math:`L_{dist}`: Cross-entropy of the read probability per position

See above for adjusting the weights :math:`\alpha` and :math:`\beta` to balance the importance of count vs distribution prediction.

**Biological replicates are not averaged** but treated separately in the loss function, which allows the model to consider and denoise variability across replicates.

Evaluation Metrics
------------------

During training, RiboNet reports:

Count prediction
~~~~~~~~~~~~~~~~

- **Pearson correlation** between predicted and true total counts
- **Percentage of variance explained (R²)** across all transcripts and conditions

Distribution prediction
~~~~~~~~~~~~~~~~~~~~~~~

- **Spearman correlation** between predicted and true distribution profiles in coding sequences (per position, after multiplying predicted probabilities by transcript counts to get predicted counts per position)
- **Normalized codon occupancy**: density of reads at each codon type normalized by the average density across the full coding sequence, and averaged across genes.

Output Files
------------

See :doc:`outputs` for detailed descriptions of training output files:

- ``ribonet.pt``: Trained model weights
- ``config_ribonet.json``: Model configuration
- ``training.pdf``: Training summary report
- ``model_prediction.parquet``: Predictions on test set transcripts
- ``model_trainingset.parquet``: Predictions on training set transcripts

Troubleshooting
---------------

Model not learning
~~~~~~~~~~~~~~~~~~

**Symptoms:** Loss not decreasing, evaluation metrics stay low

**Solutions:**

- Increase learning rate, warm-up, or batch size
- If performance failure comes from counts but not distribution, or vice versa, adjust the weights of ``alpha`` and ``beta``
- Reduce model complexity by using a codon-resolution model, applying UTR pooling, or decreasing hidden dimensions
- Check data quality in preprocessing parquet files

Overfitting
~~~~~~~~~~~

**Symptoms:** Training loss decreases but validation loss increases

**Solutions:**

- Increase dropout
- Increase weight decay
- Reduce model complexity by using a codon-resolution model, applying UTR pooling, or decreasing hidden dimensions
- Use more data (less stringent filtering in preprocessing)

GPU out of memory
~~~~~~~~~~~~~~~~~

**Solutions:**

- Reduce batch size
- Reduce model complexity by using a codon-resolution model, applying UTR pooling, or decreasing hidden dimensions
- Use a GPU with more memory

Language model download fails
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Solutions:**

- Check your internet connection
- Ensure you have sufficient disk space (~5-10 GB per model)
- Set cache and model directories explicitly: ``--cache_dir /path/to/cache`` and ``--model_dir /path/to/models``
