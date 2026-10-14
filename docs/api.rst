API Reference
=============

Each ``ribonet`` subcommand is a thin wrapper around the Python functions and
classes below, which can also be used directly, e.g. to run custom analyses on
a trained model:

.. code-block:: python

   from ribonet.interpretation import RiboInterpretation

   interpretation = RiboInterpretation("my_project")
   interpretation.plot_logo(gene="YAL001C", conds=["KD"], ctrl_cond="WT")

Using RiboNet from Python or Jupyter
------------------------------------

The functions behind the command-line subcommands can be called from scripts or notebooks:

.. code-block:: python

   import logging
   from ribonet.ribonet import preprocess_inputs, train_ribonet, interpret_ribonet, report_interpretation

   logging.basicConfig(level=logging.INFO)  # show progress messages

   preprocess_inputs("samples.txt", "rna_counts.txt", "genome.fa", "annotation.gff",
                     "my_project", threads=8)
   train_ribonet("my_project", "codon")
   interpret_ribonet("my_project", n_batches=1, scoring="ISM", save_distribution=False)
   report_interpretation("my_project", ctrl_cond="WT", threads=8, motif_db="default",
                         diffsites=5000, pwind=30, max_seqlets=1000, filename="interpretation.pdf")

Relative BAM paths in the sample file are resolved from the current working directory. With ``threads > 1``, the work is distributed over separate processes; in a plain Python script (not needed in notebooks), run the code under ``if __name__ == "__main__":`` so that it also works on systems that start new processes with ``spawn`` (e.g. macOS).

Command-line entry points
-------------------------

.. automodule:: ribonet.ribonet
   :members: preprocess_inputs, train_ribonet, interpret_ribonet, score_batch, report_interpretation
   :no-special-members:

Preprocessing
-------------

.. autoclass:: ribonet.preprocessing.RiboPreprocess
   :members:

.. autoclass:: ribonet.utils.annotation_utils.AnnotationDB
   :members:

Training
--------

.. autoclass:: ribonet.training.RiboTraining
   :members:

Interpretation
--------------

.. autoclass:: ribonet.interpretation.RiboInterpretation
   :members:
