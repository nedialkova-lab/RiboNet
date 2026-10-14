"""RiboNet: deep learning analysis toolkit for ribosome profiling.

The command-line interface is exposed as ``ribonet`` (see :mod:`ribonet.ribonet`).
The pipeline steps can also be run from Python::

    from ribonet.preprocessing import RiboPreprocess
    from ribonet.training import RiboTraining
    from ribonet.interpretation import RiboInterpretation
"""

from .version import __version__

__all__ = ["__version__"]
