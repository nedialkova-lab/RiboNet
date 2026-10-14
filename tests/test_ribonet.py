"""Fast checks of the RiboNet installation (CPU only, ~1 min).

Run with ``pytest tests/``. The end-to-end pipeline test is ``bash tests/run_test.sh``.
"""

import importlib
import os
import pkgutil
import signal
import subprocess
import sys

import pytest

import ribonet

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
needs_data = pytest.mark.skipif(not os.path.isfile(os.path.join(DATA, "annotation.gff3")),
                                reason="test dataset not available (clone the GitHub repository)")


def test_all_modules_import():
    for mod in pkgutil.walk_packages(ribonet.__path__, "ribonet."):
        importlib.import_module(mod.name)


def test_package_data():
    root = os.path.dirname(ribonet.__file__)
    assert os.path.isfile(os.path.join(root, "data", "RBP_motifs", "attract_ornament.meme"))
    assert os.path.isfile(os.path.join(root, "data", "img", "ribonet.png"))


@pytest.mark.parametrize("args", [["--version"], ["--help"], ["preprocess", "--help"], ["train", "--help"],
                                  ["interpret", "--help"], ["score_batch", "--help"], ["report", "--help"]])
def test_cli(args):
    out = subprocess.run([sys.executable, "-m", "ribonet"] + args, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    if args == ["--version"]:
        assert ribonet.__version__ in out.stdout


def test_reverse_complement():
    from ribonet.utils.seq_utils import reverse_complement
    assert reverse_complement("ATGCcgNa") == "tNcgGCAT"


@needs_data
def test_fetch_sequence():
    import pysam
    from ribonet.utils.seq_utils import fetch_sequence, reverse_complement
    fasta = os.path.join(DATA, "genome.fa")
    with pysam.FastaFile(fasta) as fa:
        ref = fa.fetch("chrII", 1000, 1050)
    assert fetch_sequence(fasta, "chrII", 1000, 1050, "+") == ref
    assert fetch_sequence(fasta, "chrII", 1000, 1050, "-") == reverse_complement(ref)


@needs_data
def test_cds_extraction():
    from ribonet.utils.annotation_utils import AnnotationDB
    from ribonet.utils.seq_utils import START_CODON, STOP_CODONS
    db = AnnotationDB(os.path.join(DATA, "annotation.gff3"))
    cds = db.extract_cds_sequences(os.path.join(DATA, "genome.fa"))
    assert len(cds) > 500
    for seq in cds.values():
        assert len(seq) % 3 == 0
        assert seq[-3:] in STOP_CODONS
    # nearly all yeast ORFs start with ATG (a few annotated ones do not)
    assert sum(seq[:3] in START_CODON for seq in cds.values()) / len(cds) > 0.95
    tx_to_gene = db.build_feature_to_gene_map()
    assert {tx_to_gene[t] for t in cds} == {t[:-len("_mRNA")] for t in cds}


@needs_data
def test_riboa_offsets(tmp_path):
    """RiboA recovers the simulated A-site offsets of long footprints (15 nt for 28 nt, 16 nt for 30 nt)."""
    from ribonet.aip import run_offset
    from ribonet.utils.annotation_utils import AnnotationDB
    tab = str(tmp_path / "cds_info.tab")
    AnnotationDB(os.path.join(DATA, "annotation.gff3")).write_aip_tab(tab)
    offsets = run_offset(str(tmp_path), os.path.join(DATA, "WT_rep1.bam"), tab, os.path.join(DATA, "genome.fa"),
                         27, 32, False, 0, 0.5, 50.0, 5.0, None, False, "genome", False, 0, "A", False)
    assert offsets[28][0] == "15"
    assert offsets[30][0] == "16"


def _square(x):
    return x * x


def _kill_on_three(x):
    if x == 3:
        os.kill(os.getpid(), signal.SIGKILL)  # like the out-of-memory killer
    return x


def test_parallel_map_order():
    from ribonet.utils.parallel import parallel_map
    expected = [x * x for x in range(100)]
    assert parallel_map(_square, range(100), threads=1) == expected
    assert parallel_map(_square, range(100), threads=4, chunksize=7) == expected


def test_parallel_map_killed_worker():
    """A killed worker raises an error instead of hanging forever."""
    from ribonet.utils.parallel import parallel_map
    with pytest.raises(RuntimeError, match="ran out of memory"):
        parallel_map(_kill_on_three, range(20), threads=2)
