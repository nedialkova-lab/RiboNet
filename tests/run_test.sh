#!/usr/bin/env bash
# End-to-end RiboNet test on the small synthetic dataset in tests/data
# (2 conditions x 2 replicates, short and long footprints, 561 yeast genes).
#
# Usage:  bash run_test.sh [output_dir] [threads]
#   output_dir  where results are written (default: tests/test_output)
#   threads     CPU threads for preprocessing and report (default: 4)
#
# Takes ~1 h on an NVIDIA A40 GPU (mostly in silico mutagenesis in steps 3 and 4). Set RIBONET="python -m ribonet" to test a source checkout.
set -euo pipefail

TESTDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$(realpath -m "${1:-$TESTDIR/test_output}")"
THREADS="${2:-4}"
RIBONET="${RIBONET:-ribonet}"

cd "$TESTDIR"  # BAM paths in data/sample_file.txt are relative to this folder

echo "==> RiboNet $($RIBONET --version) test, writing results to $OUT"

# 1. Preprocessing: A-site offset estimation (RiboA, using all genes), ribosome profiles and
#    RNA abundances. Only short transcripts (~190 genes) are kept for training to keep the
#    in silico mutagenesis of step 3 short; real analyses normally keep the defaults.
$RIBONET preprocess \
    --sample_file data/sample_file.txt \
    --rna_counts data/rna_counts.txt \
    --genome_fasta data/genome.fa \
    --gff data/annotation.gff3 \
    --project_dir "$OUT" \
    --threads "$THREADS" \
    --maxutr5length 100 \
    --maxutr3length 100 \
    --maxlength 1200

# 2. Training (codon resolution, ResNet embeddings)
$RIBONET train \
    --project_dir "$OUT" \
    --resolution codon

# 3. In silico mutagenesis, in a single batch
$RIBONET interpret \
    --project_dir "$OUT" \
    --n_batches 1

# 4. Interpretation report: KD compared to WT
$RIBONET report \
    --project_dir "$OUT" \
    --control WT \
    --threads "$THREADS"

for f in riboA/QC_report.pdf training/training.pdf interpretation/interpretation.pdf; do
    if [ ! -s "$OUT/$f" ]; then
        echo "TEST FAILED: $OUT/$f was not created" >&2
        exit 1
    fi
done
echo "==> TEST PASSED. Reports: $OUT/riboA/QC_report.pdf, $OUT/training/training.pdf, $OUT/interpretation/interpretation.pdf"
