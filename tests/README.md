# RiboNet tests

## End-to-end test

`run_test.sh` runs the complete RiboNet pipeline (`preprocess`, `train`, `interpret`, `report`)
on the synthetic dataset in [`data`](data). Run it on a GPU node (~1 h on an NVIDIA A40):

```bash
bash run_test.sh [output_dir] [threads]
```

It ends with `TEST PASSED` if the QC, training and interpretation reports were generated.
To test a source checkout without installing it, use `RIBONET="python -m ribonet" bash run_test.sh`.

## Quick checks

`test_ribonet.py` checks imports, the command-line interface, CDS extraction and A-site
offset estimation (CPU, ~1 min):

```bash
pytest tests
```

## Test dataset

| File | Description |
| --- | --- |
| `data/sample_file.txt` | Sample sheet: each BAM is listed twice, for short (20-23 nt) and long (27-32 nt) footprints |
| `data/{WT,KD}_rep{1,2}.bam` | Ribosome footprints (2 conditions x 2 replicates), with index |
| `data/rna_counts.txt` | RNA-seq counts per condition, with gene lengths |
| `data/genome.fa` | *S. cerevisiae* chromosomes II and III (SGD R64-2-1) |
| `data/annotation.gff3` | Gene models of 561 protein-coding genes, with UTRs |

The genome and gene models are real; footprints and RNA counts are simulated by
[`generate_test_data.py`](generate_test_data.py) to mimic yeast ribosome profiling data
(footprint length distribution, reading frame, A-site offsets, reads in UTRs, initiation and
termination peaks, gene expression). In the KD condition, AAA, CAA and GAA codons are decoded
more slowly (as upon loss of the mcm5s2U tRNA modification). The dataset can be regenerated
from the public SGD files with the command given in `generate_test_data.py`.
