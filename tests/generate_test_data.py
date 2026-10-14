#!/usr/bin/env python3
"""Generate the synthetic RiboNet test dataset in ``tests/data``.

The dataset is fully simulated: only the genome sequence and gene models are
real, taken from the public *S. cerevisiae* S288C reference (SGD release
R64-2-1). Ribosome footprints (BAM) and RNA-seq counts are random draws from a
simple model of translation whose parameters (footprint length distribution,
reading frame, A-site offsets, fraction of reads in UTRs, initiation and
termination peaks, gene expression spread) mimic typical yeast ribosome
profiling data.

Design
------
* Genome: whole chromosomes (default chrII and chrIII, ~520 genes).
* Annotation: GFF3 with gene / mRNA / exon / CDS features for every protein
  coding gene fully contained in the region. mRNA ends follow the SGD UTR
  annotations when available (otherwise random UTR lengths).
* Two conditions (WT, KD) with two replicates each. Each BAM contains short
  (20-23 nt) and long (27-32 nt) footprints, plus some reads of other lengths.
* KD effect: slower decoding (higher A-site occupancy) at AAA, CAA and GAA
  codons, mimicking the loss of the mcm5s2U tRNA modification, plus small
  random changes in mRNA abundance.

Usage
-----
The public input files can be downloaded from the SGD archive
(http://sgd-archive.yeastgenome.org/sequence/S288C_reference/)::

    python generate_test_data.py \\
        --genome sacCer3.fa \\
        --gff saccharomyces_cerevisiae_R64-2-1_20150113.gff \\
        --utr5 SGD_all_ORFs_5prime_UTRs.fsa \\
        --utr3 SGD_all_ORFs_3prime_UTRs.fsa \\
        --outdir data
"""

import argparse
import os
import re
from urllib.parse import unquote

import numpy as np
import pysam

# --- Empirical parameters (typical yeast ribosome profiling) -----------------
# Footprint length distributions within each class
LONG_LENGTHS = {27: 0.04, 28: 0.14, 29: 0.26, 30: 0.28, 31: 0.19, 32: 0.09}
SHORT_LENGTHS = {20: 0.16, 21: 0.27, 22: 0.32, 23: 0.25}
OTHER_LENGTHS = {16: 0.06, 17: 0.09, 18: 0.17, 19: 0.10, 24: 0.14, 25: 0.13,
                 26: 0.14, 33: 0.11, 34: 0.04, 35: 0.02}
# Short footprints are slightly enriched (~15% in real data) to give the short-footprint
# tracks enough coverage in this small dataset.
CLASS_FRACTIONS = {"long": 0.68, "short": 0.22, "other": 0.10}
# A-site offset from the read 5' end, per footprint length
ASITE_OFFSET = {20: 15, 21: 15, 22: 16, 23: 16, 27: 15, 28: 15, 29: 16, 30: 16, 31: 16, 32: 16}
# Fraction of 5' ends in frame 0/1/2 of the CDS, per footprint length
FRAME_PROBS = {20: [0.39, 0.28, 0.33], 21: [0.50, 0.21, 0.29], 22: [0.37, 0.19, 0.44],
               23: [0.28, 0.25, 0.47], 27: [0.24, 0.49, 0.27], 28: [0.74, 0.16, 0.10],
               29: [0.50, 0.08, 0.42], 30: [0.34, 0.16, 0.50], 31: [0.27, 0.21, 0.52],
               32: [0.18, 0.28, 0.54]}
# UTR read density relative to the mean CDS density (per nt)
UTR_DENSITY = {"long": (0.10, 0.055), "short": (0.22, 0.17)}
# Initiation (P-site at AUG) and termination (A-site at stop) peaks
START_PEAK = {"long": 2.5, "short": 3.0}
STOP_PEAK = {"long": 12.0, "short": 6.0}
# KD: slower decoding of mcm5s2U-dependent codons
KD_CODONS = {"AAA": (2.5, 4.0), "CAA": (2.5, 4.0), "GAA": (2.5, 4.0)}  # (long, short) fold change
CONDITIONS = ["WT", "KD"]
REPLICATES = 2
FLANK = 100  # genomic nt kept on each side of a transcript for reads overhanging its ends

COMPLEMENT = str.maketrans("ACGTN", "TGCAN")
CODONS = [a + b + c for a in "ACGT" for b in "ACGT" for c in "ACGT"]


def read_fasta(path, chrom):
    seq, keep = [], False
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                if keep:
                    break
                keep = line[1:].split()[0] == chrom
            elif keep:
                seq.append(line.strip())
    if not seq:
        raise ValueError(f"{chrom} not found in {path}")
    return "".join(seq).upper()


def read_genes(gff, chrom, end):
    """Protein-coding genes (no frameshift) fully inside chrom:1-end, 600 nt away from its ends."""
    genes, cds, skip = {}, {}, set()
    with open(gff) as fh:
        for line in fh:
            if line.startswith("##FASTA"):
                break
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9 or f[0] != chrom:
                continue
            attrs = dict(kv.split("=", 1) for kv in f[8].split(";") if "=" in kv)
            if f[2] == "gene":
                genes[attrs["ID"]] = (int(f[3]), int(f[4]), f[6])
            elif f[2] == "CDS":
                gene = re.sub(r"_mRNA$", "", attrs["Parent"].split(",")[0])
                cds.setdefault(gene, []).append((int(f[3]) - 1, int(f[4])))  # 0-based half-open
            elif "frameshift" in f[2]:
                skip.add(unquote(attrs.get("Parent", "")))
    out = {}
    for g, (s, e, strand) in genes.items():
        if g in skip or g not in cds or e > end - 600 or s < 600:
            continue
        segs = sorted(cds[g])
        if sum(b - a for a, b in segs) % 3 != 0:
            continue
        out[g] = {"chrom": chrom, "strand": strand, "cds": segs}
    return out


def read_utr_lengths(path):
    """Longest annotated UTR length per gene from the SGD UTR FASTA headers."""
    lengths = {}
    if path is None:
        return lengths
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                gene = line[1:].split("_")[4]
                m = re.search(r"range=\S+:(\d+)-(\d+)", line)
                L = int(m.group(2)) - int(m.group(1)) + 1
                lengths[gene] = max(lengths.get(gene, 0), L)
    return lengths


def build_transcripts(genes, utr5, utr3, rng, chrom_lens):
    """Add UTRs and the transcript -> genome coordinate map to each gene."""
    for g, d in genes.items():
        chrom_len = chrom_lens[d["chrom"]]
        l5 = utr5.get(g, int(rng.lognormal(np.log(60), 0.6)))
        l3 = utr3.get(g, int(rng.lognormal(np.log(120), 0.5)))
        l5, l3 = int(np.clip(l5, 10, 500)), int(np.clip(l3, 30, 500))
        exons = [list(x) for x in d["cds"]]
        if d["strand"] == "+":
            exons[0][0] -= l5
            exons[-1][1] += l3
        else:
            exons[-1][1] += l5
            exons[0][0] -= l3
        exons[0][0], exons[-1][1] = max(exons[0][0], 0), min(exons[-1][1], chrom_len)
        pos = np.concatenate([np.arange(a, b) for a, b in exons])
        left = np.arange(pos[0] - FLANK, pos[0])
        right = np.arange(pos[-1] + 1, pos[-1] + 1 + FLANK)
        pos = np.concatenate([left, pos, right])
        if d["strand"] == "-":
            pos = pos[::-1]
        d.update(exons=exons, utr5=l5, utr3=l3, cds_len=sum(b - a for a, b in d["cds"]),
                 tx2g=pos, tx_start=FLANK)  # transcript (incl. flanks) index -> genome position


def write_gff(genes, chroms, path):
    with open(path, "w") as fh:
        fh.write("##gff-version 3\n")
        for g, d in sorted(genes.items(), key=lambda x: (chroms.index(x[1]["chrom"]), x[1]["exons"][0][0])):
            s, e = d["exons"][0][0] + 1, d["exons"][-1][1]
            st, chrom = d["strand"], d["chrom"]
            fh.write(f"{chrom}\tsimulated\tgene\t{s}\t{e}\t.\t{st}\t.\tID={g};Name={g}\n")
            fh.write(f"{chrom}\tsimulated\tmRNA\t{s}\t{e}\t.\t{st}\t.\tID={g}_mRNA;Parent={g}\n")
            for a, b in d["exons"]:
                fh.write(f"{chrom}\tsimulated\texon\t{a + 1}\t{b}\t.\t{st}\t.\tParent={g}_mRNA\n")
            # CDS phases in transcription order
            segs = d["cds"] if st == "+" else d["cds"][::-1]
            done = 0
            phases = []
            for a, b in segs:
                phases.append((3 - done % 3) % 3)
                done += b - a
            if st == "-":
                phases = phases[::-1]
            for (a, b), ph in zip(d["cds"], phases):
                fh.write(f"{chrom}\tsimulated\tCDS\t{a + 1}\t{b}\t.\t{st}\t{ph}\tParent={g}_mRNA\n")


def simulate(genes, genome, rng, outdir, reads_per_sample):
    names = sorted(genes)
    n = len(names)
    # Gene level: mRNA abundance, translation efficiency, KD expression changes
    mrna = rng.lognormal(0, 1.4, n)
    te = rng.lognormal(0, 0.5, n)
    kd_lfc = rng.normal(0, 0.3, n)
    mrna_cond = {"WT": mrna, "KD": mrna * 2 ** kd_lfc}
    # Codon dwell times (A-site), different for long and short footprints
    dwell = {"long": dict(zip(CODONS, rng.lognormal(0, 0.35, 64))),
             "short": dict(zip(CODONS, rng.lognormal(0, 0.5, 64)))}

    # Per-gene relative occupancy along the extended transcript (A-site position, nt)
    occupancy = {}  # (gene, cls, cond) -> array over transcript positions
    for g in names:
        d = genes[g]
        tx = "".join(genome[d["chrom"]][p] for p in d["tx2g"])
        if d["strand"] == "-":
            tx = tx.translate(COMPLEMENT)
        d["tx_seq"] = tx
        cs = d["tx_start"] + d["utr5"]
        ncod = d["cds_len"] // 3
        noise = rng.gamma(4, 1 / 4, ncod)  # sequence-independent variability, shared by samples
        ramp = 1 + 0.5 * np.exp(-np.arange(ncod) / 15)
        for cls in ("long", "short"):
            for cond in CONDITIONS:
                occ = np.zeros(len(tx))
                cod = np.array([dwell[cls].get(tx[cs + 3 * i:cs + 3 * i + 3], 1.0) for i in range(ncod)])
                if cond == "KD":
                    j = 0 if cls == "long" else 1
                    cod *= [KD_CODONS[c][j] if c in KD_CODONS else 1.0
                            for c in (tx[cs + 3 * i:cs + 3 * i + 3] for i in range(ncod))]
                cod *= noise * ramp
                cod[1] *= START_PEAK[cls]   # A-site at codon 2 = P-site at AUG
                cod[-1] *= STOP_PEAK[cls]   # A-site at the stop codon
                occ[cs:cs + 3 * ncod:3] = cod
                mean_nt = cod.mean() / 3
                k5, k3 = UTR_DENSITY[cls]
                occ[d["tx_start"]:cs] = k5 * mean_nt
                occ[cs + 3 * ncod:cs + 3 * ncod + d["utr3"]] = k3 * mean_nt
                occupancy[(g, cls, cond)] = occ

    header = {"HD": {"VN": "1.6", "SO": "unsorted"},
              "SQ": [{"SN": c, "LN": len(seq)} for c, seq in genome.items()],
              "PG": [{"ID": "generate_test_data.py", "PN": "generate_test_data.py"}]}
    samples = []
    for cond in CONDITIONS:
        for rep in range(1, REPLICATES + 1):
            name = f"{cond}_rep{rep}"
            nreads = int(reads_per_sample * rng.uniform(0.85, 1.15))
            rep_noise = rng.lognormal(0, 0.15, n)  # biological replicate variability
            unsorted = os.path.join(outdir, name + ".unsorted.bam")
            with pysam.AlignmentFile(unsorted, "wb", header=header) as bam:
                rid = 0
                for cls, frac in CLASS_FRACTIONS.items():
                    occ_cls = "long" if cls == "other" else cls
                    lens = LONG_LENGTHS if cls == "long" else SHORT_LENGTHS if cls == "short" else OTHER_LENGTHS
                    # expected reads per gene: initiation rate x total dwell time
                    w = np.array([mrna_cond[cond][i] * te[i] * rep_noise[i] *
                                  occupancy[(g, occ_cls, cond)].sum() for i, g in enumerate(names)])
                    per_gene = rng.multinomial(int(nreads * frac), w / w.sum())
                    for i, g in enumerate(names):
                        if per_gene[i] == 0:
                            continue
                        d, occ = genes[g], occupancy[(g, occ_cls, cond)]
                        asites = rng.choice(len(occ), size=per_gene[i], p=occ / occ.sum())
                        rlens = rng.choice(list(lens), size=per_gene[i], p=list(lens.values()))
                        cs = d["tx_start"] + d["utr5"]
                        for a, L in zip(asites, rlens):
                            if L in ASITE_OFFSET:
                                frame = rng.choice(3, p=FRAME_PROBS[L])
                                off = ASITE_OFFSET[L]
                                # shift the 5' end by -1/0/+1 nt to reach the sampled frame
                                delta = (frame - (a - off - cs)) % 3
                                five = a - off + (delta if delta < 2 else -1)
                            else:
                                five = a - 15 + rng.integers(-1, 2)
                            if five < 0 or five + L > len(d["tx2g"]):
                                continue
                            gpos = d["tx2g"][five:five + L]
                            if np.any(np.abs(np.diff(gpos)) != 1):
                                continue  # spans an intron
                            left = int(gpos.min())
                            seg = pysam.AlignedSegment()
                            seg.query_name = f"r{rid}"
                            seg.flag = 0 if d["strand"] == "+" else 16
                            seg.reference_id = list(genome).index(d["chrom"])
                            seg.reference_start = left
                            seg.mapping_quality = 255
                            seg.cigarstring = f"{L}M"
                            seg.query_sequence = genome[d["chrom"]][left:left + L]
                            seg.set_tag("NH", 1)
                            bam.write(seg)
                            rid += 1
            bam_path = os.path.join(outdir, name + ".bam")
            pysam.sort("-o", bam_path, unsorted)
            os.remove(unsorted)
            pysam.index(bam_path)
            samples.append((name, cond, rid))
            print(f"{name}: {rid} reads")

    # RNA-seq counts per condition (featureCounts-like table)
    with open(os.path.join(outdir, "rna_counts.txt"), "w") as fh:
        fh.write("Geneid\tLength\t" + "\t".join(CONDITIONS) + "\n")
        for i, g in enumerate(names):
            L = genes[g]["cds_len"]
            counts = []
            for cond in CONDITIONS:
                mu = mrna_cond[cond][i] * L * 0.4
                counts.append(rng.negative_binomial(10, 10 / (10 + mu)))
            fh.write(f"{g}\t{L}\t" + "\t".join(map(str, counts)) + "\n")

    with open(os.path.join(outdir, "sample_file.txt"), "w") as fh:
        fh.write("sample_id\tcondition\treadlength\tbam\n")
        for name, cond, _ in samples:
            rep = name.split("_rep")[1]
            for cls, rl in (("long", "27-32"), ("short", "20-23")):
                fh.write(f"{cond}_{cls}{rep}\t{cond}\t{rl}\tdata/{name}.bam\n")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--genome", required=True, help="S. cerevisiae genome FASTA (SGD R64-2-1)")
    p.add_argument("--gff", required=True, help="SGD GFF3 annotation")
    p.add_argument("--utr5", default=None, help="SGD 5' UTR FASTA (optional)")
    p.add_argument("--utr3", default=None, help="SGD 3' UTR FASTA (optional)")
    p.add_argument("--chroms", default="chrII,chrIII", help="Comma-separated chromosomes to keep")
    p.add_argument("--reads_per_sample", type=int, default=850000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outdir", default="data")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.outdir, exist_ok=True)
    chroms = args.chroms.split(",")
    genome = {c: read_fasta(args.genome, c) for c in chroms}
    with open(os.path.join(args.outdir, "genome.fa"), "w") as fh:
        for c, seq in genome.items():
            fh.write(f">{c}\n")
            for i in range(0, len(seq), 60):
                fh.write(seq[i:i + 60] + "\n")
    genes = {}
    for c in chroms:
        genes.update(read_genes(args.gff, c, len(genome[c])))
    build_transcripts(genes, read_utr_lengths(args.utr5), read_utr_lengths(args.utr3), rng,
                      {c: len(seq) for c, seq in genome.items()})
    write_gff(genes, chroms, os.path.join(args.outdir, "annotation.gff3"))
    print(f"{len(genes)} genes in {', '.join(chroms)}")
    simulate(genes, genome, rng, args.outdir, args.reads_per_sample)


if __name__ == "__main__":
    main()
