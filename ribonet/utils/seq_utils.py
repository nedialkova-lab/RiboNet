import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from .model_utils import tokenize
import os
from collections import defaultdict, Counter
import pysam
import logging
log = logging.getLogger(__name__)

Nlist = ["A","C","G","T","N","X"]
START_CODON = {"ATG"}
STOP_CODONS = {"TAA", "TAG", "TGA"}
Cdict = {
    'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M',
    'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
    'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K',
    'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',
    'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L',
    'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
    'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q',
    'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
    'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V',
    'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
    'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E',
    'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
    'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S',
    'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
    'TAC':'Y', 'TAT':'Y', 'TAA':'_', 'TAG':'_',
    'TGC':'C', 'TGT':'C', 'TGA':'_', 'TGG':'W',
    'XXX':'0'}
Clist = sorted(Cdict.keys())

_COMPLEMENT = str.maketrans("ACGTNRYKMBVDHacgtnrykmbvdh", "TGCANYRMKVBHDtgcanyrmkvbhd")
_FASTA_HANDLES = {}


def reverse_complement(seq):
    '''Reverse complement a DNA sequence, preserving case and IUPAC codes.'''
    return seq.translate(_COMPLEMENT)[::-1]


def fetch_sequence(fasta_file, chrom, start, end, strand="+"):
    '''Fetch a genomic interval, equivalent to ``bedtools getfasta -s``.

    Parameters
    ----------
    fasta_file : str
        Path to the genome FASTA file.
    chrom : str
        Chromosome / sequence name.
    start, end : int
        0-based, half-open interval (BED convention).
    strand : str, default = "+"
        If "-", the reverse complement is returned.

    Returns
    -------
    str
        The interval sequence.
    '''
    # One open handle per process (safe with multiprocessing workers).
    key = (fasta_file, os.getpid())
    if key not in _FASTA_HANDLES:
        _FASTA_HANDLES[key] = pysam.FastaFile(fasta_file)
    seq = _FASTA_HANDLES[key].fetch(chrom, max(0, start), end)
    return reverse_complement(seq) if strand == "-" else seq

def build_utr_exon_mask(
    cds_ref: int,
    strand: str,
    exon_intervals: list[tuple[int, int]],
    utr_type: str,
    extend_prof: int,
) -> np.ndarray:
    """Build a boolean exonic mask of length *extend_prof* for a UTR window.

    Each element of the returned array is ``True`` where the corresponding
    transcript-oriented window position is exonic and ``False`` where it is
    intronic.  If *exon_intervals* is empty the whole window is assumed to be
    exonic (safe fallback when no annotation is available).

    Coordinate conventions
    ----------------------
    The RD / sequence arrays stored in the riboA profiles are in **transcript**
    order (position 0 = furthest from the start / stop codon, position
    ``extend_prof - 1`` = adjacent to ATG / stop).  The mapping to genomic
    coordinates depends on strand and UTR type as follows:

    * **5'UTR, + strand** – window position ``i`` → genomic ``cds_ref - extend_prof + i``
      (*cds_ref* = 0-based inclusive start of ATG).
    * **5'UTR, − strand** – window position ``i`` → genomic ``cds_ref + extend_prof - 1 - i``
      (*cds_ref* = 0-based exclusive end of ATG = ``max(f.stop)`` from gffutils).
    * **3'UTR, + strand** – window position ``i`` → genomic ``cds_ref + i``
      (*cds_ref* = 0-based exclusive end of the CDS / stop codon).
    * **3'UTR, − strand** – window position ``i`` → genomic ``cds_ref - 1 - i``
      (*cds_ref* = 0-based inclusive start of the CDS / stop codon = ``min(f.start-1)``).

    Parameters
    ----------
    cds_ref:
        Genomic reference coordinate of the CDS boundary adjacent to this UTR.
        See the table above for the exact definition per strand and UTR type.
    strand:
        ``'+'`` or ``'-'``.
    exon_intervals:
        List of 0-based half-open genomic exon intervals ``(start, end)``
        for this UTR (as returned by :meth:`AnnotationDB.parse_utr_exon_intervals`).
    utr_type:
        ``'5UTR'`` or ``'3UTR'``.
    extend_prof:
        Length of the UTR window (= ``self.extend_prof`` in RiboPreprocess).

    Returns
    -------
    np.ndarray of bool, shape ``(extend_prof,)``
    """
    if not exon_intervals:
        return np.ones(extend_prof, dtype=bool)

    mask = np.zeros(extend_prof, dtype=bool)

    for es, ee in exon_intervals:
        if utr_type == "5UTR":
            if strand == "+":
                # position i → genomic cds_ref - extend_prof + i
                i_start = es - (cds_ref - extend_prof)
                i_end = ee - (cds_ref - extend_prof)
            else:
                # position i → genomic cds_ref + extend_prof - 1 - i
                i_start = cds_ref + extend_prof - ee
                i_end = cds_ref + extend_prof - es
        else:  # 3UTR
            if strand == "+":
                # position i → genomic cds_ref + i
                i_start = es - cds_ref
                i_end = ee - cds_ref
            else:
                # position i → genomic cds_ref - 1 - i
                i_start = cds_ref - ee
                i_end = cds_ref - es

        i_start = max(0, i_start)
        i_end = min(extend_prof, i_end)
        if i_start < i_end:
            mask[i_start:i_end] = True

    # If no exon interval mapped into the UTR window the mask is
    # all-False, which would silently splice out the entire UTR.
    if not mask.any():
        log.warning(
            "build_utr_exon_mask: all-False mask for %s (cds_ref=%d, strand=%s). "
            "No exon interval projects into the UTR window despite %d interval(s) "
            "being provided. Treating the full window as exonic.",
            utr_type, cds_ref, strand, len(exon_intervals),
        )
        return np.ones(extend_prof, dtype=bool)

    return mask


def splice_utr_seq(seq_window: str, mask: np.ndarray, utr_type: str) -> str:
    """Remove intronic nucleotides from a UTR sequence window and pad with 'X'.

    The window sequence has length ``extend_prof + 3`` where the extra 3 nt are
    the ATG (for 5'UTR, at the end) or the stop codon (for 3'UTR, at the
    beginning).  Splicing only affects the *extend_prof* UTR portion; the flanking
    3 nt are always preserved unchanged.

    Padding convention:
    * **5'UTR** – 'X' padding is added to the **left** so the ATG remains at the
      right end of the string.
    * **3'UTR** – 'X' padding is added to the **right** so the stop codon remains
      at the left end of the string.

    Parameters
    ----------
    seq_window:
        Raw window sequence of length ``extend_prof + 3``.
    mask:
        Boolean exonic mask of length ``extend_prof`` (True = exonic).
    utr_type:
        ``'5UTR'`` or ``'3UTR'``.

    Returns
    -------
    str
        Spliced and padded window sequence, same length as *seq_window*.
    """
    extend_prof = len(mask)

    if utr_type == "5UTR":
        utr_seq = seq_window[:extend_prof]
        flanking = seq_window[extend_prof:]  # ATG (3 nt)
        spliced = "".join(c for c, m in zip(utr_seq, mask) if m)
        padded_utr = "X" * (extend_prof - len(spliced)) + spliced
    else:  # 3UTR
        flanking = seq_window[:3]  # stop codon (3 nt)
        utr_seq = seq_window[3:]
        spliced = "".join(c for c, m in zip(utr_seq, mask) if m)
        padded_utr = spliced + "X" * (extend_prof - len(spliced))

    return flanking + padded_utr if utr_type == "3UTR" else padded_utr + flanking


def splice_utr_rd(maskedrd: np.ndarray, mask: np.ndarray, utr_type: str) -> np.ndarray:
    """Remove intronic positions from a single-replicate UTR RD array and pad with NaN.

    Applies the exonic mask, extracts at most *sliceint* positions adjacent to
    the CDS boundary, and pads the rest with ``np.nan`` to restore the original
    ``extend_prof`` shape.

    Parameters
    ----------
    maskedrd:
        1-D RD array of length ``extend_prof`` for one replicate (may already
        contain NaNs from surrounding-CDS masking done upstream).
    mask:
        Boolean exonic mask of length ``extend_prof``.
    utr_type:
        ``'5UTR'`` or ``'3UTR'``.

    Returns
    -------
    np.ndarray, shape ``(extend_prof,)``
        Spliced RD with NaN padding maintaining the original array shape.
    """
    extend_prof = len(mask)
    # Set intronic positions to NaN then extract only exonic positions
    tmp = maskedrd.copy()
    tmp[~mask] = np.nan
    exonic_rd = tmp[mask]  # length = sum(mask)
    exonic_len = len(exonic_rd)

    condRD = np.full(extend_prof, np.nan)
    if exonic_len > 0:
        if utr_type == "5UTR":
            # pad left; place exonic values at the right end
            condRD[-exonic_len:] = exonic_rd[-exonic_len:]
        else:  # 3UTR
            # pad right; place exonic values at the left end
            condRD[:exonic_len] = exonic_rd[:exonic_len]

    return condRD


def fasta2tab(filename, utr):
    '''
    Convert a FASTA file to a tabular DataFrame format.

    Parameters
    ----------
    filename : str
        Path to the FASTA file.

    Returns
    -------
    pd.DataFrame
        DataFrame containing gene, chr, start, end, strand, length, and seq columns.
    '''
    # Open the file
    fileHandle = open(filename)
    # Read all the lines in the file
    seqs = fileHandle.read().split(">") # Split file whenever a > is detected
    # Close the file
    fileHandle.close()
    # Loop over the lines
    outdf = pd.DataFrame(columns=["gene","chr","start", "end", "strand", "length"])
    for n,seq in enumerate(seqs):
        if seq: # Skip empty sequences
            splitted = seq.split("\n") # Split different lines of one same sequence
            # the first line is the name of the sequence, the rest is the sequence itself
            gene_name = splitted[0].split("_")[4]
            coord = splitted[0].split(" ")[1].split("=")[1]
            strand = splitted[0].split(" ")[4].split("=")[1]
            start = int(coord.split(":")[1].split("-")[0])
            end = int(coord.split(":")[1].split("-")[1])
            if (utr=="3UTR" and strand=="-") or (utr=="5UTR" and strand=="+"):
                start-=1
                end-=1
            outdf.loc[n] = [gene_name,coord.split(":")[0],start,end,strand,
                            abs(end - start)]
    return outdf

def trim_y_c_arrays(y_c, x_c):
    '''
    Trim output arrays to match annotated UTR lengths.

    Parameters
    ----------
    y_c : pd.DataFrame
        Output labels containing ribosome profiling data.
    x_c : pd.DataFrame
        Input features containing UTR length information.

    Returns
    -------
    pd.DataFrame
        Trimmed output labels.
    '''
    for gene in y_c.index:
        utr5_len = x_c.loc[gene, 'utr5_len'] if 'utr5_len' in x_c.columns else None
        utr3_len = x_c.loc[gene, 'utr3_len'] if 'utr3_len' in x_c.columns else None
        
        for col in y_c.columns:
            # Determine region from column name
            if 'utr5' in col:
                trim_len = utr5_len
                # UTR5: take last N elements (right side)
                trim_func = lambda arr: arr[-int(trim_len):] if trim_len!=0 else []
            elif 'utr3' in col:
                trim_len = utr3_len
                # UTR3: take first N elements (left side)
                trim_func = lambda arr: arr[:int(trim_len)] if trim_len!=0 else []
            else:
                continue  # skip if not a recognized region
            
            # Apply trimming to each array in the nested structure
            original_value = y_c.loc[gene, col]
            if len(original_value) > 0:
                trimmed_arrays = [trim_func(sub_array) for sub_array in original_value]
                y_c.loc[gene, col] = trimmed_arrays
    
    return y_c

def combine_counts(y_c, condition, maxlen):
    '''
    Combine UTR and CDS counts for a specific condition.

    Parameters
    ----------
    y_c : pd.DataFrame
        Output labels with separate UTR5, CDS, and UTR3 data.
    condition : str
        Condition name to extract.
    maxlen : int
        Maximum sequence length for padding.

    Returns
    -------
    np.ndarray
        Combined and padded array of counts across all replicates.
    '''
    utr5 = y_c['utr5'][condition]
    cds = y_c['cds'][condition]
    utr3 = y_c['utr3'][condition]
    replicates = []
    for replicate in range(len(cds)):
        combined = pad_array(np.concatenate([utr5[replicate],cds[replicate],utr3[replicate]]), maxlen)
        replicates.append(combined)
    return np.array(replicates)

def pad_array(array, length):
    '''
    Pad array to a fixed length with NaN values.

    Parameters
    ----------
    array : np.ndarray
        Array to pad.
    length : int
        Target length after padding.

    Returns
    -------
    np.ndarray
        Padded array.
    '''
    # Ensure padding to fixed length with nan
    if len(array) < length:
        array = np.concatenate([array, np.full(length - len(array), np.nan)])
    return array

def create_annotation_array(utr5len, cdslen, utr3len, maxlen):
    '''
    Create annotation array marking different genomic regions.

    Parameters
    ----------
    utr5len : int
        Length of 5' UTR.
    cdslen : int
        Length of CDS (must be multiple of 3).
    utr3len : int
        Length of 3' UTR.
    maxlen : int
        Maximum total length for padding.

    Returns
    -------
    np.ndarray
        Array with values: 5 for 5'UTR, 1-2-3 pattern for CDS, 4 for 3'UTR, 0 for padding.

    Raises
    ------
    ValueError
        If cdslen is not a multiple of 3.
    '''
    if cdslen % 3 != 0:
        raise ValueError(f"cds length must be a multiple of 3. Found length {cdslen} for cds.")
    # 0s for utr5
    utr5_pattern = np.full(utr5len, 5, dtype=int)
    #123 pattern for cds
    cds_pattern = np.tile([1, 2, 3], cdslen // 3)
    # 4s for utr3
    utr3_pattern = np.full(utr3len, 4, dtype=int)
    pad_lenght = maxlen - (utr5len + cdslen + utr3len)
    padding = np.zeros(pad_lenght, dtype=int)
    
    return np.concatenate([utr5_pattern, cds_pattern, utr3_pattern, padding])

def kmers_stride1(seq, k=6):
    '''
    Split a sequence into overlapping k-mers with stride 1.

    Parameters
    ----------
    seq : str
        Input nucleotide sequence.
    k : int, default=6
        Length of each k-mer.

    Returns
    -------
    list of str
        List of overlapping k-mers.
    '''
    # splits a sequence into overlapping k-mers
    return [seq[i:i + k] for i in range(0, len(seq)-k+1)]

def reshape_training_data(x_input, y_output, seq_resolution, lm=None, utrpool=1.0, seq_args=None):
    '''
    Reshape training data into arrays suitable for RiboNet model training.
    
    Parameters
    ----------
    x_input : pd.DataFrame
        Input features for RiboNet generated in preprocessing.
    y_output : pd.DataFrame
        Output labels for RiboNet generated in preprocessing.
    seq_resolution : str
        Sequence resolution, either "codon" or "nucleotide".
    lm : str or dict, default = None
        If None, no language model is used and sequences are one-hot encoded and processed by a residual convolutional network.
        If seq_resolution is "nucleotide", lm should be a list with 4 items: name of model, tokenizer, model, species_proxy (if any).
        If seq_resolution is "codon", lm should be a dict with keys "UTR5" and "UTR3" each a list with the 4 items.
    utrpool : float, default = 1.0
        Reduce the UTR lengths by pooling (summing counts, average pooling embeddings) by this factor.
        Ignored if seq_resolution is "nucleotide".
    seq_args : dict, default = None
        Relevant arguments used for reshaping. If provided, will override the calculated values.
    
    Returns
    -------
    x_c : pd.DataFrame
        Reshaped input features for RiboNet.
    ymat : np.ndarray
        Reshaped output labels for RiboNet.
    seq_args : dict
        Relevant arguments used for reshaping.
    '''
    # TODO: remove utrs in calculating tot_reads
    # provide utr_size to loss function
    # flexible windows embeddings (window_size, overlap)
    # for splecieslm makes more sense to trucate afterward before padding? test it
    if seq_resolution=="codon":
        if not seq_args:
            maxcodonlen = int(x_input["cds_len"].max()/3)
            maxutr5len = int(x_input["utr5_len"].max())
            maxutr3len = int(x_input["utr3_len"].max())
        else:
            maxcodonlen = seq_args["maxcodonlen"]
            maxutr5len = seq_args["maxutr5len"]
            maxutr3len = seq_args["maxutr3len"]
        maxkmer5len = maxutr5len-2 if lm is not None and lm["UTR5"] is not None and (lm["UTR5"][0]=="specieslm5" or lm["UTR5"][0]=="specieslm3") else maxutr5len+3
        maxkmer3len = maxutr3len-2 if lm is not None and lm["UTR3"] is not None and (lm["UTR3"][0]=="specieslm5" or lm["UTR3"][0]=="specieslm3") else maxutr3len+3
        if not seq_args:
            utr5poolsize = min(int(maxutr5len*utrpool), maxkmer5len) # account for kmerization (-5 + ATG)
            utr3poolsize = min(int(maxutr3len*utrpool), maxkmer3len) # account for kmerization (-5 + stop codon)
        else:
            utr5poolsize = seq_args["utr5poolsize"]
            utr3poolsize = seq_args["utr3poolsize"]
        x_c = pd.DataFrame(index=x_input.index, columns=["utr5", "cds", "utr3", "rna", "posemb"])
        for gene, df in x_input.iterrows():
            cdsseq = df.loc["cds"]
            utr5seq = df.loc["utr5"][-int(df.loc["utr5_len"]+3):]
            utr3seq = df.loc["utr3"][:int(df.loc["utr3_len"]+3)]
            cdslen = int(df.loc["cds_len"]/3)
            # CDS
            x_c.loc[gene, "cds"] = [Clist.index(cdsseq[s:s+3]) for s in range(0,len(cdsseq),3)] + [Clist.index("XXX")]*int(maxcodonlen - cdslen)
            # UTR5
            if lm is None or lm["UTR5"] is None:
                x_c.loc[gene, "utr5"] = [Nlist.index("X")]*int(maxutr5len - df.loc["utr5_len"]) + [Nlist.index(s) for s in list(utr5seq)]
            else:
                utr5seq = utr5seq.lstrip("X")  # remove X padding; tokenizer will re-pad
                if lm["UTR5"][0]=="specieslm5" or lm["UTR5"][0]=="specieslm3":
                    utr5seq = kmers_stride1(utr5seq) if len(utr5seq) >= 6 else []
                x_c.loc[gene, "utr5"] = tokenize(list(utr5seq), lm["UTR5"][1], maxkmer5len, padsite="left", proxy_species=lm["UTR5"][3])
            # UTR3
            if lm is None or lm["UTR3"] is None:
                x_c.loc[gene, "utr3"] = [Nlist.index(s) for s in list(utr3seq)] + [Nlist.index("X")]*int(maxutr3len - df.loc["utr3_len"])
            else:
                utr3seq = utr3seq.rstrip("X")  # remove X padding; tokenizer will re-pad
                if lm["UTR3"][0]=="specieslm5" or lm["UTR3"][0]=="specieslm3":
                    utr3seq = kmers_stride1(utr3seq) if len(utr3seq) >= 6 else []
                x_c.loc[gene, "utr3"] = tokenize(list(utr3seq), lm["UTR3"][1], maxkmer3len, proxy_species=lm["UTR3"][3])
            # RNA
            x_c.loc[gene, "rna"] = df.loc["rna"]
            # Positional embedding. TODO: maybe change utr5/3poolsize as they can be smaller than maxutr5/3len 
            x_c.loc[gene,"posemb"] = np.concatenate([np.zeros(utr5poolsize-int(df.loc["utr5_len"]*utrpool)),
                                                     np.arange(1,int(df.loc["utr5_len"]*utrpool)+cdslen+1),
                                                     np.zeros(maxcodonlen-cdslen),
                                                     np.arange(int(df.loc["utr5_len"]*utrpool)+cdslen+1, int(df.loc["utr5_len"]*utrpool)+cdslen+int(df.loc["utr3_len"]*utrpool)+1),
                                                     np.zeros(utr3poolsize-int(df.loc["utr3_len"]*utrpool))], axis=0)

        # Reformat y to numpy matrix
        utr5pool = nn.AdaptiveAvgPool1d(utr5poolsize)
        utr3pool = nn.AdaptiveAvgPool1d(utr3poolsize)
        ymat = np.full([y_output.shape[0], y_output["cds"].shape[1], (maxcodonlen+utr5poolsize+utr3poolsize), max([s.shape[0] for s in y_output.iloc[0].values])], np.nan) # genes, conds, seqlen, reps
        for n, cond in enumerate(y_output["cds"].columns):
            nrep = y_output["cds"].iloc[0,n].shape[0]
            conddf = y_output.xs(cond, level=1, axis=1)
            cdstemp = np.stack([[pad_array(row.loc["cds"][s].astype(np.float32), maxcodonlen*3)  for s in range(row.iloc[0].shape[0])] for _,row in conddf.iterrows()])
            cdstemp = np.add.reduceat(cdstemp, np.arange(0,cdstemp.shape[2],3), axis=2)
            utr5temp = utr5pool(torch.tensor(np.stack([[row.loc["utr5"][s][-maxutr5len:].astype(np.float32) for s in range(row.iloc[0].shape[0])] for _,row in conddf.iterrows()]), dtype = torch.float32))*(maxutr5len/utr5poolsize)
            utr3temp = utr3pool(torch.tensor(np.stack([[row.loc["utr3"][s][:maxutr3len].astype(np.float32) for s in range(row.iloc[0].shape[0])] for _,row in conddf.iterrows()]), dtype = torch.float32))*(maxutr3len/utr3poolsize)
            ymat[:,n,:,:nrep] = np.transpose(np.concatenate([utr5temp,cdstemp,utr3temp], axis=2), (0,2,1))
        
        seq_args = {
            "maxcodonlen": maxcodonlen,
            "maxutr5len": maxutr5len,
            "maxutr3len": maxutr3len,
            "utr5poolsize": utr5poolsize,
            "utr3poolsize": utr3poolsize,
            "maxkmer5len": maxkmer5len,
            "maxkmer3len": maxkmer3len
        }

    elif seq_resolution=="nucleotide":
        if not seq_args:
            maxlen = int(x_input[["utr5_len","cds_len","utr3_len"]].sum(axis=1).max())
        else:
            maxlen = seq_args["maxseqlen"]
        maxkmerlen = maxlen-5 if lm is not None and (lm[0]=="specieslm5" or lm[0]=="specieslm3") else maxlen
        x_c = pd.DataFrame(index=x_input.index, columns=["seq", "rna", "annotation"])
        for gene, df in x_input.iterrows():
            seqlen = int(df.loc["utr5_len"] + df.loc["cds_len"] + df.loc["utr3_len"])
            fullseq = df.loc["utr5"][-int(df.loc["utr5_len"]+3):-3] + df.loc["cds"] + df.loc["utr3"][3:int(df.loc["utr3_len"]+3)]
            # Sequence
            if lm is None:
                x_c.loc[gene, "seq"] = [Nlist.index(s) for s in list(fullseq)] + [Nlist.index("X")]*int(maxlen - seqlen)
            else:
                if lm[0]=="specieslm5" or lm[0]=="specieslm3":
                    fullseq = kmers_stride1(fullseq)
                x_c.loc[gene, "seq"] = tokenize(list(fullseq), lm[1], maxkmerlen,  proxy_species=lm[3])
            # RNA
            x_c.loc[gene, "rna"] = df.loc["rna"]
            # Annotation
            utr5kmerlen = int(df.loc["utr5_len"])-2 if lm is not None and (lm[0]=="specieslm5" or lm[0]=="specieslm3") else int(df.loc["utr5_len"])
            cdskmerlen = int(df.loc["cds_len"])-1 if lm is not None and (lm[0]=="specieslm5" or lm[0]=="specieslm3") else int(df.loc["cds_len"])
            utr3kmerlen = int(df.loc["utr3_len"])-2 if lm is not None and (lm[0]=="specieslm5" or lm[0]=="specieslm3") else int(df.loc["utr3_len"])
            x_c.loc[gene, "annotation"] = create_annotation_array(utr5kmerlen, cdskmerlen, utr3kmerlen, maxkmerlen)
        
        # trim the profiling to the corresponding utr lengths
        y_c = trim_y_c_arrays(y_output.copy(), x_input)
        # combine the sequences utr5, cds, utr3
        for condition in y_output["cds"].columns: 
            y_c.loc[:, ("combined", condition)] = y_c.apply(combine_counts,args=(condition,maxlen), axis=1)
        y_c = y_c["combined"]
        # Reformat y to numpy matrix
        seqpool = nn.AdaptiveAvgPool1d(maxkmerlen)
        ymat = np.full([y_c.shape[0], y_c.shape[1], maxkmerlen, max([s.shape[0] for s in y_c.iloc[0].values])], np.nan) # genes, conds, seqlen, reps
        for n in range(y_c.shape[1]):
            nrep = y_c.iloc[0,n].shape[0]
            seqtemp = seqpool(torch.tensor(np.stack([[s.astype(np.float32) for s in m] for m in y_c.iloc[:,n].values]), dtype = torch.float32))*(maxlen/maxkmerlen)
            ymat[:,n,:,:nrep] = np.transpose(seqtemp, (0,2,1))
        
        maxcodonlen = int(x_input["cds_len"].max()/3)
        maxutr5len = int(x_input["utr5_len"].max())
        maxutr3len = int(x_input["utr3_len"].max())
        
        seq_args = {
            "maxseqlen": maxlen,
            "maxkmerlen": maxkmerlen,
            "maxutr5len": maxutr5len,
            "maxutr3len": maxutr3len,
            "maxcodonlen": maxcodonlen
        }
    
    else:
        raise ValueError("seq_resolution must be either 'codon' or 'nucleotide'")
    
    
    return x_c, ymat, seq_args

def count_library_reads(y_c, sample_file):
    '''
    Count total reads per library across all genes.

    Parameters
    ----------
    y_c : pd.DataFrame
        Output labels containing ribosome profiling counts.
    sample_file : pd.DataFrame
        Sample metadata with track and sample_id columns.

    Returns
    -------
    np.ndarray
        Array of total reads per library, shape (n_conditions, max_replicates).
    '''
    # Get total reads per library
    tot_reads = y_c["cds"].map(lambda x: [np.nansum(s) for s in x]).apply(lambda x: np.nansum(np.stack(x.values), axis=0)).T
    if type(tot_reads) == pd.DataFrame:
        tot_reads = pd.Series({r: tot_reads.loc[r].values for r in tot_reads.index})
    reps = pd.Series({c:sorted(sample_file.loc[sample_file['track']==c,'sample_id'].to_list()) for c in tot_reads.index})
    tot_reads_arr = np.full([tot_reads.shape[0], np.max([tot_reads.apply(len)])], np.nan)
    for i, c in enumerate(tot_reads.index):
        for j, r in enumerate(reps.loc[c]):
            agg_samples = sample_file.loc[np.array([s==sample_file.loc[sample_file["sample_id"]==r,"bam"] for s in sample_file.bam]),"sample_id"].values
            tot_reads_arr[i,j] = np.sum([r for reads,samples in zip(tot_reads,reps) for r,s in zip(reads,samples) if s in agg_samples])
    return tot_reads_arr

def print_seq(xcgene, regsizes, chunksize = 100):
    '''
    Print gene sequence with annotations in a formatted manner.

    Parameters
    ----------
    xcgene : pd.Series
        Gene data containing utr5, cds, and utr3 sequences.
    regsizes : list of int
        List containing [utr5_len, cds_len, utr3_len].
    chunksize : int, default=100
        Number of nucleotides to display per line.

    Returns
    -------
    str
        Complete gene sequence (UTR5 + CDS + UTR3).
    '''
    seqcds = xcgene.loc["cds"]
    sequtr5 = xcgene.loc["utr5"][int(-(regsizes[0]+3)):-3]
    sequtr3 = xcgene.loc["utr3"][3:int(regsizes[2]+3)]
    seq = "".join([sequtr5,seqcds,sequtr3])
    splitseq = []
    counter = 0
    for n in range(0,len(sequtr5),chunksize):
        splitseq.append(str("%i " % counter) + sequtr5[n:n+chunksize])
        counter += len(sequtr5[n:n+chunksize])
    codonchunksize = chunksize-(chunksize%3)
    for n in range(0,len(seqcds),codonchunksize):
        subseq = seqcds[n:n+codonchunksize]
        codonseq = " ".join([subseq[s:s+3] for s in range(0,len(subseq),3)])
        splitseq.append(str("%i " % counter) + codonseq)
        counter += len(subseq)
    for n in range(0,len(sequtr3),chunksize):
        splitseq.append(str("%i " % counter) + sequtr3[n:n+chunksize])
        counter += len(sequtr3[n:n+chunksize])
    print(*splitseq, sep=os.linesep)
    return seq

def count_reads_around_cds_starts(
    bam_path: str,
    cds_info: dict,
    lengths: list[int],
    window: int = 40,
    min_mapq: int = 0,
):
    """
    Count 5' read starts around CDS start sites using pysam.
    Parameters
    ----------
    bam_path : str
        Path to the BAM file to analyze.
    cds_info : dict
        Dictionary of CDS start information parsed from the GFF file, keyed by (seqid, gene_id) with values containing 'cds_start' and 'strand'.
    lengths : list[int]
        List of read lengths to include in the analysis.
    window : int, optional
        +/- window in nt around CDS start to count reads (default 40).
    min_mapq : int, optional
        Minimum mapping quality to include a read in the counts (default 0, i.e. include all mapped reads).
    
    Returns:
    --------
        dict[length] -> Counter(relative_position -> count)
    """
    counts_by_len = {L: Counter() for L in lengths}
    lengths_set = set(lengths)  # Use set for O(1) lookup

    bam = pysam.AlignmentFile(bam_path, "rb")

    # Group CDS starts by chromosome for efficient fetching
    cds_by_chrom = defaultdict(list)
    for (_, _), info in cds_info.items():
        cds_by_chrom[info["seqid"]].append(info)

    # Process each chromosome
    for chrom, cds_list in cds_by_chrom.items():
        if chrom not in bam.references or not cds_list:
            continue

        # Calculate the range of CDS positions in this chromosome
        cds_positions = {info["cds_start"]: info for info in cds_list}
        min_cds = min(cds_positions.keys())
        max_cds = max(cds_positions.keys())
        
        # Fetch all reads in the region at once for efficiency
        fetch_start = max(0, min_cds - window)
        fetch_end = max_cds + window + 1

        for read in bam.fetch(chrom, fetch_start, fetch_end):
            # Combine conditions for faster filtering
            if read.is_unmapped or read.mapping_quality < min_mapq:
                continue

            read_len = (
                read.query_length
                if read.query_length is not None
                else (read.infer_read_length() or 0)
            )

            if read_len not in lengths_set:
                continue

            # 5' genomic coordinate (0-based)
            fivep = read.reference_start if not read.is_reverse else read.reference_end - 1

            # Check this read against all CDS starts in the chromosome
            for cds_start, info in cds_positions.items():
                strand = info["strand"]
                
                # Relative position in transcript orientation
                rel = (fivep - cds_start) if strand == "+" else (cds_start - fivep)

                if -window <= rel <= window:
                    counts_by_len[read_len][rel] += 1

    bam.close()
    return counts_by_len