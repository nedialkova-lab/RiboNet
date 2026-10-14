import pandas as pd
import numpy as np
import os
import shutil
from .aip import run_offset, run_profile
from functools import partial
from itertools import repeat
from .utils.seq_utils import *
from .utils.plot_utils import generate_offset_figure, plot_length_distribution, color_map
from .utils import report_utils as _ru
import seaborn as sns
from .utils.annotation_utils import AnnotationDB
from .utils.parallel import parallel_map
import logging
from pathlib import Path
import pysam
import matplotlib.pyplot as plt

log = logging.getLogger(__name__)

class RiboPreprocess:
    def __init__(self, sample_file, genome_fasta_file, gff_file, project_dir, spikein=None):
        '''
        Initialize RiboPreprocess class for preprocessing ribosome profiling data.

        Parameters
        ----------
        sample_file : str
            Path to tab-separated sample file containing sample metadata.
            Must include columns: sample_id, condition, readlength, bam.
        genome_fasta_file : str
            Path to genome FASTA file.
        gff_file : str
            Path to GFF/GTF annotation file.
        project_dir : str
            Path to project directory where outputs will be saved.
        spikein : str or None, default = None
            Comma-separated list of spike-in gene names or path to text file with one gene name per line.
            Used in RNA abundance extraction for normalization. If not set, no spike-in normalization will be performed.

        Returns
        -------
        None
        '''
        # Validate input files exist
        if not os.path.isfile(sample_file):
            err_message = f"Sample file not found: {sample_file}"
            log.error(err_message)
            raise FileNotFoundError(err_message)
        if not os.path.isfile(genome_fasta_file):
            err_message = f"Genome FASTA file not found: {genome_fasta_file}"
            log.error(err_message)
            raise FileNotFoundError(err_message)
        if not os.path.isfile(gff_file):
            err_message = f"Annotation file not found: {gff_file}"
            log.error(err_message)
            raise FileNotFoundError(err_message)

        try:
            self.sample_file = pd.read_csv(sample_file, sep="\t")
        except Exception as e:
            err_message = f"Could not read sample file '{sample_file}': {e}. Ensure it is a valid tab-separated file."
            log.error(err_message)
            raise ValueError(err_message) from e

        # Validate required columns
        required_cols = {"sample_id", "condition", "readlength", "bam"}
        missing_cols = required_cols - set(self.sample_file.columns)
        if missing_cols:
            err_message = (f"Sample file '{sample_file}' is missing required columns: {sorted(missing_cols)}. "
                           f"Required columns are: {sorted(required_cols)}.")
            log.error(err_message)
            raise ValueError(err_message)

        # Validate that BAM files exist
        missing_bams = [b for b in self.sample_file["bam"] if not os.path.isfile(b)]
        if missing_bams:
            err_message = (f"The following BAM file(s) listed in the sample file were not found: "
                           f"{', '.join(missing_bams)}")
            log.error(err_message)
            raise FileNotFoundError(err_message)

        # Validate readlength format (expected: 'min-max', e.g. '27-30')
        for rl in self.sample_file["readlength"]:
            parts = str(rl).split("-")
            if len(parts) != 2 or not all(p.strip().isdigit() for p in parts):
                err_message = (f"Invalid readlength format '{rl}' in sample file. "
                               f"Expected format: 'min-max' (e.g. '27-30').")
                log.error(err_message)
                raise ValueError(err_message)

        for bam in self.sample_file["bam"]:
            bam = Path(bam)
            if not any(bam.with_suffix(bam.suffix + ext).exists() for ext in (".bai", ".csi")):
                pysam.index(str(bam))

        self.genome_fasta_file = genome_fasta_file
        self.gff_file = gff_file
        self.annot_db = AnnotationDB(gff_file)
        self.spikein = spikein
        if self.spikein is not None:
            if os.path.isfile(self.spikein):
                try:
                    genelist = []
                    with open(self.spikein, 'r') as f:
                        for lines in f:
                            fields = lines.strip().split('\t')
                            gene = fields[0]
                            genelist.append(gene)
                    self.spikein = genelist
                except Exception as e:
                    err_message = f"Could not read spike-in file '{self.spikein}': {e}. Ensure it is a valid text file with one gene name per line."
                    log.error(err_message)
                    raise ValueError(err_message) from e
            else:
                self.spikein = self.spikein.split(",")
        self.project_dir = project_dir
        self.preprocess_dir = os.path.join(self.project_dir, "preprocessing")
        self.riboAdir = os.path.join(self.project_dir, "riboA")
        self.qc_plots = os.path.join(self.riboAdir, "qc_plots")

        os.makedirs(self.project_dir, exist_ok=True)
        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.riboAdir, exist_ok=True)
        os.makedirs(self.qc_plots, exist_ok=True)
        
        self.sample_file["track"] = [i+"_"+j for i,j in zip(self.sample_file["condition"], self.sample_file["readlength"])]
        self.sample_file.to_csv(os.path.join(self.preprocess_dir, "sample_file.txt"), sep="\t", index=False)

    def _compute_offset_single_sample(self, bam, offset_window, cds_info):
        """
        Helper method to compute offset for a single sample in parallel.
        
        Parameters
        ----------
        sample_row_tuple : tuple
            Tuple of (index, row) from sample_file DataFrame.
            
        Returns
        -------
        tuple
            (sample_id, bam, condition, readlength, frag_min, frag_max, counts_by_len).
        """

        frags = self.sample_file.loc[self.sample_file["bam"] == bam, "readlength"].tolist()
        readlengths = []
        for f in frags:
            f = f.split("-")
            readlengths.extend(list(range(int(f[0]), int(f[1])+1)))
        readlengths = sorted(set(readlengths))
        
        counts_by_len = count_reads_around_cds_starts(
            bam_path=bam,
            cds_info=cds_info,
            lengths=readlengths,
            window=offset_window,
        )
        
        return (bam, readlengths, counts_by_len)

    @staticmethod
    def _get_lengths_from_bam(bam: str) -> pd.DataFrame:
        """
        Extract mapped read lengths from a single BAM file.

        Parameters
        ----------
        bam : str
            Path to BAM file.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns 'Length' and 'bam' (stem of the BAM filename).
        """
        with pysam.AlignmentFile(bam, "rb") as bam_file:
            lengths = [
                read.query_length
                for read in bam_file.fetch(until_eof=True)
                if not read.is_unmapped
            ]
        return pd.DataFrame({"Length": lengths, "bam": Path(bam).stem})

    def generate_offset_report(
        self, filename: str = "QC_report.pdf",
        window: int = 40,
        threads: int = 1,
        finalize: bool = True,
    ):
        """
        Generate PDF report with offset plots using RiboA parsing functions.

        Parameters
        ----------
        filename : str
            Output PDF filename to create.
        window : int, optional
            +/- window in nt to aggregate and plot (default 40).
        temp_root : str, optional
            If provided, use this directory to store temporary SAM files. Otherwise use system tmpdir.
        threads : int, optional
            Number of parallel workers for BAM length extraction (default 1).

        Returns
        -------
        None
        """
        reportpath = os.path.join(self.riboAdir, filename)
        if not os.path.isfile(reportpath):
            cds_info = self.annot_db.parse_cds_starts()
            if not cds_info:
                err_message = (f"No CDS entries found in annotation file '{self.gff_file}'. "
                            f"Ensure the annotation file is a valid GFF/GTF with CDS features.")
                log.error(err_message)
                raise RuntimeError(err_message)

            # Initialize PDF report
            log.info("Generating QC report with read length and offset analyses.")
            _ru.create_report(reportpath, title="QC Report")
            _ru.add_text("This report presents quality control analyses of the ribosome profiling data. It contains:")
            _ru.add_text("1. Distribution of footprint lengths per sample to assess the expected size range of ribosome-protected fragments. "
                    "A strong peak in the expected size range (typically 27-30 nt for long footprints, 20-22 nt for short footprints) indicates good enrichment of ribosome footprints.")
            _ru.add_text("2. Offset analysis of 5' read starts around CDS start sites to evaluate the periodicity and P-site offset of ribosome footprints. "
                    "A clear 3-nt periodic pattern with a strong peak at the expected offset (typically -12 nt for 28 nt footprints) supports accurate mapping of ribosome positions and good data quality.")
            _ru.add_page_break()

            # Footprint Size Distribution Analysis
            _ru.add_heading("Footprint Length Distribution")

            unique_bams = self.sample_file["bam"].unique().tolist()
            log.info("Extracting read lengths from %d BAM file(s) for length distribution plot.", len(unique_bams))
            all_lengths = parallel_map(self._get_lengths_from_bam, unique_bams, threads=threads)

            # Combine all BAMs into one DataFrame
            df = pd.concat(all_lengths, ignore_index=True)
            df.to_csv(os.path.join(self.qc_plots, "length_distribution.csv"), index=False)

            # Generate the figure
            fig = plot_length_distribution(df)

            # Save and register the figure
            plt.savefig(os.path.join(self.qc_plots, "length_distribution.pdf"), format='pdf', bbox_inches='tight')
            _ru.add_plot(fig)
            plt.close(fig)
            _ru.add_page_break()

            # Offset Analysis
            _ru.add_heading("Metagene 5' Offset at CDS Start")
            log.info("Performing offset QC analysis. This may take some time depending on the number and size of BAM files.")
            
            # Process samples in parallel for efficiency
            cds_info = self.annot_db.parse_cds_starts()
            log.info("Processing %d BAM file(s) using %d thread(s).", len(unique_bams), min(threads, len(unique_bams)))
            frozen_fun = partial(self._compute_offset_single_sample, offset_window=window, cds_info=cds_info)
            results = parallel_map(frozen_fun, unique_bams, threads=threads, desc="Computing offsets")

            # Process results and generate plots
            for bam, readlengths, counts_by_len in results:
                _ru.add_text(f"<b>BAM file:</b> {bam}")
                _ru.add_text(f"<b>Read lengths:</b> {', '.join(map(str, readlengths))}")

                fig = generate_offset_figure(counts_by_len, readlengths, window) ####

                _ru.add_plot(fig)

                plt.savefig(
                    os.path.join(self.qc_plots, f"{os.path.basename(bam)}_offset_plot.pdf"),
                    format="pdf",
                    bbox_inches="tight",
                )

                plt.close(fig)

            if finalize:
                _ru.finalize_report()
                log.info("QC report generated and saved to %s", reportpath)
            else:
                log.info("QC report content built; awaiting finalization.")
        else:
            log.info("QC report already exists at %s. Skipping generation.", reportpath)

    def _riboA_1file(self, file, annot_table, three_prime, overlap, minRD,
                     threshold_gene_pct, threshold_start_codon, genesubset, include,
                     alignment_type, get_profile, site, omit_nt_calibrate,
                     fixed_5offsets):
        '''
        Run RiboA offset calculation and P-site profiling for a single BAM file.

        Parameters
        ----------
        file : tuple
            Tuple containing (index, row) from sample_file DataFrame.
        annot_table : str
            Path to annotation table file.
        three_prime : bool
            Whether to quantify reads from 3' end.
        overlap : int
            Number of nucleotides beyond CDS to avoid overlap.
        minRD : float
            Minimum reads per codon threshold for offset computation.
        threshold_gene_pct : float
            Minimum percentage of genes for unique offset assignment.
        threshold_start_codon : float
            Minimum ratio for start codon filtering.
        genesubset : str or None
            Path to file with gene subset.
        include : bool
            Whether to include or exclude genesubset.
        alignment_type : str
            Alignment type ("genome" or "transcriptome").
        get_profile : bool
            Whether to compute site counts after offset determination.
        site : str
            Site type ("A", "P", or "E").
        omit_nt_calibrate : bool
            Whether to skip nucleotide calibration.
        fixed_5offsets: dict or None
            Specify transcript offsets instead of performing estimation.

        Returns
        -------
        None
            Saves RiboA results to the riboA directory.
        '''
        outdir = os.path.join(self.riboAdir, file[1].loc["sample_id"])
        log.info(f"Processing sample '{file[1].loc['sample_id']}' with BAM '{file[1].loc['bam']}' for RiboA analysis.")
        if os.path.exists(os.path.join(outdir,"scratch")):
            shutil.rmtree(os.path.join(outdir))
        if not os.path.exists(outdir):
            os.makedirs(outdir)
            bamfile = file[1].loc["bam"]
            min_frag, max_frag = file[1].loc["readlength"].split("-")
            min_frag, max_frag = int(min_frag), int(max_frag)

            if fixed_5offsets is None:
                offsets_dict = run_offset(outdir, bamfile, annot_table, self.genome_fasta_file, min_frag, max_frag, three_prime, overlap, minRD,
                                          threshold_gene_pct, threshold_start_codon, genesubset, include, alignment_type, get_profile, self.extend_prof, site,
                                          omit_nt_calibrate)
                offset_parts = ", ".join(f"{rl} nt (F0 {frames[0]}, F1 {frames[1]}, F2 {frames[2]})" for rl, frames in offsets_dict.items())
                log.info("A-offsets for %s: %s", file[1].loc["sample_id"], offset_parts)
            elif all(length in fixed_5offsets.keys() for length in range(min_frag, max_frag+1)):
                offset_dict = dict()
                for fsize, off in fixed_5offsets.items():
                    offset_dict[fsize] = {
                        0: {"off": off, "perc": "NA"},
                        1: {"off": off, "perc": "NA"},
                        2: {"off": off, "perc": "NA"}}
                run_profile(outdir, bamfile, annot_table, self.genome_fasta_file, offset_dict, min_frag, max_frag, three_prime,
                            alignment_type, self.extend_prof, site)
            else:
                message = f"Fixed offsets provided does not match the read lengths in sample file ({max_frag}-{min_frag})."
                log.error(message)
                raise ValueError(message)
        else:
            log.info(f"RiboA output already exists for sample '{file[1].loc['sample_id']}' at {outdir}. Skipping RiboA computation for this sample.")

    def compute_riboA(self, threads=1, three_prime=False, overlap=0, minRD=0.5,
                      threshold_gene_pct=50.0, threshold_start_codon=5.0, genesubset=None,
                      include=False, alignment_type="genome", get_profile=True, extend_prof=2001,
                      site="P", omit_nt_calibrate=False, fixed_5offsets=None):
        '''
        Runs RiboA to define the P site coverage of aligned reads.

        Parameters
        ----------
        threads : int, default = 1
            Number of threads to use for multithreading.
        three_prime : bool, default = False
            Indicator whether to quantify reads from 3' end. If True, quantify reads from 3' end.
        overlap : int, default = 0
            Number of nucleotides beyond the CDS region of a gene which are to be avoided to overlap with another gene.
        minRD : float, default = 0.5
            Compute offset with genes which have greater than threshold reads per codon on average.
        threshold_gene_pct : float, default = 50.0
            Minimum percentage of genes with the most probable offset for assigning a unique offset.
        threshold_start_codon : float, default = 5.0
            Minimum ratio between the average reads in the second, third and fourth codon and the reads in the first codon.
        genesubset : str, default = None
            If the IP algorithm has to be applied for a selected subset of genes, they must be loaded from a text file here. If include option is false, these genes will be excluded.
        include : bool, default = False
            If include option is true, offset will be computed for genesubset only. If false, offset will be computed for all genes except those in genesubset.
        alignment_type : str, default = "genome"
            Alignment type, either genome or transcriptome.
        get_profile : bool, default = True
            Compute site counts after determining the offset.
        extend_prof : int, default = 2001
            If different than 0, nucleotides to extend before and after CDS in the nucleotide profile. It should be multiple of 3.
        site : str, default = "P"
            Get coverage at A, P, or E site.
        omit_nt_calibrate : bool, default = False
            Calibrate nucleotide resolution so that frame0 > frame1,frame2.
        fixed_5offsets: dict or None, default = None
            Specify transcript offsets instead of performing estimation.

        Returns
        -------
        Saved riboA results

        '''
        self.extend_prof = extend_prof
        log.info("Start running riboA")
        if self.spikein is not None and genesubset is None:
            log.info("Excluding spike-in genes for offset calculation: %s", ", ".join(self.spikein))
            genesubset = self.spikein
        # Generate annot table
        tab_file = os.path.join(self.riboAdir, "cds_info.tab")
        self.annot_db.write_aip_tab(tab_file)
        # Run riboA
        frozen_fun = partial(self._riboA_1file, annot_table=tab_file, three_prime=three_prime, overlap=overlap, minRD=minRD,
                             threshold_gene_pct=threshold_gene_pct, threshold_start_codon=threshold_start_codon, genesubset=genesubset,
                             include=include, alignment_type=alignment_type, get_profile=get_profile,
                             site=site, omit_nt_calibrate=omit_nt_calibrate, fixed_5offsets=fixed_5offsets)
        parallel_map(frozen_fun, self.sample_file.iterrows(), threads=threads)
        os.remove(tab_file)

        log.info("Generated riboA results and saved them to %s", self.riboAdir)

    def count_RPF(self):
        '''
        Count P sites per gene per sample from RiboA output.

        Returns
        -------
        self
        '''

        if len(os.listdir(self.riboAdir)) == 0:
            err_message = "Run compute_riboA() first, riboA directory is empty."
            log.error(err_message)
            raise RuntimeError(err_message)

        log.info("Counting RPFs")

        for idx, row in self.sample_file.iterrows():
            sample_id = row["sample_id"]
            bamfile = row["bam"]
            riboA_prof_file = os.path.join(self.riboAdir, sample_id, "P-site_profiles_nucleotide.tab")
            if not os.path.isfile(riboA_prof_file):
                err_message = (f"RiboA P-site profile file not found for sample '{sample_id}': {riboA_prof_file}. "
                               f"Ensure compute_riboA() completed successfully for this sample.")
                log.error(err_message)
                raise FileNotFoundError(err_message)
            prof_df = pd.read_csv(riboA_prof_file, sep="\t", names=["Geneid", "Length", "Psite_counts"], index_col=0)
            counts = [np.sum(np.array(s.split(","),dtype='i')) for s in prof_df["Psite_counts"]]
            if idx == 0:
                psitesdf = pd.DataFrame(0, index = prof_df.index, columns = ["Length"]+self.sample_file.bam.unique().tolist())
                psitesdf["Length"] = prof_df["Length"]
            psitesdf[bamfile] += counts
        self.rpf_counts = psitesdf
        savedir = os.path.join(self.riboAdir, "counts.rpf.txt")
        psitesdf.to_csv(savedir, sep="\t")

        log.info('RPF counts created and written to %s', savedir)

    def _compute_nf(self, RPFfc):
        '''
        Compute normalization factors based on spike-in controls for relative abundance.

        Parameters
        ----------
        RPFfc : pd.DataFrame
            DataFrame containing RPF (ribosome profiling) read counts.

        Returns
        -------
        pd.Series
            Normalization factors for each sample.
        '''
        spike_in_factor = RPFfc.loc[self.spikein, :].sum(axis=0) / RPFfc.sum(axis=0)
        counts_without_spike_in = RPFfc.loc[~RPFfc.index.isin(self.spikein)].sum(axis=0)
        norm_factors = spike_in_factor / counts_without_spike_in
        norm_factors = norm_factors / norm_factors.product() ** (1 / len(norm_factors))
        nf = pd.Series(norm_factors, index=RPFfc.columns)
        return nf

    def _compute_nf_abs(self, RNAtpm, RPFrpm, ribosomes):
        '''
        Compute normalization factors for absolute ribosome quantification using spike-ins.

        Parameters
        ----------
        RNAtpm : pd.DataFrame
            DataFrame containing RNA abundances in TPM (transcripts per million).
        RPFrpm : pd.DataFrame
            DataFrame containing RPF counts in RPM (reads per million).
        ribosomes : int or float
            Total number of ribosomes per transcript in spikeins (e.g. Fluc trisomes, 3, and Rluc disomes, 2, sum up to 5).

        Returns
        -------
        pd.Series
            Absolute normalization factors for each sample.
        '''
        nf = pd.Series(
                (ribosomes / np.sum([RPFrpm.loc[s].values / RNAtpm.loc[s, RPFrpm.columns].values for s in self.spikein],
                                    axis=0)),
                index=RPFrpm.columns)
        return nf

    @staticmethod
    def _avg_RNA_per_condition(RNAcounts):
        '''
        Average RNA counts across samples of the same condition.

        Parameters
        ----------
        RNAcounts : pd.DataFrame
            DataFrame containing RNA counts with gene identifiers as rows and sample IDs as columns.

        Returns
        -------
        pd.DataFrame
            DataFrame with averaged RNA counts per condition.
        '''
        # Create outdf with same index
        outdf = pd.DataFrame(index=RNAcounts.index, columns=np.unique(RNAcounts.columns))
        # Average RNA counts across samples of the same condition
        for cond in outdf.columns:
            frac = RNAcounts[[cond]].div(RNAcounts[[cond]].sum(axis=0), axis=1)
            outdf[cond] = frac.mean(axis=1)*1e6
        return outdf

    def extract_rna_abundance(self, rna_counts_file, spikein_absolute_ribosomes=None):
        '''
        normalizes the ribosome and RNA counts. If given, it takes the spikeins into account.

        Parameters
        ----------
        rna_counts_file : str
            Path to tab-separated file containing RNA counts with gene identifiers as rows and conditions as columns. Must include a 'Length' column for gene lengths.
        spikein_absolute_ribosomes : int or None
            If known, total number of ribosomes per transcript in spike-in genes (e.g. Fluc trisomes, 3, and Rluc disomes, 2, sum up to 5). Used for absolute normalization. If not set, relative normalization will be performed.

        Returns
        -------
        None

        '''

        if len(os.listdir(self.riboAdir)) == 0:
            err_message = "Run compute_riboA() first, riboA directory is empty."
            log.error(err_message)
            raise RuntimeError(err_message)

        if not hasattr(self, 'rpf_counts'):
            self.count_RPF()

        if not os.path.isfile(rna_counts_file):
            err_message = f"RNA counts file not found: {rna_counts_file}"
            log.error(err_message)
            raise FileNotFoundError(err_message)

        try:
            self.rna_counts = pd.read_csv(rna_counts_file, sep="\t", index_col=0)
        except Exception as e:
            err_message = f"Could not read RNA counts file '{rna_counts_file}': {e}. Ensure it is a valid tab-separated file."
            log.error(err_message)
            raise ValueError(err_message) from e
        RNA = self.rna_counts

        # Validate that RNA file has a 'Length' column
        if "Length" not in RNA.columns:
            err_message = (f"RNA counts file '{rna_counts_file}' is missing a 'Length' column. "
                           f"Ensure the file has a 'Length' column with gene lengths.")
            log.error(err_message)
            raise ValueError(err_message)

        # Validate that RNA file has a column matching each condition in the sample file
        missing_conds = [c for c in self.sample_file["condition"] if c not in RNA.columns]
        if missing_conds:
            err_message = (f"Condition(s) {sorted(set(missing_conds))} from the sample file were not found as columns in RNA counts file "
                           f"'{rna_counts_file}'. Ensure the RNA counts file has one column per condition.")
            log.error(err_message)
            raise ValueError(err_message)

        RNAuniq = self._avg_RNA_per_condition(RNA)
        RNA_all_samples = RNAuniq.loc[:, self.sample_file["condition"]]
        RNA_all_samples.columns = self.sample_file["sample_id"]
        RNAfc = pd.concat((RNA["Length"], RNA_all_samples), axis='columns')

        if self.spikein is not None:
            missing_spikeins = [s for s in self.spikein if s not in self.rna_counts.index]
            if missing_spikeins:
                err_message = (f"Spike-in gene(s) not found in RNA counts file: {', '.join(missing_spikeins)}. "
                               f"Check that the spike-in gene names match the gene identifiers in the RNA counts file.")
                log.error(err_message)
                raise ValueError(err_message)
            if spikein_absolute_ribosomes is None:
                log.info("Extract the RNA abundances and normalize relative abundance compared to spikins")

                RNAfc = RNAfc.loc[~RNAfc.index.isin(self.spikein)]
                RNAfc = RNAfc.iloc[:,1:].div(RNAfc["Length"], axis=0)
                kbnorm = RNAfc.sum(axis=0)/1e6
                RNAtpm = RNAfc.div(kbnorm, axis=1)
                RNAtpm[RNAtpm<1] = np.nan

                RPFfc = self.rpf_counts.loc[:,self.sample_file["bam"]]
                RPFfc.columns = self.sample_file["sample_id"]
                spikenf = self._compute_nf(RPFfc)
                RNAdata = RNAtpm.mul(spikenf.loc[RNAtpm.columns], axis=1)
            else:
                log.info("Extract the RNA abundances and normalize absolute ribosomes per transcript based on the spikins")

                RNAfc = RNAfc.iloc[:,1:].div(RNAfc["Length"], axis=0)
                kbnorm = RNAfc.sum(axis=0)/1e6
                RNAtpm = RNAfc.div(kbnorm, axis=1)
                RNAtpm[RNAtpm<1] = np.nan

                RPFfc = self.rpf_counts.loc[:,self.sample_file["bam"]]
                RPFfc.columns = self.sample_file["sample_id"]
                RPFrpm = RPFfc.div(RPFfc.sum(axis=0), axis=1)*1e6
                spikenf = self._compute_nf_abs(RNAtpm, RPFrpm, spikein_absolute_ribosomes)
                RNAdata = RNAtpm.div(spikenf.loc[RNAtpm.columns], axis=1)
        else:
            log.info("Extract the RNA abundances")

            RNAfc = RNAfc.iloc[:,1:].div(RNAfc["Length"], axis=0)
            kbnorm = RNAfc.sum(axis=0)/1e6
            RNAdata = RNAfc.div(kbnorm, axis=1)
            RNAdata[RNAdata<1] = np.nan

        self.norm_rna_counts = RNAdata

    def _merge_reps(self, samples):
        '''
        input
        sample_ids of replicates to merge

        output:
        mergedf: df with concatenated ribosome reads per nucleotide along each gene
        '''
        mergedf = None
        filepaths = [os.path.join(self.riboAdir, sample_id, "P-site_profiles_nucleotide_extend2001nt.tab") for sample_id in samples]
        for file in filepaths:
            log.info(f"Loading P-site occupancies file {file}")
            if not os.path.isfile(file):
                err_message = (f"P-site profile file not found: {file}. "
                               f"Ensure compute_riboA() completed successfully for all samples.")
                log.error(err_message)
                raise FileNotFoundError(err_message)
            if mergedf is None:
                mergedf = pd.read_csv(file, sep="\t", header=None, names=["gene", "length", "psites"], index_col=0)
                mergedf["psites"] = mergedf["psites"].apply(lambda x: np.array(x.split(","), dtype=float))
            else:
                tempdf = pd.read_csv(file, sep="\t", header=None, names=["gene", "length", "psites"], index_col=0)["psites"].apply(lambda x: np.array(x.split(","), dtype=float))
                mergedf = pd.concat([mergedf, tempdf], axis=1)
        return mergedf

    def _extract_coding_sequences(self):
        '''
        Extract coding sequences from genome fasta and gff annotation.

        CDS segments of each transcript are spliced together from the genome
        (see :meth:`AnnotationDB.extract_cds_sequences`). Gene names are
        resolved via gffutils by traversing the GFF feature hierarchy
        (transcript → gene). When multiple isoforms map to the same gene the
        first isoform in genomic order is kept.

        Returns
        -------
        cds_seq : dict
            Dictionary mapping gene name to CDS sequence string.
        '''
        tx_seq = self.annot_db.extract_cds_sequences(self.genome_fasta_file)
        if not tx_seq:
            message = (f"No CDS sequences could be extracted from '{self.gff_file}' with genome '{self.genome_fasta_file}'. "
                       f"Ensure the annotation and genome files are compatible.")
            log.error(message)
            raise RuntimeError(message)

        # Build transcript_id -> gene_name mapping from GFF using gffutils.
        tx_to_gene = self.annot_db.build_feature_to_gene_map()

        cds_seq: dict[str, str] = {}
        n_isoforms = 0
        for tx_id, seq in tx_seq.items():
            # Map transcript ID to gene name; fall back to tx_id if not found.
            gene_name = tx_to_gene.get(tx_id, tx_id)
            # Keep only the first isoform per gene.
            if gene_name not in cds_seq:
                cds_seq[gene_name] = seq
            else:
                n_isoforms += 1
        if n_isoforms > 0:
            log.warning("%d genes have multiple coding transcripts. Keeping the first isoform of each gene.", n_isoforms)

        log.info("Extracted CDS sequences for %d genes.", len(cds_seq))
        return cds_seq

    def _process1gene(self, gene, maxlength, mindensity, maxutr5length, maxutr3length):
        '''
        Process a single gene to extract features and ribosome profiling data.

        Parameters
        ----------
        gene : str
            Gene identifier.
        maxlength : int
            Maximum allowed total sequence length (UTR5 + CDS + UTR3).
        mindensity : float
            Minimum average ribosome density (reads per codon) for inclusion.
        maxutr5length : int
            Maximum length of 5' UTR to extract.
        maxutr3length : int
            Maximum length of 3' UTR to extract.

        Returns
        -------
        tuple of pd.Series
            (x_c, y_c) where x_c contains input features and y_c contains output labels.
            Returns (None, None) if gene doesn't meet filtering criteria.
        '''

        x_c = pd.Series(index=["utr5", "cds", "utr3", "utr5_len", "cds_len", "utr3_len", "rna", "longest_utr5_annot", "longest_utr3_annot"], name=gene, dtype=object)
        y_c = pd.Series(name=gene, dtype=object)

        # Use riboA Psite centred coverage
        sortedcond = sorted(self.psites.keys())
        data = {cond:genedf.loc[gene] for cond, genedf in self.psites.items()}
        exampledf = list(data.values())[0]
        if (exampledf.loc["length"]/3) > 50 :
            RDs = {}
            for condition, df in data.items():
                psites_data = df.loc["psites"]
                stacked_data = np.stack(psites_data)
                stacked_data = np.atleast_2d(stacked_data)
                transposed_data = stacked_data.T
                RDs[condition] = transposed_data
            cdsRDs = {}
            for condition, data in RDs.items():
                start = self.extend_prof
                end = data.shape[0] - self.extend_prof
                cdsRDs[condition] = data[start:end, :]
            utr5RDs = {cond:df[:self.extend_prof,:] for cond, df in RDs.items()}
            utr3RDs = {cond:df[(df.shape[0]-self.extend_prof):,:] for cond, df in RDs.items()}
            geneseq = self.cds_seq[gene]
            geneannot = self.cds_annot.loc[self.cds_annot.gene==gene].iloc[0]
            chrom = geneannot.loc["chr"]
            cdsrange = [geneannot.loc["start"], geneannot.loc["end"]]
            chrom_len = self.chrom_sizes.get(chrom, float("inf"))
            if geneannot.loc["strand"] == "+":
                # positive strand
                bed5 = (chrom, max(0, cdsrange[0]-self.extend_prof), cdsrange[0]+3, "+")
                bed3 = (chrom, cdsrange[1]-3, min(int(chrom_len), cdsrange[1]+self.extend_prof), "+")
                # Identify surrounding CDS
                surroundutr5 = self.cds_annot.loc[self.cds_annot.gene!=gene].loc[np.logical_and(self.cds_annot.chr==geneannot.loc["chr"], np.logical_and(self.cds_annot.strand=="+", np.logical_and(geneannot.loc["end"]>self.cds_annot.end, self.cds_annot.end>(geneannot.loc["start"]-self.extend_prof))))]
                surroundutr3 = self.cds_annot.loc[self.cds_annot.gene!=gene].loc[np.logical_and(self.cds_annot.chr==geneannot.loc["chr"], np.logical_and(self.cds_annot.strand=="+", np.logical_and(geneannot.loc["start"]<self.cds_annot.start, self.cds_annot.start<(geneannot.loc["end"]+self.extend_prof))))]
                maskutr5 = []
                for _, row in surroundutr5.iterrows():
                    maskutr5.append([int(max(0,row.loc["start"]-(geneannot.loc["start"]-self.extend_prof))), int(min(self.extend_prof,row.loc["end"]-(geneannot.loc["start"]-self.extend_prof)))])
                maskutr3 = []
                for _, row in surroundutr3.iterrows():
                    maskutr3.append([int(max(0,row.loc["start"]-geneannot.loc["end"])), int(min(self.extend_prof,row.loc["end"]-geneannot.loc["end"]))])
            else:
                # negtive strand
                bed5 = (chrom, cdsrange[0]-3, min(int(chrom_len), cdsrange[0]+self.extend_prof), "-")
                bed3 = (chrom, max(0, cdsrange[1]-self.extend_prof), cdsrange[1]+3, "-")
                # Identify surrounding CDS
                surroundutr5 = self.cds_annot.loc[self.cds_annot.gene!=gene].loc[np.logical_and(self.cds_annot.chr==geneannot.loc["chr"], np.logical_and(self.cds_annot.strand=="-", np.logical_and(geneannot.loc["end"]<self.cds_annot.end, self.cds_annot.end<(geneannot.loc["start"]+self.extend_prof))))]
                surroundutr3 = self.cds_annot.loc[self.cds_annot.gene!=gene].loc[np.logical_and(self.cds_annot.chr==geneannot.loc["chr"], np.logical_and(self.cds_annot.strand=="-", np.logical_and(geneannot.loc["start"]>self.cds_annot.start, self.cds_annot.start>(geneannot.loc["end"]-self.extend_prof))))]
                maskutr5 = []
                for _, row in surroundutr5.iterrows():
                    maskutr5.append([int(max(0,-(row.loc["start"]-(geneannot.loc["start"]+self.extend_prof)))), int(min(self.extend_prof,-(row.loc["end"]-(geneannot.loc["start"]+self.extend_prof))))])
                maskutr3 = []
                for _, row in surroundutr3.iterrows():
                    maskutr3.append([int(max(0,-(row.loc["start"]-geneannot.loc["end"]))), int(min(self.extend_prof,-(row.loc["end"]-geneannot.loc["end"])))])

            # Build UTR exon masks: True = exonic position in the extend_prof window.
            gene_utr_exons = self.utr_exon_intervals.get(gene, {"5UTR": [], "3UTR": []})
            strand = geneannot.loc["strand"]
            mask5 = build_utr_exon_mask(cdsrange[0], strand, gene_utr_exons["5UTR"], "5UTR", self.extend_prof)
            mask3 = build_utr_exon_mask(cdsrange[1], strand, gene_utr_exons["3UTR"], "3UTR", self.extend_prof)

            # Gather UTR length data
            closestATG = min([self.extend_prof - s[0] for s in maskutr5 if s[0] < self.extend_prof and mask5[s[0]]], default=np.nan)
            if self.utr5 is not None:
                tss5 = self.utr5.loc[self.utr5.gene == gene]
                maxlen5 = max(tss5.end.values, default=np.nan) - min(tss5.start.values, default=np.nan)
                raw_maxlen5 = maxlen5 if not tss5.empty else closestATG
            else:
                raw_maxlen5 = closestATG
                maxlen5 = np.nan

            if self.utr3 is not None:
                tss3 = self.utr3.loc[self.utr3.gene == gene]
                maxlen3 = max(tss3.end.values, default=np.nan) - min(tss3.start.values, default=np.nan)
                raw_maxlen3 = maxlen3 if not tss3.empty else np.nan
            else:
                raw_maxlen3 = np.nan
                maxlen3 = np.nan

            # Convert raw genomic UTR length → exonic (spliced) length.
            annot_slice5 = int(np.nanmin([raw_maxlen5, self.extend_prof]))
            len5 = min(int(mask5[(self.extend_prof - annot_slice5):].sum()), maxutr5length)
            if not np.isnan(maxlen5):
                maxlen5 = int(mask5[(self.extend_prof - annot_slice5):].sum()) + max((raw_maxlen5 - self.extend_prof), 0)

            annot_slice3 = int(np.nanmin([raw_maxlen3, self.extend_prof]))
            len3 = min(int(mask3[:annot_slice3].sum()), maxutr3length)
            if not np.isnan(maxlen3):
                maxlen3 = int(mask3[:annot_slice3].sum()) + max((raw_maxlen3 - self.extend_prof), 0)

            if np.mean([np.mean(c)*3 for c in cdsRDs.values()]) > mindensity and (len5+len(geneseq)+len3) < maxlength:
                seq5 = fetch_sequence(self.genome_fasta_file, *bed5)
                seq3 = fetch_sequence(self.genome_fasta_file, *bed3)
                # Pad to expected length (extend_prof + 3) if coordinates were clamped.
                seq5 = seq5.rjust(self.extend_prof + 3, "X")
                seq3 = seq3.ljust(self.extend_prof + 3, "X")
                # Splice out intronic positions from UTR windows; pad with 'X'.
                seq5 = splice_utr_seq(seq5, mask5, "5UTR")
                seq3 = splice_utr_seq(seq3, mask3, "3UTR")
                # Trim to fixed output length
                seq5 = seq5[-(maxutr5length + 3):]
                seq3 = seq3[:(maxutr3length + 3)]
                x_c.loc[["utr5", "utr3"]] = [seq5, seq3]

                # Record info
                x_c.loc[["utr5_len","utr3_len"]] = [len5, len3]
                x_c.loc[["longest_utr5_annot","longest_utr3_annot"]] = [maxlen5, maxlen3]
                x_c.loc["cds"] = geneseq
                x_c.loc["cds_len"] = len(geneseq)
                cdslen = x_c.loc["cds_len"]
                # Get RNA-seq counts
                x_c.loc["rna"] = [np.mean(self.norm_rna_counts.loc[gene, self.sample_file.loc[self.sample_file["track"]==cond,"sample_id"].values]) for cond in sortedcond]

                # Build per-condition RD arrays for each region.
                filledRDs = []
                for region in ["cds", "utr5", "utr3"]:
                    rdtemp = cdsRDs if region=="cds" else utr5RDs if region=="utr5" else utr3RDs
                    for cond in sortedcond:
                        repslist = []
                        for n in range(rdtemp[cond].shape[1]):
                            if region=="cds":
                                condRD = np.full((cdslen,), np.nan)
                                condRD[:min(cdslen,rdtemp[cond].shape[0])] = rdtemp[cond][:min(cdslen,rdtemp[cond].shape[0]),n]
                            elif region=="utr5":
                                maskedrd = rdtemp[cond][:, n].copy()
                                # mask positions that overlap with surrounding CDS (pre-splice coords)
                                for coord in maskutr5:
                                    maskedrd[coord[0]:coord[1]] = np.nan
                                # splice out intronic positions and apply padding
                                condRD = splice_utr_rd(maskedrd, mask5, "5UTR")
                                # Trim to fixed output length (right-aligned, ATG-adjacent at end)
                                condRD = condRD[-maxutr5length:]
                            elif region=="utr3":
                                maskedrd = rdtemp[cond][:, n].copy()
                                # mask positions that overlap with surrounding CDS (pre-splice coords)
                                for coord in maskutr3:
                                    maskedrd[coord[0]:coord[1]] = np.nan
                                # splice out intronic positions and apply padding
                                condRD = splice_utr_rd(maskedrd, mask3, "3UTR")
                                # Trim to fixed output length (left-aligned, stop-codon-adjacent at start)
                                condRD = condRD[:maxutr3length]
                            repslist.append(condRD)
                        filledRDs.append(repslist)

                condindex = pd.MultiIndex.from_product([["cds","utr5","utr3"],sortedcond], names=["region", "cond"])
                y_c = pd.Series(filledRDs, index=condindex, name=gene, dtype=object)

        return x_c, y_c

    def _extract_TSS_annotations(self, utrannot, utr_type, maxutrlength):
        '''
        Extract transcription start site (TSS) annotations from UTR annotation files.

        Parameters
        ----------
        utrannot : str or None
            Path to UTR annotation file. Supported formats: .gtf, .fasta, .fa, .fsa.
        utr_type : str
            Type of UTR ("5UTR" or "3UTR").
        maxutrlength : int
            Maximum length of UTR to consider.

        Returns
        -------
        pd.DataFrame or None
            DataFrame containing UTR annotations, or None if no file provided.

        Raises
        ------
        ValueError
            If utrannot has an unsupported file format.
        '''
        if utrannot is None:
            log.info(f"No {utr_type} annotation fasta provided. Extracting annotations from '{self.gff_file}'.")
            utr = self.annot_db.parse_annotation(utr_type)
            if utr.empty:
                utr = None
                if utr_type == "5UTR":
                    log.warning(f"No 5' UTR annotations found in '{self.gff_file}'. Taking default '{maxutrlength}' upstream nucleotides from the start codon.")
                else:
                    log.warning(f"No 3' UTR annotations found in '{self.gff_file}'. Taking default '{maxutrlength}' downstream nucleotides from the stop codon.")
        elif utrannot.endswith(".fsa") or utrannot.endswith(".fasta") or utrannot.endswith(".fa"):
            utr = fasta2tab(utrannot, utr_type)
        else:
            log.error("Argument utrannot received an invalid option. Supported formats are .fasta/.fa/.fsa.")
            raise ValueError("Argument utrannot received an invalid option. Supported formats are .fasta/.fa/.fsa.")

        return utr

    def generate_training_data(self, threads=1, maxlength=4000, mindensity=0.1, utr5annot=None, utr3annot=None,
                               maxutr5length=500, maxutr3length=500):
        '''
        Generate training datasets for RiboNet.

        Parameters
        ----------
        threads : int, default = 1
            Number of threads to use for multithreading.
        maxlength : int, default = 3000
            Maximum allowed length of the combined sequence (UTR5 + CDS + UTR3).
        mindensity : float, default = 0.1
            Minimum ribosome density (reads per codon) to include a gene in the training set.
        utr5annot : str, default = None
            Path to 5' UTR annotation file (.gtf or .fasta).
        utr3annot : str, default = None
            Path to 3' UTR annotation file (.gtf or .fasta).
        maxutr5length : int, default = 500
            Maximum length of 5' UTR to consider.
        maxutr3length : int, default = 500
            Maximum length of 3' UTR to consider.

        Returns
        -------
        x_c : pd.DataFrame
            DataFrame with input features for training.
        y_c : pd.DataFrame
            DataFrame with output labels for training.
        '''
        # Load P site data
        conditions = sorted(self.sample_file['track'].unique().tolist())
        self.psites = dict()
        for cond in conditions:
            self.psites[cond] = self._merge_reps(sorted(self.sample_file.loc[self.sample_file['track']==cond,'sample_id'].to_list()))

        # Extract coding sequences
        log.info("Extract coding sequences")

        self.cds_seq = self._extract_coding_sequences()
        self.cds_annot = self.annot_db.parse_annotation("CDS")

        # Get UTR annotations
        log.info("Load UTR annotations")

        self.utr5 = self._extract_TSS_annotations(utr5annot, "5UTR", maxutr5length)
        self.utr3 = self._extract_TSS_annotations(utr3annot, "3UTR", maxutr3length)

        # Parse per-gene UTR exon structure from the annotation database.
        log.info("Parsing UTR exon intervals for intron splicing.")
        self.utr_exon_intervals = self.annot_db.parse_utr_exon_intervals(utr5_df=self.utr5, utr3_df=self.utr3)

        # Load chromosome sizes for coordinate clamping in _process1gene
        with pysam.FastaFile(self.genome_fasta_file) as fa:
            self.chrom_sizes = dict(zip(fa.references, fa.lengths))

        log.info("Generating training datasets...")
        frozen_fun = partial(self._process1gene, maxlength=maxlength, mindensity=mindensity, maxutr5length=maxutr5length, maxutr3length=maxutr3length)
        rpfset = set(list(self.psites.values())[0].index.unique())
        rnaset = set(self.norm_rna_counts.index)
        unmapped = list(set((rpfset-rnaset)|(rnaset-rpfset)))
        if len(unmapped)>0:
            log.warning("Genes could not be mapped between RPF and RNA: %s", ','.join(unmapped))
        genelist = rpfset.intersection(rnaset).intersection(set(self.cds_seq.keys()))
        if self.spikein is not None:
            genelist = genelist - set(self.spikein)
        genelist = sorted(genelist)  # deterministic gene order (the train/test split depends on it)

        if len(genelist) == 0:
            err_message = ("No genes remain after intersecting RPF, RNA-seq and CDS annotation gene sets. "
                           "Check that gene identifiers are consistent across the BAM files (via GFF annotation), "
                           "RNA counts file, and genome annotation.")
            log.error(err_message)
            raise RuntimeError(err_message)
        log.info("Processing %d genes for training data generation.", len(genelist))

        chunksize = int(np.ceil(len(genelist) / threads))
        x_c, y_c = zip(*parallel_map(frozen_fun, genelist, threads=threads, chunksize=chunksize, desc="Processing genes"))

        # Merge dataframes
        x_c = pd.DataFrame(x_c).dropna(how="all")
        y_c = pd.DataFrame(y_c).dropna(how="all")

        if x_c.empty:
            err_message = ("No genes passed the filtering criteria (mindensity and maxlength). "
                           f"Current settings: mindensity={mindensity}, maxlength={maxlength}. "
                           "Consider relaxing these thresholds or checking your input data.")
            log.error(err_message)
            raise RuntimeError(err_message)
        log.info("Finished generating the training datasets")

        x_c.to_parquet(os.path.join(self.preprocess_dir, "x_inputs.parquet"), index=True)
        y_c.to_parquet(os.path.join(self.preprocess_dir, "y_outputs.parquet"), index=True)

        log.info("Datasets saved. Size: %s", x_c.shape[0])

        return x_c, y_c

    def compute_asite_occupancy(self, trim: int = 15) -> None:
        """Compute per-sample A-site codon occupancy and save results to the riboA directory.

        For each gene/sample, counts at each CDS position are normalised by the
        per-(gene, sample) mean (excluding the first and last *trim* nucleotides).
        The mean normalised occupancy is then averaged over all positions that map
        to the same A-site codon and saved as a TSV table.

        A heatmap of the log2 pause scores is saved to ``riboA/qc_plots/codon_occupancy.pdf``.
        If a PDF report is currently being built (i.e. the :mod:`utils.report_utils` global
        document is active), the codon-occupancy section is appended to it and the report
        is finalised.  This happens automatically when :meth:`generate_offset_report` is
        called with ``finalize=False`` followed by this method.

        Parameters
        ----------
        trim : int, default 15
            Number of nucleotides to exclude from each end of the CDS.

        Returns
        -------
        None
            Saves ``riboA/asite_occupancy.tsv`` and
            ``riboA/qc_plots/codon_occupancy.pdf``.
        """

        tsv_path = os.path.join(self.riboAdir, "asite_occupancy.tsv")
        plot_path = os.path.join(self.qc_plots, "codon_occupancy.pdf")

        x_inputs_path = os.path.join(self.preprocess_dir, "x_inputs.parquet")
        y_outputs_path = os.path.join(self.preprocess_dir, "y_outputs.parquet")

        if not os.path.isfile(x_inputs_path) or not os.path.isfile(y_outputs_path):
            log.warning(
                "x_inputs.parquet or y_outputs.parquet not found in %s. "
                "Run generate_training_data() first.",
                self.preprocess_dir,
            )
            return

        log.info("Computing A-site codon occupancy (trim=%d nt)...", trim)

        x_c = pd.read_parquet(x_inputs_path)
        y_c = pd.read_parquet(y_outputs_path)
        cds_y = y_c["cds"]
        conditions = list(cds_y.columns)

        # Build track -> ordered list of sample_ids from sample_file
        track_to_samples: dict = {}
        for track, grp in self.sample_file.groupby("track", sort=False):
            track_to_samples[track] = grp["sample_id"].tolist()

        STOP_CODONS = {"TAA", "TAG", "TGA"}
        pause_sum: dict = {}
        pause_count: dict = {}
        genes_in_y = set(cds_y.index)

        for gene in x_c.index:
            if gene not in genes_in_y:
                continue
            cds_len = int(x_c.loc[gene, "cds_len"])
            cds_seq = x_c.loc[gene, "cds"][:cds_len]
            start, end = trim, cds_len - trim
            if end <= start:
                continue
            codons = [cds_seq[i: i + 3] for i in range(0, cds_len, 3)]
            positions = np.arange(start, end)
            a_codon_idx = (positions + 3) // 3
            asite_arr = np.array(
                [
                    codons[idx] if idx < len(codons) and len(codons[idx]) == 3 else ""
                    for idx in a_codon_idx
                ]
            )
            valid_mask = asite_arr != ""

            for cond in conditions:
                rep_list = cds_y.loc[gene, cond]
                if rep_list is None:
                    continue
                try:
                    rep_mat = np.stack([np.asarray(r, dtype=float) for r in rep_list])
                except Exception:
                    continue
                trimmed = rep_mat[:, start:end]
                rep_means = np.nanmean(trimmed, axis=1)
                sample_ids = track_to_samples.get(cond)
                for rep_idx in range(rep_mat.shape[0]):
                    mean_val = rep_means[rep_idx]
                    if mean_val <= 0 or np.isnan(mean_val):
                        continue
                    pause = trimmed[rep_idx] / mean_val
                    if sample_ids is not None and rep_idx < len(sample_ids):
                        label = sample_ids[rep_idx]
                    else:
                        label = f"{cond}_rep{rep_idx}"
                    include = valid_mask & ~np.isnan(pause)
                    for p_val, a_codon in zip(pause[include], asite_arr[include]):
                        key = (label, a_codon)
                        if key in pause_sum:
                            pause_sum[key] += p_val
                            pause_count[key] += 1
                        else:
                            pause_sum[key] = p_val
                            pause_count[key] = 1

        if not pause_sum:
            log.warning("A-site occupancy: no valid data found — skipping.")
            return

        rows = [
            {
                "sample": label,
                "Asite": asite,
                "mean_pause": pause_sum[(label, asite)] / pause_count[(label, asite)],
            }
            for label, asite in pause_sum
        ]
        result = pd.DataFrame(rows).pivot(index="sample", columns="Asite", values="mean_pause")
        result.columns.name = None
        result.index.name = "cond"
        stopcod = [c for c in STOP_CODONS if c in result.columns]
        result = result.drop(columns=stopcod, errors="ignore")
        result = result[sorted(result.columns)]
        # Sort samples by readlength first, then by condition
        order_df = (
            self.sample_file[["sample_id", "condition", "readlength"]]
            .drop_duplicates("sample_id")
        )
        order_df = order_df[order_df["sample_id"].isin(result.index)]
        order_df = order_df.sort_values(["readlength", "condition"])
        ordered = order_df["sample_id"].tolist()
        extra = [s for s in result.index if s not in set(ordered)]
        result = result.loc[ordered + extra]

        result.to_csv(tsv_path, sep="\t")
        log.info("A-site occupancy table saved to %s", tsv_path)

        # Generate heatmap
        v = 2
        fig, ax = plt.subplots(figsize=(15, max(0.3 * result.shape[0], 3)))
        sns.heatmap(
            np.log2(result),
            cmap=color_map,
            center=0.0,
            vmin=-v,
            vmax=v,
            ax=ax,
            yticklabels=True,
        )
        ax.set_title("A-site codon occupancy")
        ax.set_xlabel("A-site codon")
        ax.set_ylabel("Sample")
        fig.tight_layout()
        fig.savefig(plot_path, format="pdf", bbox_inches="tight")
        log.info("A-site occupancy heatmap saved to %s", plot_path)

        # Append section to the ongoing QC report if one is active, then finalize
        if _ru._doc is not None:
            _ru.add_page_break()
            _ru.add_heading("Codon A-site Occupancy")
            _ru.add_text(
                "The heatmap below shows the per-sample A-site codon occupancy: "
                "the ratio of ribosome density at each A-site codon to the "
                "mean density along the CDS, averaged across all genes. "
                "The first and last "
                + str(trim)
                + " nucleotides of each CDS are excluded. "
                "Stop codons are omitted. "
                "Values are shown on a log2 scale; values above 0 indicate "
                "higher-than-average occupancy (slow decoding), values below 0 "
                "indicate faster-than-average decoding."
            )
            _ru.add_plot(fig)
            _ru.finalize_report()
            log.info("QC report finalised.")

        plt.close(fig)
