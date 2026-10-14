import os
import sys
import argparse
import logging

from .version import __version__

def _preload_libz():
    '''
    Work around a zlib symbol collision that corrupts the heap during report
    generation. pyarrow's ``libarrow.so`` pulls in ``liborc.so`` (Apache ORC),
    which statically bundles zlib and exports ``deflate*``/``inflate*`` as GLOBAL
    symbols. When the real ``libz`` is also loaded, ``zlib.compress`` can bind
    ``deflateInit2_`` to liborc's zlib but ``deflate``/``fill_window`` to libz;
    the two disagree on the deflate-state size, so ``deflate`` writes past the
    buffer -> ``free(): invalid next size`` / SIGABRT, typically surfacing in
    reportlab's image compression at the end of ``ribonet report``.

    Preloading the real libz forces every deflate consumer (Python's zlib module
    and libarrow alike) onto a single zlib. ``LD_PRELOAD`` only takes effect from
    process start, so if it is not already set we set it and re-exec once. This is
    a no-op when libz is already preloaded (e.g. via a conda ``env config vars``
    entry), so it never loops or double-preloads.
    '''
    libz = os.path.join(sys.prefix, "lib", "libz.so.1")
    if not os.path.exists(libz):
        return  # non-conda / unexpected layout: nothing we can safely preload
    current = os.environ.get("LD_PRELOAD", "")
    if libz in current.split(os.pathsep):
        return  # already preloaded (or we already re-exec'd)
    os.environ["LD_PRELOAD"] = os.pathsep.join(p for p in (libz, current) if p)
    if sys.argv[0].endswith("__main__.py"):
        argv = [sys.executable, "-m", "ribonet"] + sys.argv[1:]  # invoked as `python -m ribonet`
    else:
        argv = [sys.executable] + sys.argv
    os.execv(sys.executable, argv)


log = logging.getLogger(__name__)

def config_logging(project_dir, verbose=1):
    '''Configure logging for RiboNet analysis.

    Parameters
    ----------
    project_dir : str
        Directory to save project outputs including the log file.
    verbose : int, default = 1
        Verbosity level. If >= 1, sets logging level to INFO, otherwise WARNING.
    '''
    os.makedirs(project_dir, exist_ok=True)

    level = logging.INFO if verbose >= 1 else logging.WARNING
    fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

    file_handler = logging.FileHandler(os.path.join(project_dir, 'ribonet.log'))
    file_handler.setFormatter(fmt)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)

    logging.basicConfig(level=level, handlers=[file_handler, console_handler])

def preprocess_inputs(sample_file, rna_counts_file, genome_fasta_file, gff_file, project_dir,
                      threads=1, spikein=None, spikein_absolute_ribosomes=None, maxlength=4000, mindensity=0.1,
                      utr5annot=None, utr3annot=None, maxutr5length=500, maxutr3length=500, fixed_5offsets=None):
    '''Preprocess input files for RiboNet analysis.

    Parameters
    ----------
    sample_file : str
        Path to the sample information file.
    rna_counts_file : str
        Path to the RNA counts file.
    genome_fasta_file : str
        Path to the genome FASTA file.
    gff_file : str
        Path to the GFF/GTF annotation file.  Accepts .gff, .gff3, or .gtf
        for any species.
    project_dir : str
        Directory to save project outputs.
    threads : int
        Number of threads to use for multithreading.
    spikein : str, default = None
        Comma-separated list of spike-in gene names or path to text file with one gene name per line. If provided, calibrate the RPF counts with the spike-in counts.
    spikein_absolute_ribosomes : int, default = None
        If known, total number of ribosomes per transcript in spike-in genes (e.g. Fluc trisomes, 3, and Rluc disomes, 2, sum up to 5). Used for absolute normalization. If not set, relative normalization will be performed.
    maxlength : int, default = 4000
        Maximum allowed length of the combined sequence (UTR5 + CDS + UTR3).
    mindensity : float, default = 0.1
        Minimum ribosome density (reads per codon) to include a gene in the training set.
    utr5annot : str, default = None
        Path to 5\' UTR annotation FASTA file (SGD style).
    utr3annot : str, default = None
        Path to 3\' UTR annotation FASTA file (SGD style).
    maxutr5length : int, default = 500
        Maximum length of 5\' UTR to consider.
    maxutr3length : int, default = 500
        Maximum length of 3\' UTR to consider.
    fixed_5offsets: dict, default = None
        Specify transcript offsets instead of performing estimation. If used, provide one offset for each transcript length as a dictionary in the format {readlength: offset} (e.g., {27: 15, 28: 15, 29: 16, 30: 16}).

    Returns
    -------
    x_input : pd.DataFrame
        Preprocessed input features for RiboNet.
    y_output : pd.DataFrame
        Preprocessed output labels for RiboNet.
    '''

    log.info("\n+-----------------------+\
	\n| RiboNet Preprocessing |\
	\n+-----------------------+")

    from .preprocessing import RiboPreprocess

    preprocessor = RiboPreprocess(sample_file, genome_fasta_file, gff_file, project_dir, spikein)
    preprocessor.generate_offset_report(threads=threads, finalize=False)
    preprocessor.compute_riboA(threads=threads, fixed_5offsets=fixed_5offsets)
    preprocessor.count_RPF()
    preprocessor.extract_rna_abundance(rna_counts_file, spikein_absolute_ribosomes)
    x_input, y_output = preprocessor.generate_training_data(threads=threads, maxlength=maxlength, mindensity=mindensity,
                                                            utr5annot=utr5annot, utr3annot=utr3annot, maxutr5length=maxutr5length, maxutr3length=maxutr3length)
    preprocessor.compute_asite_occupancy()

    return x_input, y_output


def train_ribonet(project_dir, resolution, cache_dir=None, model_dir=None,
                  lm=None, species_proxy=None, epoch=50, batch_size=32, split=0.7,
                  learning=0.0005, weight_decay=0.01, warmup=5, alpha=0.1,
                  beta=None, utr_pool=0.3, id=None,
                  ablate_utr5=False, ablate_utr3=False):
    """Train the model and generate summary statistics of the training

    Parameters
    ----------
    project_dir : str or pathlike
        Directory to save project outputs.
    resolution : str
        Either 'codon' or 'nucleotide'.
        Determines whether the ribosome profile should be predicted per nucleotide or per codon.
    cache_dir : str, default = None
        Cache directory for downloading the language model. Ignored if no language
        model is selected. If not set, the Hugging Face default cache is used.
    model_dir : str or dict, default=None
        Directory to load the language model from if it exists, else the downloaded
        model is saved there. Ignored if no language model is selected. If not set,
        the language model is downloaded or loaded from the cache.
    lm : str or dict, default=None
        Selects the language model to use.

        - Either 'rinalmo' (nucleotide resolution) or a dict such as {"UTR5": "specieslm5", "UTR3": "specieslm3"} (codon resolution).
        - If None, a ResNet will be trained and used for generating embeddings.
    species_proxy : str or dict, default=None
        - Must be set if a speciesLM is used, else ignored.
        - Passed to the selected speciesLM
        - format: either a string or a dict: {'UTR5': species_proxy, 'UTR3': species_proxy}
    epoch : int, default 50
        The maximum number of training epochs.
        Training might finish earlier due to early stopping.
    batch_size : int, default = 32
        The batch size used for training.
    split : int, default = 0.7
        The training set size in percent of the overall data.
        The rest is equally split into validation and test set.
    learning : float, default = 0.0005
        The learning rate used for training.
    weight_decay : float, default = 0.01
        The weight decay used for the model training.
    warmup : int, default = 5
        The number of warmup steps for the learning.
    alpha : float, default = 0.1
        The weight for the count loss
    beta : float, default = None
        The weight for the profile loss
    utr_pool : float, default = 0.3
        Ignored if RiboTraining was initialized with seq_resolution = 'nucleotide'.
        Else used to condense UTR regions
    id : str, default = None
        A optional descriptive string.
    """

    log.info("\n+------------------+\
	\n| RiboNet Training |\
	\n+------------------+")

    from .training import RiboTraining

    trainer = RiboTraining(project_dir, resolution, cache_dir=cache_dir, model_dir=model_dir, lm=lm, species_proxy=species_proxy)
    trainer.initialize_training(epoch=epoch, batch=batch_size, split=split, learning=learning, weight_decay=weight_decay, warmup=warmup, alpha=alpha, beta=beta, utr_pool=utr_pool, id=id,
                                ablate_utr5=ablate_utr5, ablate_utr3=ablate_utr3)
    trainer.run_training()


def interpret_ribonet(project_dir, n_batches, scoring, save_distribution, inference_batch_size=None):
    '''Interpret and analyze a trained RiboNet model.

    Parameters
    ----------
    project_dir : str
        Directory containing the trained RiboNet model and project outputs.
    n_batches : int
        Number of batches to divide the interpretation task into.
    scoring : str
        Scoring method to use for feature importance (e.g., 'ISM' for in silico mutagenesis).
    save_distribution : bool
        Whether to save the distribution of importance scores.
    inference_batch_size : int, optional
        Batch size to use for inference during interpretation. If None, defaults to four times the training batch size.
    '''

    log.info("\n+------------------------+\
	\n| RiboNet Interpretation |\
	\n+------------------------+")

    from .interpretation import RiboInterpretation

    interpreter = RiboInterpretation(project_dir)
    interpreter.score_importances(scoring=scoring, n_batches=n_batches, save_distribution=save_distribution, inference_batch_size=inference_batch_size)


def score_batch(project_dir, batch_number):
    '''Score a specific batch of sequences using a trained RiboNet model.

    Parameters
    ----------
    project_dir : str
        Directory containing the trained RiboNet model and project outputs.
    batch_number : int
        The batch number to score.
    '''
    from .interpretation import RiboInterpretation

    interpreter = RiboInterpretation(project_dir)
    interpreter.load_batch(batch_number)
    interpreter._score()

def report_interpretation(project_dir, ctrl_cond, threads, motif_db, diffsites, pwind, max_seqlets, filename):
    '''Generate an interpretation report from scored RiboNet results.

    Parameters
    ----------
    project_dir : str
        Directory containing the scored RiboNet results.
    ctrl_cond : str
        Name of the control condition for comparison.
    threads : int
        Number of threads to use for parallel processing.
    motif_db : str
        Path to the motif database file, or 'default' to use the default database.
    diffsites : int
        Number of top differentially occupied sites to analyze.
    pwind : int
        Window size around differentially occupied sites to consider.
    max_seqlets : int
        Maximum number of seqlets per metacluster passed to TF-MoDISco.
    filename : str
        Output filename for the interpretation report.
    '''

    log.info("\n+-------------------------------+\
	\n| RiboNet Interpretation Report |\
	\n+-------------------------------+")

    from .interpretation import RiboInterpretation

    interpreter = RiboInterpretation(project_dir)
    interpreter.generate_report(ctrl_cond=ctrl_cond, threads=threads, motif_db=motif_db, diffsites=diffsites,
                                pwind=pwind, max_seqlets=max_seqlets, filename=filename)

def main():
    '''Main entry point for the RiboNet command-line interface.

    Parses command-line arguments and dispatches to the appropriate subcommand
    (preprocess, train, interpret, score_batch, or report).
    '''
    parser = argparse.ArgumentParser(
        prog="ribonet",
        description="Deep learning analysis toolkit for ribosome profiling"
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # SUBCOMMAND: preprocess
    p_pre = subparsers.add_parser("preprocess", help="Preprocess data for RiboNet training and interpretation")
    p_pre.add_argument("--sample_file", required=True, help="Path to the sample information file")
    p_pre.add_argument("--rna_counts", required=True, help="Path to the RNA counts file")
    p_pre.add_argument("--genome_fasta", required=True, help="Path to the genome FASTA file")
    p_pre.add_argument("--gff", required=True, help="Path to the GFF/GTF annotation file (.gff, .gff3, or .gtf). Works for any species.")
    p_pre.add_argument("--project_dir", required=True, help="Directory to save project outputs")
    p_pre.add_argument("--threads", type=int, default=1, help="Number of threads to use for multithreading (default: 1)")
    p_pre.add_argument("--spikein", default=None, help="Comma-separated list of spike-in gene names or path to text file with one gene name per line. Used for calibrating RPF counts")
    p_pre.add_argument("--spikein_absolute_ribosomes", type=int, default=None, help="If known, total number of ribosomes per transcript in spike-in genes (e.g. Fluc trisomes, 3, and Rluc disomes, 2, sum up to 5). Used for absolute normalization. If not set, relative normalization will be performed.")
    p_pre.add_argument("--maxlength", type=int, default=4000, help="Maximum allowed length of combined sequence (UTR5 + CDS + UTR3) (default: 4000)")
    p_pre.add_argument("--mindensity", type=float, default=0.1, help="Minimum ribosome density (reads per codon) to include a gene (default: 0.1)")
    p_pre.add_argument("--utr5annot", default=None, help="Path to 5' UTR annotation FASTA (SGD style, e.g. SGD_all_ORFs_5prime_UTRs.fsa). If not set, UTRs are taken from the --gff annotation")
    p_pre.add_argument("--utr3annot", default=None, help="Path to 3' UTR annotation FASTA (SGD style, e.g. SGD_all_ORFs_3prime_UTRs.fsa). If not set, UTRs are taken from the --gff annotation")
    p_pre.add_argument("--maxutr5length", type=int, default=500, help="Maximum length of 5' UTR to consider (default: 500)")
    p_pre.add_argument("--maxutr3length", type=int, default=500, help="Maximum length of 3' UTR to consider (default: 500)")
    p_pre.add_argument("--verbose", type=int, default=1, help="Verbosity level: 0 for warnings only, 1+ for info (default: 1)")
    p_pre.add_argument("--fixed_5offsets", default=None, help="Specify transcript offsets instead of performing RiboA estimation. " \
    "If used, provide one offset for each transcript length,in the format readlegth-offset and separated by commas (eg. `--fixed_5offsets 27-15,28-15,29-16,30-16`).")
    p_pre.set_defaults(func="preprocess")

    # SUBCOMMAND: train
    p_train = subparsers.add_parser("train", help="Train RiboNet and evaluate training performance")
    p_train.add_argument("--project_dir", required=True, help="Directory to save project outputs and trained models")
    p_train.add_argument("--resolution", default="codon", help="Prediction resolution: 'codon' or 'nucleotide' (default: codon)")
    p_train.add_argument("--cache_dir", default=None, help="Cache directory for language model downloads (default: home directory)")
    p_train.add_argument("--model_dir", default=None, help="Directory to load/save language model (for nucleotide resolution)")
    p_train.add_argument("--model_dir5", default=None, help="Directory to load/save language model for 5' UTR (for codon resolution)")
    p_train.add_argument("--model_dir3", default=None, help="Directory to load/save language model for 3' UTR (for codon resolution)")
    p_train.add_argument("--lm", default=None, help="Language model for nucleotide resolution (e.g., 'rinalmo')")
    p_train.add_argument("--lm5", default=None, help="Language model for 5' UTR when using codon resolution")
    p_train.add_argument("--lm3", default=None, help="Language model for 3' UTR when using codon resolution")
    p_train.add_argument("--species_proxy5", default=None, help="Species proxy for 5' UTR language model (required if using speciesLM)")
    p_train.add_argument("--species_proxy3", default=None, help="Species proxy for 3' UTR language model (required if using speciesLM)")
    p_train.add_argument("--epoch", type=int, default=50, help="Maximum number of training epochs (default: 50)")
    p_train.add_argument("--batch_size", type=int, default=32, help="Batch size for training (default: 32)")
    p_train.add_argument("--split", type=float, default=0.7, help="Training set size as fraction of total data (default: 0.7)")
    p_train.add_argument("--learning", type=float, default=0.0005, help="Learning rate (default: 0.0005)")
    p_train.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for regularization (default: 0.01)")
    p_train.add_argument("--warmup", type=int, default=5, help="Number of warmup steps for learning rate (default: 5)")
    p_train.add_argument("--alpha", type=float, default=0.1, help="Weight for count loss (default: 0.1)")
    p_train.add_argument("--beta", type=float, default=None, help="Weight for profile loss (default: 4.5 for nucleotide resolution, 1.0 for codon resolution)")
    p_train.add_argument("--utr_pool", type=float, default=0.3, help="Pooling factor for UTR regions in codon resolution (default: 0.3)")
    p_train.add_argument("--ablate_utr5", action="store_true", default=False, help="Ablate 5'UTR: zero out 5'UTR embedding so only CDS and 3'UTR sequence content is used")
    p_train.add_argument("--ablate_utr3", action="store_true", default=False, help="Ablate 3'UTR: zero out 3'UTR embedding so only CDS and 5'UTR sequence content is used")
    p_train.add_argument("--id", default=None, help="Optional descriptive identifier for this training run")
    p_train.add_argument("--verbose", type=int, default=1, help="Verbosity level: 0 for warnings only, 1+ for info (default: 1)")
    p_train.set_defaults(func="train")

    # SUBCOMMAND: interpret
    p_interp = subparsers.add_parser("interpret", help="Interprete and analyze RiboNet")
    p_interp.add_argument("--project_dir", required=True, help="Directory containing trained RiboNet model and outputs")
    p_interp.add_argument("--n_batches", type=int, required=True, help="Number of batches to divide interpretation task into")
    p_interp.add_argument("--scoring", type=str, default="ISM", help="Scoring method for feature importance (default: ISM for in silico mutagenesis)")
    p_interp.add_argument("--save_distribution", action="store_true", required=False, help="Save distribution of importance scores")
    p_interp.add_argument("--verbose", type=int, default=1, help="Verbosity level: 0 for warnings only, 1+ for info (default: 1)")
    p_interp.add_argument("--batch_size", type=int, default=None, help="Batch size to use for inference. If not set, defaults to four times the training batch size.")
    p_interp.set_defaults(func="interpret")

    # SUBCOMMAND: score_batch
    p_score = subparsers.add_parser("score_batch", help="Scores a batch of sequences using a trained RiboNet model. Run 'interpret' first to generate the batches.")
    p_score.add_argument("--project_dir", required=True, help="Directory containing trained RiboNet model and batch data")
    p_score.add_argument("--batch", type=int, default=1, help="Batch number to score (default: 1)")
    p_score.add_argument("--verbose", type=int, default=1, help="Verbosity level: 0 for warnings only, 1+ for info (default: 1)")
    p_score.set_defaults(func="score_batch")

    # SUBCOMMAND: report
    p_report = subparsers.add_parser("report", help="Generate interpretation report of scored RiboNet")
    p_report.add_argument("--project_dir", required=True, help="Directory containing scored RiboNet results")
    p_report.add_argument("--control", type=str, default=None, help="Name of control condition for comparison")
    p_report.add_argument("--threads", type=int, default=1, help="Number of threads for parallel processing (default: 1)")
    p_report.add_argument("--motif_db", type=str, default="default", help="Path to motif database or 'default' for built-in database (default: default)")
    p_report.add_argument("--diffsites", type=int, default=5000, help="Number of top differentially occupied sites to analyze (default: 5000)")
    p_report.add_argument("--pwind", type=int, default=30, help="Window size around differentially occupied sites in nucleotides (default: 30)")
    p_report.add_argument("--max_seqlets_per_metacluster", type=int, default=1000, help="Maximum number of seqlets per metacluster passed to TF-MoDISco (default: 1000)")
    p_report.add_argument("--filename", type=str, default="interpretation.pdf", help="Output filename for interpretation report (default: interpretation.pdf)")
    p_report.add_argument("--verbose", type=int, default=1, help="Verbosity level: 0 for warnings only, 1+ for info (default: 1)")
    p_report.set_defaults(func="report")

    args = parser.parse_args()

    # Must run before importing modules that load pyarrow/reportlab (see _preload_libz).
    _preload_libz()

    os.makedirs(args.project_dir, exist_ok=True)

    config_logging(args.project_dir, args.verbose)

    log.info("RiboNet {} run with command:".format(__version__))
    log.info(" ".join(sys.argv))

    # DISPATCH
    if args.func == "preprocess":
        # Validate that the key input files exist before starting the (long) preprocessing pipeline
        for label, path in [("--sample_file", args.sample_file),
                            ("--rna_counts",  args.rna_counts),
                            ("--genome_fasta", args.genome_fasta),
                            ("--gff",          args.gff)]:
            if not os.path.isfile(path):
                err_message = f"Input file specified by {label} not found: {path}"
                log.error(err_message)
                raise FileNotFoundError(err_message)
        for opt_label, opt_path in [("--utr5annot", args.utr5annot),
                                    ("--utr3annot", args.utr3annot)]:
            if opt_path is not None and not os.path.isfile(opt_path):
                err_message = f"Input file specified by {opt_label} not found: {opt_path}"
                log.error(err_message)
                raise FileNotFoundError(err_message)

        if args.fixed_5offsets is not None:
            # Validate fixed_5offsets format
            try:
                offsets = {}
                for item in args.fixed_5offsets.split(","):
                    length, offset = item.split("-")
                    offsets[int(length)] = int(offset)
                args.fixed_5offsets = offsets
            except Exception as e:
                err_message = ("Invalid format for --fixed_5offsets. "
                               "Expected format: readlength-offset pairs separated by commas "
                               "(e.g., `--fixed_5offsets 27-15,28-15,29-16,30-16`). "
                               f"Got: '{args.fixed_5offsets}'. Error: {e}")
                log.error(err_message)
                raise ValueError(err_message)

        preprocess_inputs(
            args.sample_file, args.rna_counts,
            args.genome_fasta, args.gff, args.project_dir,
            args.threads,
            spikein=args.spikein,
            spikein_absolute_ribosomes=args.spikein_absolute_ribosomes,
            maxlength=args.maxlength, mindensity=args.mindensity,
            utr5annot=args.utr5annot, utr3annot=args.utr3annot,
            maxutr5length=args.maxutr5length, maxutr3length=args.maxutr3length,
            fixed_5offsets=args.fixed_5offsets
        )

    elif args.func == "train":
        if args.lm is not None and (args.lm5 is not None or args.lm3 is not None):
            err_message = "Either --lm or --lm5/--lm3 can be specified, not both."
            logging.error(err_message)
            raise ValueError(err_message)

        if args.lm5 is not None or args.lm3 is not None:
            lms = {
                "UTR5":  args.lm5,
                "UTR3": args.lm3
            }
        else:
            lms = args.lm

        if args.species_proxy5 is not None or args.species_proxy3 is not None:
            species_proxy = {
                "UTR5": args.species_proxy5,
                "UTR3": args.species_proxy3
            }
        else:
            species_proxy = None

        if args.model_dir is not None and (args.model_dir5 is not None or args.model_dir3 is not None):
            err_message = "Either --model_dir or --model_dir5/--model_dir3 can be specified, not both."
            logging.error(err_message)
            raise ValueError(err_message)

        if args.model_dir5 is not None or args.model_dir3 is not None:
            model_dir = {
                "UTR5":  args.model_dir5,
                "UTR3": args.model_dir3
            }
        else:
            model_dir = args.model_dir

        train_ribonet(args.project_dir, args.resolution,
                      cache_dir=args.cache_dir, model_dir=model_dir,
                      lm=lms, species_proxy=species_proxy,
                      epoch=args.epoch, batch_size=args.batch_size, split=args.split,
                      learning=args.learning, weight_decay=args.weight_decay,
                      warmup=args.warmup, alpha=args.alpha, beta=args.beta,
                      utr_pool=args.utr_pool, id=args.id,
                      ablate_utr5=args.ablate_utr5, ablate_utr3=args.ablate_utr3)

    elif args.func == "interpret":
        interpret_ribonet(args.project_dir, args.n_batches, args.scoring, args.save_distribution, inference_batch_size=args.batch_size)

    elif args.func == "score_batch":
        score_batch(args.project_dir, args.batch)

    elif args.func == "report":
        report_interpretation(args.project_dir, args.control, args.threads, args.motif_db,
                              args.diffsites, args.pwind, args.max_seqlets_per_metacluster, args.filename)

if __name__ == '__main__':
    main()
