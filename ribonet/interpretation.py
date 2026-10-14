import itertools
import warnings
import torch
import os
import shutil
import gc
import json
import logging
import pandas as pd
import numpy as np
import math
import random
import pyarrow as pa
import pyarrow.parquet as pq
from torch.utils.data import DataLoader
from functools import partial
from scipy import stats
import logomaker
import h5py as h5
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import seaborn as sns
from .models.codon_model import RiboNet as CodonRiboNet
from .models.nucleotide_model import RiboNet as NtRiboNet
from .models.ribodataset import RiboDataset_cod, RiboDataset_nt
from .config import CDS_Config, Nuc_Config
from .utils.seq_utils import reshape_training_data, print_seq, Cdict
from .utils.training_utils import make_prediction
from .utils.model_utils import autoselect_workers, load_lm
from .utils.plot_utils import plot_pos_seqlets
from .modiscolite_2_2_1_custom.core import Seqlet
from .modiscolite_2_2_1_custom.tfmodisco import TFMoDISco
from .modiscolite_2_2_1_custom.io import save_hdf5
from .modiscolite_2_2_1_custom.report import report_motifs
from .utils.report_utils import *
from .utils.parallel import parallel_map


log = logging.getLogger(__name__)

class RiboInterpretation: 

    def __init__(self, project_dir):
        '''
        Initialize RiboInterpretation class.

        Parameters
        ----------
        project_dir : str
            Path to the project directory containing training data and outputs.

        Returns
        -------
        None
        '''
        log.info("Initializing interpretation...")

        self.project_dir = os.path.join(project_dir) 
        if not os.path.isdir(self.project_dir):
            err_message = f"Provided project directory {self.project_dir} does not exist. It should have been created during data preprocessing."
            log.error(err_message)
            raise NotADirectoryError(err_message)
        
        self.preprocess_dir = os.path.join(self.project_dir, "preprocessing")
        if not os.path.isdir(self.preprocess_dir):
            err_message = f"Training directory {self.preprocess_dir} does not exist. It should have been created during data preprocessing."
            log.error(err_message)
            raise NotADirectoryError(err_message)
        
        self.training = os.path.join(self.project_dir, "training")
        if not os.path.isdir(self.training):
            err_message = f"Training output directory {self.training} does not exist. It should have been created during training."
            log.error(err_message)
            raise NotADirectoryError(err_message)
        
        self.interpretation_dir = os.path.join(self.project_dir, "interpretation")
        os.makedirs(self.interpretation_dir, exist_ok=True)
        
        self.sample_file = pd.read_csv(os.path.join(self.preprocess_dir, "sample_file.txt"), sep="\t")
        self.batch_file = os.path.join(self.interpretation_dir, "batches.tsv")

        # Load batch splits
        if os.path.exists(self.batch_file):
            self.batch_splits = pd.read_csv(self.batch_file, sep="\t", index_col="batch_id")["genes"].map(lambda x: x.split(",")).to_dict()

        config_path = os.path.join(self.project_dir, "config_ribonet.json")
        if not os.path.isfile(config_path):
            err_message = (f"Model configuration file not found: {config_path}. "
                           f"Ensure the 'train' step has been completed successfully.")
            log.error(err_message)
            raise FileNotFoundError(err_message)
        with open(config_path, 'r') as f:
            config = json.load(f)
        self.inference_batch_size = config["batch"] * 4 if "inference_batch_size" not in config.keys() else config["inference_batch_size"]
        self.seq_resolution = config["seq_resolution"]
        self.seq_len = config["seq_len"]
        self.utr_pool_size = config.get("utr_pool_size", None)
        self.utr_pool = config.get("utr_pool", None)
        self.utr_len = config.get("utr_len", None)
        self.cache_dir = config["cache_dir"]
        self.model_dir = config["model_dir"]
        self.alpha = config["alpha"]
        self.beta = config["beta"]
        self.lm = config["lm"]
        self.species_proxy = config.get("species_proxy", None) 
        self.left_special_tokens = config.get("left_special_tokens", None)
        self.right_special_tokens = config.get("right_special_tokens", None)
        self.fixed_length = config.get("fixed_length", None)
        self.window_size = config.get("window_size", None)
        self.overlap = config.get("overlap", None)
        self.decoder = config.get("decoder", None)
        self.use_annotation_emb = config.get("use_annotation_emb", True)
        self.dataloader_workers = autoselect_workers()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lm = load_lm(self.lm, cache_dir=self.cache_dir, model_dir=self.model_dir, species_proxy=self.species_proxy)
        self.save_distribution = config.get("save_distribution", None)
        self.scoring = config.get("scoring", None)

        x_inputs_path = os.path.join(self.preprocess_dir, "x_inputs.parquet")
        y_outputs_path = os.path.join(self.preprocess_dir, "y_outputs.parquet")
        if not os.path.isfile(x_inputs_path):
            err_message = (f"Preprocessed input file not found: {x_inputs_path}. "
                           f"Run the 'preprocess' step first.")
            log.error(err_message)
            raise FileNotFoundError(err_message)
        if not os.path.isfile(y_outputs_path):
            err_message = (f"Preprocessed output file not found: {y_outputs_path}. "
                           f"Run the 'preprocess' step first.")
            log.error(err_message)
            raise FileNotFoundError(err_message)

        self.x_raw = pd.read_parquet(x_inputs_path)
        self.y_raw = pd.read_parquet(y_outputs_path)

        # prepare data
        if self.seq_resolution == "codon":
            self.seq_args = {
                "maxcodonlen": self.seq_len,
                "maxutr5len": self.utr_len[0],
                "maxutr3len": self.utr_len[1],
                "utr5poolsize": self.utr_pool_size[0],
                "utr3poolsize": self.utr_pool_size[1]
            }
        else:
            self.seq_args = {
                "maxseqlen": self.seq_len
            }
        self.x_c, self.y_mat, _ = reshape_training_data(self.x_raw, self.y_raw, seq_resolution=self.seq_resolution, lm=self.lm, utrpool=self.utr_pool, seq_args=self.seq_args)

        log.info("Interpretation initialized.")

    def score_importances(self, scoring, n_batches=1, save_distribution=False, inference_batch_size=None):
        '''
        Score feature importances using the specified scoring method.

        Parameters
        ----------
        scoring : str
            Scoring method to use. Currently only "ISM" (In Silico Mutagenesis) is implemented.
        n_batches : int, default=1
            Number of batches to split the genes into for parallel processing.
        save_distribution : bool, default=False
            Whether to save the full distribution of predictions.
        inference_batch_size : int, optional
            Batch size to use for inference. If None, defaults to four times the training batch size.

        Returns
        -------
        None
        '''

        # Store save_distribution for downstream functions
        self.save_distribution = save_distribution
        self.scoring = scoring
        if inference_batch_size is not None:
            self.inference_batch_size = inference_batch_size
        if scoring != "ISM":
            err_message = f"Scoring method '{scoring}' is not supported. Currently only 'ISM' (In Silico Mutagenesis) is implemented."
            log.error(err_message)
            raise NotImplementedError(err_message)
        path = os.path.join(self.project_dir, "config_ribonet.json")

        with open(path) as f:
            config = json.load(f)

        config.update({
            "scoring": scoring,
            "save_distribution": save_distribution,
            "inference_batch_size": self.inference_batch_size
        })

        with open(path, "w") as f:
            json.dump(config, f, indent=4)
        # for now ignore scoring, only do ISM
        self.split_gene_batches(n_batches)
        if n_batches > 1:
            self.write_slurm_array()
        else:
            self.data_batch = 0  # single batch: scores are saved as contrasts_batch_0.parquet
            self._score()


    def split_gene_batches(self, n_batches):
        '''
        Split genes into batches for parallel processing.

        Parameters
        ----------
        n_batches : int
            Number of batches to create.

        Returns
        -------
        None
            Saves batch information to batches.tsv file.
        '''
        self.n_batches = n_batches
        n_genes_per_batch = len(self.x_raw) / n_batches
        genes_per_batch = []
        for i in range(n_batches):
            start_idx = i * n_genes_per_batch
            end_idx = (i + 1) * n_genes_per_batch
            genes = self.x_raw.index[math.ceil(start_idx):math.ceil(end_idx)]
            genes_string = ",".join(genes)
            genes_per_batch.append(genes_string)
        
        batches = pd.DataFrame({"batch_id": range(n_batches),
                                "genes": genes_per_batch}).set_index("batch_id")
        batches.to_csv(self.batch_file, sep="\t")

    def write_slurm_array(self):
        '''
        Write a SLURM array job script for batch processing.

        Parameters
        ----------
        None

        Returns
        -------
        None
            Saves SLURM script file to the project directory.
        '''
        # create header
        header = f"""#!/bin/bash
#SBATCH --job-name=ribonet_interpretation
#SBATCH --output ribonet_interpretation%A.out
#SBATCH --cpus-per-task 16
#SBATCH --mem 32G
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --array=0-{self.n_batches - 1}\n\n"""
        
        source = f"source {os.path.expanduser('~/.bashrc')}\n\n"

        # get conda env
        try:
            env=os.environ['CONDA_DEFAULT_ENV']
        except KeyError:
            env=""
        conda_env = f"conda activate {env}\n\n"

        cmd = f'ribonet score_batch \\\n\t--project_dir "{os.path.abspath(self.project_dir)}" \\\n\t--batch $SLURM_ARRAY_TASK_ID \n'

        with open(os.path.join(self.project_dir, "ribonet_array.sbatch"), 'w') as f:
            f.write(header + source + conda_env + cmd)
        
        log.info(f"SLURM array job script written to {os.path.join(self.project_dir, 'ribonet_array.sbatch')}. Complete filling this script based on your cluster's requirements and run it with 'sbatch' to start batch processing.")

    def load_batch(self, task_id):
        '''
        Load a specific batch of genes for processing.

        Parameters
        ----------
        task_id : int
            The batch identifier to load.

        Returns
        -------
        None
            Updates x_raw and y_raw with the selected batch of genes.
        '''
        if not hasattr(self, "batch_splits"):
            err_message = "Batch splits not found. Please run split_gene_batches first."
            log.error(err_message)
            raise AttributeError(err_message)
        
        genes = self.batch_splits[task_id]
        
        self.x_raw = self.x_raw.loc[genes, :]
        self.y_raw = self.y_raw.loc[genes, :]

        self.x_c = self.x_c.loc[genes, :]
        self.y_mat = self.y_mat[self.x_c.index.get_indexer(genes), :]

        self.data_batch = task_id


    def build_model(self):
        '''
        Instantiate the RiboNet model, move it to the active device, and load the
        trained weights from ``training/ribonet.pt``.

        This is called internally by :meth:`_score`, but is also exposed publicly so
        that custom predictions (e.g. on designed/mutated sequences) can be run via
        :meth:`forward_pass` without going through the full ``_score`` pipeline. After
        calling this, ``self.model`` is set to an eval-mode model on ``self.device``.

        Returns
        -------
        None
        '''
        # prepare model
        args = self.__dict__.copy()

        if self.seq_resolution == 'codon':
            self.model_config = CDS_Config(seq_len=self.seq_len,
                                        utr_len=self.utr_len,
                                        utr_pool_size=self.utr_pool_size,
                                        args=args)
        elif self.seq_resolution == 'nucleotide':
            self.model_config = Nuc_Config(seq_len=self.seq_len,
                                        args=args)

        extract_model = lambda l: l[2] if l is not None else None
        if self.seq_resolution == 'nucleotide':
            self.model = NtRiboNet(extract_model(self.lm),
                                    self.model_config,
                                    n_tracks = self.y_mat.shape[1])
        else:
            self.model = CodonRiboNet(extract_model(self.lm["UTR5"]),
                                        extract_model(self.lm["UTR3"]),
                                        self.model_config,
                                        n_tracks = self.y_mat.shape[1])

        self.model.to(self.device)

        # Load trained model weights
        model_weights_path = os.path.join(self.project_dir, "training", "ribonet.pt")
        if not os.path.isfile(model_weights_path):
            err_message = (f"Trained model weights not found at {model_weights_path}. "
                           f"Ensure the 'train' step has been completed successfully.")
            log.error(err_message)
            raise FileNotFoundError(err_message)
        self.model.load_state_dict(torch.load(model_weights_path, map_location=self.device))
        self.model.eval()

    def _score(self, wsize=1, nrand=3, gene_dict=None, onthefly=False, filename=None):
        '''
        Perform in silico mutagenesis and score the effects on model predictions.

        Parameters
        ----------
        wsize : int, default=1
            Window size for mutagenesis. Must be odd.
        nrand : int, default=3
            Number of random mutations per window position.
        gene_dict : dict, optional
            Dictionary mapping gene names to specific positions to mutate.
            If None, all genes are processed.
        onthefly : bool, default=False
            If True, return results immediately without saving to file.
        filename : str, optional
            Path to save the output parquet file. If None, uses default location.

        Returns
        -------
        pd.DataFrame or None
            If onthefly=True, returns DataFrame with mutation effects.
            Otherwise, saves results to file and returns None.
        '''
        log.info('Device used for interpretation: %s', self.device)

        # prepare model (sets self.model, self.model_config, loads trained weights)
        self.build_model()

        batch_str = f"Batch {self.data_batch}: " if hasattr(self, "data_batch") else ""

        log.info(batch_str + f"Starting unmutagenized predictions.")

        if self.seq_resolution == 'nucleotide':
            dataset = RiboDataset_nt(np.stack(self.x_c["seq"].values),
                                        np.stack(self.x_c["rna"].values), 
                                        np.stack(self.x_c["annotation"].values),
                                        self.y_mat)
        else:
            dataset = RiboDataset_cod(np.stack(self.x_c["cds"].values),
                                        np.stack(self.x_c["utr5"].values),
                                        np.stack(self.x_c["utr3"].values),
                                        np.stack(self.x_c["rna"].values),
                                        np.stack(self.x_c["posemb"].values),
                                        self.y_mat)

        self.dataloader = DataLoader(dataset, batch_size=self.inference_batch_size, shuffle=False, pin_memory=True, num_workers=self.dataloader_workers)

        out_tensors = make_prediction(self.model, self.dataloader, self.device)
        profile_pred, count_pred = [s.cpu().numpy() for s in out_tensors]
        profile_pred = np.log(profile_pred)  # convert to log-probabilities
        colnames_iter = [self.y_raw["cds"].columns, ["profile_pred", "count_pred"]]
        col_index = pd.MultiIndex.from_product(colnames_iter, names=["cond", "info"])
        controldf = pd.DataFrame([[x for c in range(count_pred.shape[1]) for x in [profile_pred[r,c,:], count_pred[r,c]]] for r in range(count_pred.shape[0])],
                                index = self.y_raw.index, columns = col_index)
        
        log.info(batch_str + "Unmutagenized predictions done.")

        log.info(batch_str + "Creating output directory and file.")
        
        # Create output file
        if onthefly or filename or self.save_distribution:
            seqdata = pa.Table.from_pandas(pd.DataFrame(columns=["gene","annot","pos_seq","ori_seq","to_seq","dcount","dprofile"]), preserve_index=False,
                                            schema=pa.schema([('gene', pa.string()),('annot', pa.string()),('pos_seq', pa.list_(pa.uint16())),('ori_seq', pa.string()),
                                                                ('to_seq', pa.string()),('dcount', pa.list_(pa.float32())),('dprofile', pa.list_(pa.list_(pa.float32())))]))
        else:
            seqdata = pa.Table.from_pandas(pd.DataFrame(columns=["gene","annot","pos_seq","ori_seq","to_seq","dcount"]), preserve_index=False,
                                            schema=pa.schema([('gene', pa.string()),('annot', pa.string()),('pos_seq', pa.list_(pa.uint16())),('ori_seq', pa.string()),
                                                                ('to_seq', pa.string()),('dcount', pa.list_(pa.float32()))]))
        if onthefly:
            outlist = []
        else:
            if filename:
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                # this only runs if the function is called with a specific filename, otherwise it defaults to saving in the interpretation directory with the batch number if applicable
                pq_file = os.path.join(filename)
            else:
                outpath = os.path.join(self.interpretation_dir, "full_contrasts/")
                os.makedirs(outpath, exist_ok=True)
                # create parquet file in interpretation/ with name contrasts_batch_{task_id}.parquet or contrasts_batch.parquet if no batching
                file_batch_number = f"_{self.data_batch}" if hasattr(self, "data_batch") else ""
                pq_file = os.path.join(outpath, f"contrasts_batch" + file_batch_number + ".parquet")
            pqwriter = pq.ParquetWriter(pq_file, seqdata.schema)

        # Subset genes if needed
        if gene_dict:
            x_set = self.x_raw.loc[gene_dict.keys()]
            y_set = self.y_raw.loc[gene_dict.keys()]
        else:
            x_set = self.x_raw
            y_set = self.y_raw

        if self.scoring == "ISM":
            log.info(batch_str + f"Starting mutagenesis with window size {wsize} and {nrand} random mutations per window.")
            assert wsize % 2 == 1

            # iterate over each gene in the dataset and perform the mutagenesis and forward pass
            for i, (x_row, y_row) in enumerate(zip(x_set.iterrows(), y_set.iterrows())):
                gene_name, gene = x_row
                _, gene_y = y_row

                if (i + 1) % 20 == 0:
                    log.info(batch_str + "Mutating gene %s/%s.", i+1, len(x_set))

                gene_mutations, annotations = self.mutagenize(gene_name, gene, wsize=wsize, nrand=nrand, gene_pos=gene_dict[gene_name] if gene_dict else None)

                x_c, y_mat, _ = reshape_training_data(gene_mutations, 
                                                      pd.DataFrame([gene_y] * len(gene_mutations), index=pd.RangeIndex(len(gene_mutations))),
                                                      seq_resolution=self.seq_resolution,
                                                      lm=self.lm,
                                                      utrpool=self.utr_pool,
                                                      seq_args=self.seq_args)
                profile_rand, count_rand = self.forward_pass(x_c, y_mat)

                ycount = np.tile(controldf.loc[gene_name].xs("count_pred", level=1, drop_level=True).values, (annotations.shape[0], 1))
                annotations["dcount"] =  [row for row in (ycount - count_rand)]
                if self.save_distribution or onthefly or filename:
                    yprof = np.tile(np.stack(controldf.loc[gene_name].xs("profile_pred", level=1, drop_level=True).values), (annotations.shape[0], 1, 1))
                    annotations["dprofile"] =  [row.tolist() for row in (yprof - profile_rand)]
                pa_temp = pa.Table.from_pandas(annotations, preserve_index=False, schema=seqdata.schema)
                if onthefly:
                    outlist.append(pa_temp.to_pandas())
                else:
                    pqwriter.write_table(pa_temp)
        else:
            log.error("Currently only ISM is implemented.")
            raise NotImplementedError("Currently only ISM is implemented.")

        if onthefly:
            return pd.concat(outlist, axis=0, ignore_index=True)
        else:
            pqwriter.close()
            log.info(batch_str + "All predictions finished and saved.")
            # Free the CUDA model from self so that subsequent parallel_map
            # calls don't try to pickle CUDA tensors into forked/spawned workers.
            for attr in ("model", "model_config", "dataloader"):
                if hasattr(self, attr):
                    delattr(self, attr)


    def mutagenize(self, gene_name, gene, wsize=1, nrand=3, gene_pos=None):
        '''
        Generate mutated sequences for a single gene.

        Parameters
        ----------
        gene_name : str
            Name of the gene to mutagenize.
        gene : pd.Series
            Gene information including UTR and CDS sequences.
        wsize : int, default=1
            Window size for mutagenesis. Must be odd.
        nrand : int, default=3
            Number of random mutations per window position.
        gene_pos : list, optional
            Specific positions to mutagenize. If None, all positions are considered.

        Returns
        -------
        tuple of pd.DataFrame
            Two DataFrames: (mutated_sequences, mutation_annotations)
        '''
        if wsize % 2 != 1:
            err_message = f"Window size must be odd, got {wsize}."
            log.error(err_message)
            raise ValueError(err_message)
                
        utr5, cds, utr3 = gene.loc["utr5"], gene.loc["cds"], gene.loc["utr3"]
        seq = utr5[:-3] + cds + utr3[3:]

        # set the start to the annotated utr start, ie. ignore upstream nc for ISM
        start = self.utr_len[0] - int(gene["utr5_len"])
        # set the end to the annotated utr end, ie. ignore downstream nc for ISM
        end = self.utr_len[0] + int(gene["cds_len"]) + int(gene["utr3_len"]) - wsize + 1
        # NOTE on position spaces used throughout this module: below, "pos" (= i - start) is the
        # ISM "mutation-index" position, 0-based and anchored at the gene's own annotated UTR5 start
        # (padding/non-coding flanks excluded). This is the space used by pos_seq in the contrasts
        # parquet files and by diffsites.csv. It is identical to the nucleotide-resolution model
        # PROFILE position (since reshape_training_data trims the nucleotide "seq"/"annotation"/y_mat
        # to the real per-gene UTR5 start with no left padding), but it differs from the codon-
        # resolution PROFILE position, whose UTR5/UTR3 bins are pooled over a fixed-length, left-
        # padded window shared by all genes. Wherever both spaces coexist, functions below convert
        # via a "seqmapper" (profile position -> mutation-index position); see e.g.
        # _sites_scores_1gene_pausing_codon and _extract_sites_scores.
        seq = np.array(list(seq), dtype='U1')

        annot, pos_seq, ori_seq, to_seq = [], [], [], []
        mutated_seqs = []
        nuc_list = ["A","C","G","T"]
        all_possible_combinations = set(itertools.product(nuc_list, repeat=wsize))
        nrand = min(int(nrand), len(all_possible_combinations) - 1)

        # iterate over every relevant nucleotide window and mutate
        for i in range(start, end):
            pos = i - start
            # Subset for gene_pos if it exists
            if gene_pos and pos not in gene_pos:
                continue
            
            # create mutated sequences for window position i
            current_mutated_seqs = np.repeat(seq[np.newaxis, :], nrand, axis=0)
            mutation_window = all_possible_combinations - {tuple(seq[i:i+wsize])}
            mutation_window = tuple(random.sample(sorted(mutation_window), k=nrand))
            current_mutated_seqs[:, i:i+wsize] = mutation_window
            mutated_seqs.append(current_mutated_seqs)

            # create annotations
            region_annot = (
                "5UTR" if i < self.utr_len[0]
                else "CDS" if i < self.utr_len[0] + int(gene["cds_len"])
                else "3UTR"
            )
            pos = np.array([pos]) if wsize == 1 else np.arange(pos, pos + wsize)
            ori = "".join(seq[i:i + wsize])
            to = ["".join(mutation) for mutation in mutation_window]

            # append them nrand times to match the number of mutated sequences
            annot.extend([region_annot] * nrand)
            pos_seq.extend([pos] * nrand)
            ori_seq.extend([ori] * nrand)
            to_seq.extend(to)

        mutated_seqs = np.concat(mutated_seqs, axis=0)
        utr5_mutated = mutated_seqs[:, :(self.utr_len[0]+3)]
        cds_mutated = mutated_seqs[:, self.utr_len[0]:(self.utr_len[0]+int(gene["cds_len"]))]
        utr3_mutated = mutated_seqs[:, (self.utr_len[0]+int(gene["cds_len"])-3):]

        return pd.DataFrame({
                "utr5": ["".join(mut_gene) for mut_gene in utr5_mutated],
                "cds": ["".join(mut_gene) for mut_gene in cds_mutated],
                "utr3": ["".join(mut_gene) for mut_gene in utr3_mutated],
                "utr5_len": gene.loc["utr5_len"],
                "cds_len": gene.loc["cds_len"],
                "utr3_len": gene.loc["utr3_len"],
                "rna": [gene.loc["rna"]] * mutated_seqs.shape[0],
                "longest_utr5_annot": gene.loc["longest_utr5_annot"],
                "longest_utr3_annot": gene.loc["longest_utr3_annot"],
                "gene": gene_name
            }), pd.DataFrame({
                "annot": annot,
                "pos_seq": pos_seq,
                "ori_seq": ori_seq,
                "to_seq": to_seq,
                "gene": gene_name
            })

    
    def forward_pass(self, x_c, y_mat):
        '''
        Perform forward pass through the model to get predictions.

        Parameters
        ----------
        x_c : pd.DataFrame
            Input sequences for prediction.
        y_mat : np.ndarray
            Target output matrix.

        Returns
        -------
        tuple of np.ndarray
            (profile_predictions, count_predictions)
        '''
        # record the absolute changes of SNPs
        if self.seq_resolution == "codon":
            dataset = RiboDataset_cod(np.stack(x_c["cds"].values),
                                              np.stack(x_c["utr5"].values),
                                              np.stack(x_c["utr3"].values),
                                              np.stack(x_c["rna"].values),
                                              np.stack(x_c["posemb"].values),
                                              y_mat)
        else:
            dataset = RiboDataset_nt(np.stack(x_c["seq"].values),
                                        np.stack(x_c["rna"].values), 
                                        np.stack(x_c["annotation"].values),
                                        y_mat)
        dataloader = DataLoader(dataset, batch_size=self.inference_batch_size, shuffle=False, pin_memory=True, num_workers=self.dataloader_workers)
        profile_rand, count_rand = make_prediction(self.model, dataloader, self.device)
        profile_rand, count_rand = profile_rand.numpy(force=True), count_rand.numpy(force=True)
        profile_rand = np.log(profile_rand)  # convert to log-probabilities

        return profile_rand, count_rand
    
    def _start_stop_metagene(self, conds):
        '''
        Generate metagene plots around start and stop codons.

        Parameters
        ----------
        conds : list of str
            List of condition names to plot.

        Returns
        -------
        None
            Adds plots to the current report.
        '''
        pdf_path = os.path.join(self.interpretation_dir, "counts", "start_stop_metagene.pdf")
        log.info("Computing start/stop metagene plots for %d condition(s).", len(conds))
        ylim = []
        fig, axs = plt.subplots(len(conds),2, figsize=(10, 1.4*len(conds)))
        for n,cond in enumerate(conds):
            # Open the per-region count archives inside a context manager so their file
            # handles and read buffers are released after this condition is processed,
            # instead of leaking one NpzFile per region per condition until the report
            # finalises.
            with np.load(os.path.join(self.interpretation_dir,"counts", "5UTR_" + cond + "_modisco.npz")) as utr5npz, \
                 np.load(os.path.join(self.interpretation_dir,"counts", "CDS_" + cond + "_modisco.npz")) as cdsnpz, \
                 np.load(os.path.join(self.interpretation_dir,"counts", "3UTR_" + cond + "_modisco.npz")) as utr3npz:
                # Per-region arrays are in the ISM mutation-index space (see mutagenize / _prepare_modisco_1gene),
                # padded to each region's max length: 5UTR is left-padded (so the *last* elements are closest to
                # the CDS start), CDS/3UTR are right-padded (so the *first* elements are closest to their own
                # 5' end). Hence utr5[-53:-3] and cds[3:53]/cds[-53:-3] below correctly bracket the start/stop
                # codon boundary regardless of each gene's actual UTR/CDS length.
                utr5file = utr5npz["dcounts"]
                cdsfile = cdsnpz["dcounts"]
                utr3file = utr3npz["dcounts"]
                startdf = pd.DataFrame(columns=[str("%i-%i" % (s,m)) for s,m in zip(np.arange(-50,46,5),np.arange(-45,51,5))], index = self.x_raw.index, dtype=float)
                enddf = pd.DataFrame(columns=[str("%i-%i" % (s,m)) for s,m in zip(np.arange(-50,46,5),np.arange(-45,51,5))], index = self.x_raw.index, dtype=float)
                for i, (utr5,cds,utr3) in enumerate(zip(utr5file,cdsfile,utr3file)):
                    if (~np.isnan(cds)).sum()>0:
                        endutr5 = np.nansum(utr5[np.isnan(utr5).sum(-1)<4,:], -1)[-53:-3]
                        startcds = np.nansum(cds[np.isnan(cds).sum(-1)<4,:], -1)[3:53]
                        startdf.iloc[i] = np.add.reduceat(np.concatenate([np.full(50-endutr5.shape[0], np.nan),endutr5,startcds]), np.arange(0,100, 5), axis=0)/5
                        endcds = np.nansum(cds[np.isnan(cds).sum(-1)<4,:], -1)[-53:-3]
                        startutr3 = np.nansum(utr3[np.isnan(utr3).sum(-1)<4,:], -1)[3:53]
                        enddf.iloc[i] = np.add.reduceat(np.concatenate([endcds,startutr3,np.full(50-startutr3.shape[0], np.nan)]), np.arange(0,100, 5), axis=0)/5
            sns.barplot(startdf.mean(), ax=axs[n,0], color="#4b7b6a")
            axs[n,0].set_title(str("%s - Start codon" % (cond)))
            axs[n,0].tick_params(axis='x', labelrotation=90)
            sns.barplot(enddf.mean(), ax=axs[n,1], color="#4b7b6a")
            axs[n,1].set_title(str("%s - Stop codon" % (cond)))
            axs[n,1].tick_params(axis='x', labelrotation=90)
            ylim.append(max(axs[n,0].get_ylim()[1],axs[n,1].get_ylim()[1]))
            axs[n,0].set_ylabel("AvgScore")
            axs[n,1].tick_params(axis='y', which='both', labelleft=False)
            if n < (len(conds)-1):
                axs[n,0].tick_params(axis='x', which='both', labelbottom=False)
                axs[n,1].tick_params(axis='x', which='both', labelbottom=False)
        for ax in axs.flatten():
            ax.set_ylim(0, max(ylim))
        fig.tight_layout()
        fig.savefig(pdf_path)
        add_plot(fig)
        plt.close()
        return

    def _prepare_modisco_1gene(self, gene, nucl, conds, ctrl_cond, maxlen):
        '''
        Prepare TF-MoDISco input data for a single gene.

        Parameters
        ----------
        gene : str
            Gene identifier.
        nucl : list of str
            List of nucleotides (e.g., ["A", "C", "G", "T"]).
        conds : list of str
            List of condition names.
        ctrl_cond : dict or None
            Control condition mapping.
        maxlen : list of int
            Maximum lengths for [UTR5, CDS, UTR3] regions.

        Returns
        -------
        dict
            Dictionary containing prepared data for each region (5UTR, CDS, 3UTR).
        '''
        # Find batch
        batch_num = int([k for k,v in self.batch_splits.items() if gene in v][0])
        # Upload contrasts
        contrasts = pd.read_parquet(os.path.join(self.interpretation_dir, "full_contrasts", "contrasts_batch_%i.parquet" % batch_num), filters=[("gene", "==", gene)])
        # "pos_seq" (embedded in poslong below) is the ISM mutation-index position (see mutagenize).
        # dcounts are per-mutation scalars (total count effect), so no profile-resolution mapping is
        # ever needed for them, unlike the profile-based scores handled elsewhere in this module.
        contrast_per_pos = contrasts.loc[(ser:=contrasts["pos_seq"].explode()).index].assign(poslong=contrasts["ori_seq"]+ser.astype(str)+contrasts["to_seq"])
        dcounts = contrast_per_pos.loc[(ser:=contrast_per_pos["dcount"].explode()).index].assign(dcountval=ser, cond=list(self.y_raw["cds"].columns)*contrast_per_pos.shape[0]).pivot_table(index="poslong",columns="cond",values="dcountval",dropna=False).astype(np.float32)
        if ctrl_cond is not None:
            ctrls = [ctrl_cond[c] for c in conds]
            ddcounts = dcounts[conds].sub(dcounts[ctrls].values, axis=0).assign(pos=[int(s[1:-1]) for s in dcounts.index]).pivot_table(index="pos", aggfunc=lambda x: np.mean(x, axis=0),dropna=False)
        else:
            ddcounts = dcounts[conds].sub(dcounts[conds].mean(axis=1), axis=0).assign(pos=[int(s[1:-1]) for s in dcounts.index]).pivot_table(index="pos", aggfunc=lambda x: np.mean(x, axis=0),dropna=False)
        dcounts = dcounts[conds].assign(pos=[int(s[1:-1]) for s in dcounts.index]).pivot_table(index="pos", aggfunc=lambda x: np.max(np.abs(x), axis=0),dropna=False)
        # Record regsizes and seq
        regsizes = [int(self.x_raw.loc[gene,"utr5_len"]), int(self.x_raw.loc[gene,"cds_len"]), int(self.x_raw.loc[gene,"utr3_len"])]
        seq = contrast_per_pos["poslong"].apply(lambda x: [x[:1],x[1:-1]]).drop_duplicates().apply(lambda x: x[0]).values
        # Create ohe
        ohe = pd.get_dummies(seq)
        for b in [s for s in nucl if s not in ohe.columns]:
            ohe[b] = False
        ohe = ohe[nucl].values
        # Create input matrices for TF-Modisco (float32: source parquet scores are float32)
        attr_dcount = {cond:np.full((1,len(seq),4), np.nan, dtype=np.float32) for cond in conds}
        attr_ddcount = {cond:np.full((1,len(seq),4), np.nan, dtype=np.float32) for cond in conds}
        for cond in conds:
            attr_dcount[cond][0,ohe] = dcounts[cond].values
            attr_ddcount[cond][0,ohe] = ddcounts[cond].values
        # Pad positions until maxlen
        attr = {}
        padding = np.full((1,maxlen[0]-regsizes[0],4), np.nan, dtype=np.float32)
        attr["5UTR"] = [np.concatenate([np.tile(np.array([1,0,0,0]),(maxlen[0]-regsizes[0],1)),ohe[:(regsizes[0]+5),:]],axis=0)[np.newaxis,:,:],
                        {k:[np.concatenate([padding,attr_dcount[k][:,:(regsizes[0]+5),:]],axis=1),
                            np.concatenate([padding,attr_ddcount[k][:,:(regsizes[0]+5),:]],axis=1)] for k in attr_dcount.keys()}]
        padding = np.full((1,maxlen[1]-regsizes[1],4), np.nan, dtype=np.float32)
        attr["CDS"] = [np.concatenate([ohe[(regsizes[0]):(regsizes[0]+regsizes[1]),:],np.tile(np.array([1,0,0,0]),(maxlen[1]-regsizes[1],1))],axis=0)[np.newaxis,:,:],
                    {k:[np.concatenate([attr_dcount[k][:,(regsizes[0]):(regsizes[0]+regsizes[1]),:],padding],axis=1),
                        np.concatenate([attr_ddcount[k][:,(regsizes[0]):(regsizes[0]+regsizes[1]),:],padding],axis=1)] for k in attr_dcount.keys()}]
        padding = np.full((1,maxlen[2]-regsizes[2],4), np.nan, dtype=np.float32)
        attr["3UTR"] = [np.concatenate([ohe[(regsizes[0]+regsizes[1]-5):,:],np.tile(np.array([1,0,0,0]),(maxlen[2]-regsizes[2],1))],axis=0)[np.newaxis,:,:],
                        {k:[np.concatenate([attr_dcount[k][:,(regsizes[0]+regsizes[1]-5):,:],padding],axis=1),
                            np.concatenate([attr_ddcount[k][:,(regsizes[0]+regsizes[1]-5):,:],padding],axis=1)] for k in attr_dcount.keys()}]
        
        return attr
    
    def _prepare_modisco_inputs(self, conds, ctrl_cond, threads, nucl=["A","C","G","T"]):
        '''
        Generate input files for TF-MoDISco based on the full contrasts.

        Parameters
        ----------
        conds : list of str
            List of condition names.
        ctrl_cond : dict or None
            Control condition mapping.
        threads : int
            Number of threads for parallel processing.
        nucl : list of str, default=["A","C","G","T"]
            List of nucleotides.

        Returns
        -------
        None
            Saves prepared input files as .npz files for each region and condition.
        '''
        missing_npz = [f"{region}_{cond}_modisco.npz" for region in ["5UTR","CDS","3UTR"] for cond in conds
                       if not os.path.exists(os.path.join(self.interpretation_dir, "counts", region + "_" + cond + "_modisco.npz"))]
        if missing_npz:
            log.info("Missing %d modisco input file(s), recomputing: %s", len(missing_npz), ", ".join(missing_npz))
            # Define maxlen
            maxlen = [int(self.x_raw["utr5_len"].max()), int(self.x_raw["cds_len"].max()), int(self.x_raw["utr3_len"].max())]
            # Process all genes
            genes = self.x_raw.index.tolist()
            frozen_fun = partial(self._prepare_modisco_1gene, nucl=nucl, conds=conds, ctrl_cond=ctrl_cond, maxlen=maxlen)
            attr = parallel_map(frozen_fun, genes, threads=threads, chunksize=10, desc="Preparing TF-MoDISco inputs")

            for region in ["5UTR","CDS","3UTR"]:
                # one-hot stored as float32 (instead of the int64 the int padding tiles produce):
                # halves the on-disk/in-RAM size and matches the float32 score arrays
                ohe = np.concatenate([s[region][0] for s in attr]).astype(np.float32)
                for cond in conds:
                    cond_dcount = np.concatenate([s[region][1][cond][0] for s in attr])
                    cond_ddcount = np.concatenate([s[region][1][cond][1] for s in attr])
                    condpath  = os.path.join(self.interpretation_dir, "counts", region + "_" + cond + "_modisco.npz")
                    np.savez(condpath, ohe=ohe, dcounts=cond_dcount, ddcounts=cond_ddcount)
        else:
            log.info("All modisco input .npz files already exist, skipping preparation.")
    
    def _run_modisco_counts(self, conds, ctrlcond, region, arr, motif_db, max_seqlets=1000):
        '''
        Run TF-MoDISco on count-based importance scores.

        Parameters
        ----------
        conds : list of str
            List of condition names.
        ctrlcond : str or None
            Control condition mapping.
        region : str
            Genomic region ("5UTR", "CDS", or "3UTR").
        arr : str
            Array type ("dcounts" or "ddcounts").
        motif_db : str or None
            Path to motif database file or None to skip database search.
        max_seqlets : int, default=1000
            Maximum number of seqlets per metacluster passed to TF-MoDISco.

        Returns
        -------
        None
            Saves TF-MoDISco results and adds them to the report.
        '''
        sitedir = os.path.join(self.interpretation_dir,"counts","-".join(conds), arr)
        if not os.path.exists(sitedir):
            os.makedirs(sitedir, exist_ok=True)
        if not all([os.path.exists(os.path.join(sitedir, region+"_modisco_results.h5")),
                    os.path.exists(os.path.join(sitedir, region+"_modisco_report"))]):
            log.info("Running TF-MoDISco for %s (%s).", region, arr)
            # Concat inputs
            snp_attr = []
            snp_seq = []
            for cond in conds:
                if arr=="ddcounts" and ctrlcond is not None and cond==ctrlcond:
                    continue
                # Read fully into owned arrays and close the file handle immediately:
                # both members are concatenated below, so there is nothing to gain from
                # keeping the NpzFile (and its buffers) open, and the arrays must outlive it.
                with np.load(os.path.join(self.interpretation_dir, "counts", region + "_" + cond + "_modisco.npz")) as tempfile:
                    snp_attr.append(np.nan_to_num(tempfile[arr]))
                    snp_seq.append(np.array(tempfile["ohe"]))
            snp_attr = np.concatenate(snp_attr)
            snp_seq = np.concatenate(snp_seq)
            # Run TF-MoDISco
            sliding_window_size = 11
            max_seqlets_per_metacluster = max_seqlets
            def seqlet_selection_fn(seqlets: list[Seqlet]) -> list[Seqlet]:
                scores = []; indices = []
                for i, seqlet in enumerate(seqlets):
                    flank = int(0.5 * (len(seqlet) - sliding_window_size))
                    scores.append(np.abs(np.sum(seqlet.contrib_scores[flank:-flank])))
                    indices.append(i)
                scores = np.array(scores)
                indices = np.array(indices)
                ordered = indices[np.argsort(scores)[::-1]]
                selection = ordered[:max_seqlets_per_metacluster]
                return [seqlets[i] for i in selection]
            pos_patterns, neg_patterns = TFMoDISco(snp_seq, snp_attr, n_leiden_runs=10, sliding_window_size=sliding_window_size,
                                                flank_size=5, final_min_cluster_size=10, verbose=True, revcomp=False,
                                                max_seqlets_per_metacluster=max_seqlets_per_metacluster, seqlet_selection_fn=seqlet_selection_fn)
            # Save results
            if pos_patterns is None:
                pos_patterns = []
            if neg_patterns is None:
                neg_patterns = []
            log.info(f"{region}_{arr}: Found {len(pos_patterns)} positive patterns" f" and {len(neg_patterns)} negative patterns.")
            save_hdf5(os.path.join(sitedir, region+"_modisco_results.h5"), pos_patterns, neg_patterns, snp_seq.shape[1])
            # Generate report
            if motif_db=="default":
                motif_db = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "RBP_motifs","attract_ornament.meme")
            elif motif_db==None or (isinstance(motif_db, str) and os.path.exists(motif_db)):
                pass
            else:
                warn_message = f"Provided motif database not found. Defaulting to Attract and oRNAment."
                log.warning(warn_message)
                motif_db = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "RBP_motifs","attract_ornament.meme")
            if motif_db is not None and shutil.which("tomtom") is None:
                log.warning("`tomtom` (MEME suite) not found in PATH: skipping motif database annotation. "
                            "Install it with `conda install -c bioconda meme` to annotate motifs.")
                motif_db = None
            report_motifs(os.path.join(sitedir, region+"_modisco_results.h5"),
                        os.path.join(sitedir, region+"_modisco_report"),
                        img_path_suffix=os.path.join(sitedir, region+"_modisco_report/"),
                        meme_motif_db=motif_db, is_writing_tomtom_matrix=False, revcomp=False)
        else:
            log.info("Using cached TF-MoDISco results for %s (%s).", region, arr)
        # Print
        alignment = "Sequences are aligned to the 3' end of the region, so the last positions correspond to the start of the coding sequence." if region=="5UTR" else f"Sequences are aligned to the 5' end of the region, so the first positions correspond to the beginning of the {region}."
        add_text(f"The TF-MoDISco analysis identified recurring sequence motifs in the {region} region. "
                 f"The motif report below shows discovered patterns, their sequence logos, and matches to known RNA-binding protein motifs from the database. "
                 f"{alignment}")
        add_html(os.path.join(sitedir, region+"_modisco_report","motifs.html"))
        add_text(f"The seqlet position plot shows the location of discovered motifs across the {region} region. "
                 f"Each row represents a discovered pattern, and colored positions indicate where seqlets (instances of that motif) were found across the sequence. "
                 f"Horizontal lines separate different patterns, ordered by their pattern number in the table. "
                 f"The intensity of the color corresponds to the importance score at that position for the seqlet. "
                 f"Red colors indicate the motif contributes positively to ribosome counts, while in blue colors, it contributes negatively.")
        seqlen = self.x_raw["utr5_len"].max()+5 if region=="5UTR" else self.x_raw["cds_len"].max() if region=="CDS" else self.x_raw["utr3_len"].max()+5
        fig = plot_pos_seqlets(os.path.join(sitedir, region+"_modisco_results.h5"), seqlen)
        fig.savefig(os.path.join(sitedir, region+"_seqlet_positions.pdf"))
        add_plot(fig)
        plt.close()
        return
    
    def get_genes_pattern(self, pattern, typepat, arr, conds, region):
        '''
        Extract genes associated with a specific TF-MoDISco pattern.

        Parameters
        ----------
        pattern : str
            Pattern identifier.
        typepat : str
            Pattern type ("pos_patterns" or "neg_patterns").
        arr : str
            Array type ("dcounts" or "ddcounts").
        conds : str
            List of condition names.
        region : str
            Genomic region ("5UTR", "CDS", or "3UTR").

        Returns
        -------
        pd.DataFrame
            DataFrame with genes, counts, max attribution, and location information.
        '''
        fh = h5.File(os.path.join(self.interpretation_dir,"counts", conds, arr, region+"_modisco_results.h5"))
        seqlets = np.array([s % len(self.x_raw.index) for s in fh[typepat][pattern]["seqlets"]["example_idx"]])
        maxattr = np.array([np.max(np.abs(s)) for s in fh[typepat][pattern]["seqlets"]["contrib_scores"]])    
        # start/end are positions within the padded, per-region ISM mutation-index arrays built by
        # _prepare_modisco_1gene (region-specific left/right padding; see _start_stop_metagene for
        # the alignment convention), NOT raw transcript or model profile positions.
        loc1 = np.array([s for s in fh[typepat][pattern]["seqlets"]["start"]])
        loc2  = np.array([s for s in fh[typepat][pattern]["seqlets"]["end"]])
        fh.close()
        genes = self.x_raw.index[seqlets]
        unique, counts = np.unique(genes, return_counts=True)
        attr = [max(maxattr[genes==s]) for s in unique]
        avgloc1 = [int(np.mean(loc1[genes==s])) for s in unique]
        avgloc2 = [int(np.mean(loc2[genes==s])) for s in unique]
        return pd.DataFrame(zip(unique, counts, attr, [(s,e) for s,e in zip(avgloc1, avgloc2)]), columns=["gene","counts","maxattr","loc"]).set_index("gene").sort_values(by="maxattr", ascending=False)

    def extract_counts_motifs(self, ctrl_cond=None, threads=1, motif_db="default", max_seqlets=1000, filename=None):
        '''
        Extract motifs based on count predictions using TF-MoDISco.

        Parameters
        ----------
        ctrl_cond : str, dict, optional
            Control condition specification.
        threads : int, default=1
            Number of threads for multiprocessing.
        motif_db : str, default="default"
            Path to motif database or "default" for built-in database.
        max_seqlets : int, default=1000
            Maximum number of seqlets per metacluster passed to TF-MoDISco.
        filename : str, optional
            Output filename for the report.

        Returns
        -------
        None
            Saves motif analysis results and adds plots to report.
        '''
        # Some checks
        if filename:
            create_report(os.path.join(self.interpretation_dir, filename), title="RiboInterpretation Counts")
            # Add introduction
            if self.scoring=="ISM":
                score_expl = "ISM systematically mutates each nucleotide position and measures the resulting change in model predictions, revealing which sequence positions are most important for determining ribosome counts and distribution."
            else:
                score_expl = ""
            add_text("This report presents the interpretation of RiboNet model predictions through feature importance analysis. "
                    f"The analysis identifies sequence features that influence ribosome occupancy predictions. "
                    f"{score_expl}")
            add_page_break()
            add_toc()
        if not os.path.exists(os.path.join(self.interpretation_dir,"counts")):
            os.makedirs(os.path.join(self.interpretation_dir,"counts"),exist_ok=True)
        # Define control conditions if not done before
        ctrl_cond = self._define_ctrl_conditions(ctrl_cond)
        # Define conditions
        conds = self.y_raw["cds"].columns if ctrl_cond is None else [c for c in self.y_raw["cds"].columns if c in ctrl_cond.keys()]
        # Prepare modisco inputs
        log.info("Retrieving scored sequences and preparing inputs for TF-MoDISco.")
        self._prepare_modisco_inputs(conds=conds, ctrl_cond=ctrl_cond, threads=threads)
        # Metaplot start stop
        add_heading("Metagene importance scores around START and STOP")
        self._start_stop_metagene(conds)
        add_page_break()
        # Define condition sets
        if ctrl_cond is None:
            condsets = [(conds, ctrl_cond)]
        else:
            condsets = [([k for k,v in ctrl_cond.items() if c==v],c) for c in list(set(ctrl_cond.values()))]
        for c,ctrl in condsets:
            ctrlname = ctrl if ctrl is not None else "mean"
            sitedir = os.path.join(self.interpretation_dir,"counts","-".join(c))
            if not os.path.exists(sitedir):
                os.makedirs(sitedir, exist_ok=True)
            add_heading("Counts predictive features among %s" % ("-".join(c)))
            add_subheading("Motifs predictive of absolute ribosome counts")
            add_text("Absolute nucleotide importance scores quantify how changes at each nucleotide position affect the predicted ribosome count. "
                     "TF-MoDISco analyzes these scores across all genes and conditions to identify recurrently appearing sequence motifs. "
                     "The discovered motifs represent sequence elements that consistently influence ribosome occupancy along transcripts.")
            log.info("Running TF-MoDISco for 5'UTR predictive features in absolute counts among %s." % ("-".join(c)))
            add_subsubheading("5'UTR predictive features")
            self._run_modisco_counts(c, ctrl, "5UTR", "dcounts", motif_db, max_seqlets=max_seqlets)
            log.info("Running TF-MoDISco for CDS predictive features in absolute counts among %s." % ("-".join(c)))
            add_subsubheading("CDS predictive features")
            self._run_modisco_counts(c, ctrl, "CDS", "dcounts", motif_db, max_seqlets=max_seqlets)
            log.info("Running TF-MoDISco for 3'UTR predictive features in absolute counts among %s." % ("-".join(c)))
            add_subsubheading("3'UTR predictive features")
            self._run_modisco_counts(c, ctrl, "3UTR", "dcounts", motif_db, max_seqlets=max_seqlets)
            add_page_break()
            # Relative counts
            log.info("Running TF-MoDISco for 5'UTR predictive features in relative counts among %s." % ("-".join(c)))
            add_subheading("Motifs predictive of relative ribosome counts vs %s" % ctrlname)
            add_text(f"Relative nucleotide importance scores show how each nucleotide affects the difference in ribosome counts compared to the control condition ({ctrlname}). "
                     "These differential scores highlight sequence features that cause condition-specific changes in ribosome occupancy. "
                     "TF-MoDISco identifies recurring motifs that drive these differential effects across transcripts.")
            add_subsubheading("5'UTR predictive features")
            self._run_modisco_counts(c, ctrl, "5UTR", "ddcounts", motif_db, max_seqlets=max_seqlets)
            log.info("Running TF-MoDISco for CDS predictive features in relative counts among %s." % ("-".join(c)))
            add_subsubheading("CDS predictive features")
            self._run_modisco_counts(c, ctrl, "CDS", "ddcounts", motif_db, max_seqlets=max_seqlets)
            log.info("Running TF-MoDISco for 3'UTR predictive features in relative counts among %s." % ("-".join(c)))
            add_subsubheading("3'UTR predictive features")
            self._run_modisco_counts(c, ctrl, "3UTR", "ddcounts", motif_db, max_seqlets=max_seqlets)
            add_page_break()
        if filename:
            gc.collect()
            add_interactive_guide()
            finalize_report()
        return

    def _define_ctrl_conditions(self, ctrl_cond, conds=None):
        '''
        Define control conditions based on user input.

        Parameters
        ----------
        ctrl_cond : str, dict, or None
            Control condition specification.
            Can be a track name, condition name, or dictionary mapping.

        Returns
        -------
        dict or None
            Validated control condition mapping or None.
        '''
        if ctrl_cond is None or isinstance(ctrl_cond, dict):
            ctrl_cond = ctrl_cond
        elif ctrl_cond in self.sample_file["track"].values:
            if conds is None:
                ctrl_cond = {t:ctrl_cond for t in self.sample_file["track"].values}
            else:
                ctrl_cond = {t:ctrl_cond for t in conds if t in self.sample_file["track"].values}
                notfound = list(set([s for s in conds if s not in ctrl_cond.keys()]))
                if any(notfound):
                    err_message = f"Provided conditions {', '.join(notfound)} not found in sample file."
                    log.error(err_message)
                    raise ValueError(err_message)
        elif ctrl_cond in self.sample_file["condition"].values:
            ctrl_cond = {df.loc["track"]:"_".join([ctrl_cond,df.loc["readlength"]]) for _, df in self.sample_file.iterrows()}
            notfound = list(set([s for s in ctrl_cond.values() if s not in self.sample_file["track"].values]))
            if any(notfound):
                err_message = f"Provided control tracks {', '.join(notfound)} not found in sample file."
                log.error(err_message)
                raise ValueError(err_message)
        else:
            err_message = f"Provided control condition {ctrl_cond} not found in sample file."
            log.error(err_message)
            raise ValueError(err_message)
        return ctrl_cond

    def _diffsites_1gene_pausing_codon(self, gene, conds, diffsites):
        '''
        Identify differential ribosome occupancy (pausing) sites for a single codon resolution gene.

        This is the codon-resolution pausing analysis: a site is a single codon, and its occupancy
        is compared against the rest of the transcript across conditions.

        Parameters
        ----------
        gene : str
            Gene identifier.
        conds : list of str
            List of condition names to compare.
        diffsites : int or float
            If int >1, number of top differential sites to select.
            If float 0-1, p-value threshold for significance.

        Returns
        -------
        list
            List of differential site positions or (position, p-value) tuples.
        '''
        p_idx = []
        pvals = []
        p_idx_temp = []
        # Get profiles
        profiles = np.squeeze(np.nansum(self.y_mat[self.y_raw.index.get_loc(gene),np.where([s in conds for s in self.y_raw["cds"].columns]),:,:], axis=-1))
        # Loop through CDS positions only. `m` is a codon-resolution PROFILE position (index into
        # the model's fixed-length [pooled UTR5 | CDS codons | pooled UTR3] axis), NOT the ISM
        # mutation-index position used for pos_seq/diffsites.csv. The two are reconciled later via
        # the seqmapper conversion in _sites_scores_1gene_pausing_codon / _extract_sites_scores.
        for m in range(self.utr_pool_size[0], self.utr_pool_size[0] + self.seq_len):
            if ~np.isnan(profiles[:,m]).any():
                cont_table = np.array([[round(p[m]), round(np.nansum(np.delete(p,m)))] for p in profiles])
                if all(cont_table.sum(axis=0)>0):
                    cont_table = cont_table[cont_table.sum(axis=1)>0,:]
                    pvals.append(stats.chi2_contingency(cont_table)[1])
                    p_idx_temp.append([gene, m])

        if diffsites<=1.0:
            pcors = stats.multitest.fdrcorrection(pvals)[1]
            p_idx =  list(itertools.compress(p_idx_temp,pcors<0.05))
        elif diffsites>1.0:
            p_idx = [(x, y) for x, y in zip(p_idx_temp,pvals)]
        return p_idx
    
    def _diffsites_1gene_frameshift(self, gene, conds, diffsites):
        '''
        Identify differential in- vs out-of-frame (frameshift) sites for a single nucleotide
        resolution gene.

        A site is the first nucleotide of a canonical codon (``annotation == 1``); the in-frame
        count ``p[m]`` is compared against the out-of-frame counts ``p[m+1] + p[m+2]`` across
        conditions. This detects positions where the reading-frame distribution changes between
        conditions (frameshifting), and only makes sense at nucleotide resolution.

        Parameters
        ----------
        gene : str
            Gene identifier.
        conds : list of str
            List of condition names to compare.
        diffsites : int or float
            If int >1, number of top differential sites to select.
            If float 0-1, p-value threshold for significance.

        Returns
        -------
        list
            List of differential site positions or (position, p-value) tuples.
        '''
        p_idx = []
        pvals = []
        p_idx_temp = []
        # Get profiles
        profiles = np.squeeze(np.nansum(self.y_mat[self.y_raw.index.get_loc(gene),np.where([s in conds for s in self.y_raw["cds"].columns]),:,:], axis=-1))
        annotations = self.x_c.loc[gene, "annotation"]
        # Loop through all positions. At nucleotide resolution the PROFILE axis is anchored at the
        # gene's own annotated UTR5 start (no left padding, see reshape_training_data), so `m` is
        # already identical to the ISM mutation-index position used for pos_seq/diffsites.csv -
        # no seqmapper conversion is needed here (unlike the codon-resolution case).
        for m in np.nonzero(annotations == 1)[0]: # get first nucleotide of all codons in cds
            if ~np.isnan(profiles[:,m]).any():
                cont_table = np.array([[round(p[m]), round(p[m + 1] + p[m + 2])] for p in profiles])
                if all(cont_table.sum(axis=0)>0):
                    cont_table = cont_table[cont_table.sum(axis=1)>0,:]
                    pvals.append(stats.chi2_contingency(cont_table)[1])
                    p_idx_temp.append([gene, m])

        if diffsites<=1.0:
            pcors = stats.multitest.fdrcorrection(pvals)[1]
            p_idx =  list(itertools.compress(p_idx_temp,pcors<0.05))
        elif diffsites>1.0:
            p_idx = [(x, y) for x, y in zip(p_idx_temp,pvals)]
        return p_idx

    def _diffsites_1gene_pausing_nt(self, gene, conds, diffsites):
        '''
        Identify differential ribosome occupancy (pausing) sites for a single nucleotide
        resolution gene.

        This is the nucleotide-resolution counterpart of :meth:`_diffsites_1gene_pausing_codon`:
        a site is a canonical codon (first nucleotide marked ``annotation == 1``) whose occupancy
        is obtained by *summing the three nucleotide counts of the triplet* ``p[m:m+3]``, and this
        codon-level occupancy is compared against the rest of the transcript across conditions.

        Parameters
        ----------
        gene : str
            Gene identifier.
        conds : list of str
            List of condition names to compare.
        diffsites : int or float
            If int >1, number of top differential sites to select.
            If float 0-1, p-value threshold for significance.

        Returns
        -------
        list
            List of differential site positions (first nucleotide of the codon) or
            (position, p-value) tuples.
        '''
        p_idx = []
        pvals = []
        p_idx_temp = []
        # Get profiles (nucleotide resolution)
        profiles = np.squeeze(np.nansum(self.y_mat[self.y_raw.index.get_loc(gene),np.where([s in conds for s in self.y_raw["cds"].columns]),:,:], axis=-1))
        annotations = self.x_c.loc[gene, "annotation"]
        # Loop through all codons in cds (first nucleotide of each canonical codon). As in
        # _diffsites_1gene_frameshift, `m` is a nucleotide-resolution PROFILE position, which at this
        # resolution equals the ISM mutation-index position (no seqmapper conversion needed).
        for m in np.nonzero(annotations == 1)[0]:
            if ~np.isnan(profiles[:, m:m + 3]).any():
                # codon occupancy = sum of the triplet; rest = whole transcript minus this codon
                cont_table = np.array([[round(np.nansum(p[m:m + 3])), round(np.nansum(p) - np.nansum(p[m:m + 3]))] for p in profiles])
                if all(cont_table.sum(axis=0)>0):
                    cont_table = cont_table[cont_table.sum(axis=1)>0,:]
                    pvals.append(stats.chi2_contingency(cont_table)[1])
                    p_idx_temp.append([gene, m])

        if diffsites<=1.0:
            pcors = stats.multitest.fdrcorrection(pvals)[1]
            p_idx =  list(itertools.compress(p_idx_temp,pcors<0.05))
        elif diffsites>1.0:
            p_idx = [(x, y) for x, y in zip(p_idx_temp,pvals)]
        return p_idx

    def _get_diffsites(self, conds, diffsites, threads, analysis="pausing"):
        '''
        Identify differential ribosome occupancy sites across all genes.

        Parameters
        ----------
        conds : list of str
            List of condition names to compare.
        diffsites : int or float
            If int >1, number of top differential sites to select.
            If float 0-1, p-value threshold for significance.
        threads : int
            Number of threads for parallel processing.
        analysis : str, default="pausing"
            Which differential-site definition to use: "pausing" (differential codon occupancy) or
            "frameshift" (differential in- vs out-of-frame ratio, nucleotide resolution only).

        Returns
        -------
        pd.DataFrame
            DataFrame containing differential sites.
        '''
        # Pick the per-gene differential-site function for this analysis and resolution
        if analysis == "frameshift":
            if self.seq_resolution != "nucleotide":
                err_message = "Frameshift analysis is only available at nucleotide resolution."
                log.error(err_message)
                raise ValueError(err_message)
            diffsites_fn = self._diffsites_1gene_frameshift
        elif analysis == "pausing":
            diffsites_fn = self._diffsites_1gene_pausing_codon if self.seq_resolution == "codon" else self._diffsites_1gene_pausing_nt
        else:
            err_message = f"Unknown analysis '{analysis}'. Expected 'pausing' or 'frameshift'."
            log.error(err_message)
            raise ValueError(err_message)
        # Identify p_idx
        frozen_fun = partial(
            diffsites_fn,
            conds=conds,
            diffsites=diffsites)
        p_idx = parallel_map(frozen_fun, self.y_raw.index, threads=threads, chunksize=100)
        # Flatten gene lists
        p_idx = [s for genelist in p_idx for s in genelist]
        # Get final list pausing sites
        if diffsites<=1.0:
            p_idx = pd.DataFrame(p_idx)
        elif diffsites>1.0:
            pcors_idx = np.argsort([s[1] for s in p_idx])
            p_idx = [s[0] for s in p_idx]
            p_idx =  [p_idx[i] for i in pcors_idx][:int(diffsites)]
            p_idx = pd.DataFrame(p_idx)
        return p_idx

    def _sites_scores_1gene_frameshift(self, genepeaks, pwind, conds, ctrlcond, nucl, subdir="frameshift"):
        '''
        Extract frameshift importance scores around differential sites for a single nucleotide
        resolution gene.

        For each differential site ``i`` (first nucleotide of a canonical codon) it records the
        differential ISM effect of mutating every window position on the in-frame count (score_0,
        profile position ``i``) and the two out-of-frame counts (score_1/score_2, profile positions
        ``i+1`` / ``i+2``), contrasted against ``ctrlcond`` if given, else against the mean.

        The window is clipped to the transcript and padded with 'A' (scores stay NaN on the padded
        flanks), mirroring the codon-resolution coordinate handling in
        :meth:`_sites_scores_1gene_pausing_codon`.

        Parameters
        ----------
        genepeaks : tuple
            Tuple of (gene_name, peak_positions).
        pwind : int
            Window size around each peak.
        conds : list of str
            List of condition names.
        ctrlcond : str or None
            Control condition name.
        nucl : list of str
            List of nucleotides.
        subdir : str, default="frameshift"
            Sub-directory under the condition-set distribution directory to read the windowed
            ``diffsites_scores.parquet`` from (ignored when ``save_distribution`` is True).

        Returns
        -------
        tuple
            (windows_dict, one_hot_encoded_sequences)
            windows_dict has structure {cond: {"score_0": ndarray, "score_1": ndarray, "score_2": ndarray}}
        '''
        # Find batch
        gene, pos = genepeaks
        if self.save_distribution:
            batch_num = int([k for k,v in self.batch_splits.items() if gene in v][0])
            # Upload contrasts
            contrasts = pd.read_parquet(os.path.join(self.interpretation_dir, "full_contrasts", "contrasts_batch_%i.parquet" % batch_num), filters=[("gene", "==", gene)])
        else:
            ctrlname = ctrlcond if ctrlcond is not None else "mean"
            sitedir = os.path.join(self.interpretation_dir,"distribution","-".join(conds)+"_vs_"+ctrlname)
            if subdir:
                sitedir = os.path.join(sitedir, subdir)
            contrasts = pd.read_parquet(os.path.join(sitedir, "diffsites_scores.parquet"), filters=[("gene", "==", gene)])
        contrast_per_pos = contrasts.loc[(ser:=contrasts["pos_seq"].explode()).index].assign(poslong=contrasts["ori_seq"]+ser.astype(str)+contrasts["to_seq"])
        dprofile = contrast_per_pos.loc[(ser:=contrast_per_pos["dprofile"].explode()).index].assign(dprofval=ser, cond=list(self.y_raw["cds"].columns)*contrast_per_pos.shape[0]).pivot_table(index="poslong",columns="cond",values="dprofval",dropna=False)
        if ctrlcond is not None:
            ddprofile = dprofile[conds].sub(dprofile[ctrlcond], axis=0).assign(pos=[int(s[1:-1]) for s in dprofile.index]).pivot_table(index="pos", aggfunc=lambda x: np.mean(np.stack(x), axis=0),dropna=False)
        else:
            avgprof = dprofile[conds].apply(lambda x: np.mean(np.stack(x, axis=-1), axis=-1), axis=1)
            ddprofile = dprofile[conds].sub(avgprof, axis=0).assign(pos=[int(s[1:-1]) for s in dprofile.index]).pivot_table(index="pos", aggfunc=lambda x: np.mean(np.stack(x), axis=0), dropna=False)

        # Record seq (one nucleotide per scored transcript position, indexed by absolute position)
        seq = contrast_per_pos["poslong"].apply(lambda x: [x[:1],x[1:-1]]).drop_duplicates().apply(lambda x: x[0]).values

        # Slice windows (float32 throughout: source parquet scores are float32)
        A_pad = np.array([[1, 0, 0, 0]], dtype=np.float32)  # 'A' as padding nucleotide
        frames = ["score_0", "score_1", "score_2"]
        windows = {cond: {frame: np.full((len(pos), np.sum(2*pwind), 4), np.nan, dtype=np.float32) for frame in frames} for cond in conds}
        oheall = np.zeros((len(pos),np.sum(2*pwind),4), dtype=np.float32)
        # mutation axis and profile axis are both the (padded) nucleotide length
        nt_rows = self.seq_len
        for cond,col in ddprofile.items():
            score = np.full((nt_rows, self.seq_len), np.nan, dtype=np.float32)
            score[col.index,:] = np.stack(col)
            for n,i in enumerate(pos):
                i = int(i)  # identity map: frameshift peaks (annotation==1 nt positions) already share the mutation-index space
                win_start = i - pwind
                win_end = i + pwind
                clip_start = max(0, win_start)
                clip_end = min(len(seq), win_end)
                if clip_start >= clip_end:
                    continue  # peak center outside transcript; oheall stays zeros
                left_pad = clip_start - win_start  # >= 0, positions to pad on the left
                right_pad = win_end - clip_end      # >= 0, positions to pad on the right
                n_valid = clip_end - clip_start
                subseq = seq[clip_start:clip_end]
                sub_ohe = pd.get_dummies(list(subseq))
                for b in [s for s in nucl if s not in sub_ohe.columns]:
                    sub_ohe[b] = False
                sub_ohe = sub_ohe[nucl].values  # (n_valid, 4)
                # Padded OHE: out-of-transcript positions use A ([1,0,0,0]); scores stay NaN there
                full_ohe = np.concatenate([
                    np.tile(A_pad, (left_pad, 1)),
                    sub_ohe,
                    np.tile(A_pad, (right_pad, 1))
                ])  # (2*pwind, 4)
                oheall[n, :, :] = full_ohe
                # For the in-frame (score_0) and out-of-frame (score_1/_2) profile columns
                for f_off, frame in enumerate(frames):
                    col_idx = i + f_off
                    if col_idx >= self.seq_len:
                        continue  # profile column outside model window
                    valid_sis = score[clip_start:clip_end, col_idx]  # (n_valid,)
                    windows[cond][frame][n, left_pad:left_pad + n_valid][sub_ohe.astype(bool)] = valid_sis
        return windows, oheall

    def _sites_scores_1gene_pausing_codon(self, genepeaks, pwind, conds, ctrlcond, nucl, subdir=""):
        '''
        Extract ribosome-pausing importance scores around differential sites for a single gene
        with codon resolution.

        For each differential codon site the differential ISM effect on the codon occupancy is
        recorded across the surrounding window, contrasted against ``ctrlcond`` if given, else
        against the mean.

        Parameters
        ----------
        genepeaks : tuple
            Tuple of (gene_name, peak_positions).
        pwind : int
            Window size around each peak.
        conds : list of str
            List of condition names.
        ctrlcond : str or None
            Control condition name.
        nucl : list of str
            List of nucleotides.
        subdir : str, default=""
            Sub-directory under the condition-set distribution directory to read the windowed
            ``diffsites_scores.parquet`` from (ignored when ``save_distribution`` is True).
            Defaults to "" so the codon-resolution paths are byte-identical to previous runs.

        Returns
        -------
        tuple
            (windows_dict, one_hot_encoded_sequences)
        '''
        # Find batch
        gene, pos = genepeaks
        if self.save_distribution:
            batch_num = int([k for k,v in self.batch_splits.items() if gene in v][0])
            # Upload contrasts
            contrasts = pd.read_parquet(os.path.join(self.interpretation_dir, "full_contrasts", "contrasts_batch_%i.parquet" % batch_num), filters=[("gene", "==", gene)])
        else:
            ctrlname = ctrlcond if ctrlcond is not None else "mean"
            sitedir = os.path.join(self.interpretation_dir,"distribution","-".join(conds)+"_vs_"+ctrlname)
            if subdir:
                sitedir = os.path.join(sitedir, subdir)
            contrasts = pd.read_parquet(os.path.join(sitedir, "diffsites_scores.parquet"), filters=[("gene", "==", gene)])
        contrast_per_pos = contrasts.loc[(ser:=contrasts["pos_seq"].explode()).index].assign(poslong=contrasts["ori_seq"]+ser.astype(str)+contrasts["to_seq"])
        dprofile = contrast_per_pos.loc[(ser:=contrast_per_pos["dprofile"].explode()).index].assign(dprofval=ser, cond=list(self.y_raw["cds"].columns)*contrast_per_pos.shape[0]).pivot_table(index="poslong",columns="cond",values="dprofval",dropna=False)
        if ctrlcond is not None:
            ddprofile = dprofile[conds].sub(dprofile[ctrlcond], axis=0).assign(pos=[int(s[1:-1]) for s in dprofile.index]).pivot_table(index="pos", aggfunc=lambda x: np.mean(np.stack(x), axis=0),dropna=False)
        else:
            avgprof = dprofile[conds].apply(lambda x: np.mean(np.stack(x, axis=-1), axis=-1), axis=1)
            ddprofile = dprofile[conds].sub(avgprof, axis=0).assign(pos=[int(s[1:-1]) for s in dprofile.index]).pivot_table(index="pos", aggfunc=lambda x: np.mean(np.stack(x), axis=0), dropna=False)
        # Record seq
        seq = contrast_per_pos["poslong"].apply(lambda x: [x[:1],x[1:-1]]).drop_duplicates().apply(lambda x: x[0]).values
        # Create ohe
        ohe = pd.get_dummies(seq)
        for b in [s for s in nucl if s not in ohe.columns]:
            ohe[b] = False
        ohe = ohe[nucl].values
        # Slice windows
        regsizes = [self.x_raw.loc[gene,"utr5_len"], self.x_raw.loc[gene,"cds_len"], self.x_raw.loc[gene,"utr3_len"]]
        # seqmapper converts a codon-resolution PROFILE position n (pooled UTR5 bin / CDS codon index /
        # pooled UTR3 bin, indexing the fixed-length model window) into the ISM mutation-index position
        # (i.e. the same space as `seq`/pos_seq, anchored at this gene's own annotated UTR5 start). Each
        # branch first recovers the absolute nt position within the fixed, left/right-padded model window,
        # then subtracts the padding specific to this gene (utr5 left-pad, cds right-pad) to land in the
        # gene-relative mutation-index space.
        seqmapper = {n:(int(n/self.utr_pool) - (self.utr_len[0]-regsizes[0])) if n<self.utr_pool_size[0] 
                     else (self.utr_len[0]+int((n-self.utr_pool_size[0])*3) - (self.utr_len[0]-regsizes[0])) if n<sum([self.seq_len,self.utr_pool_size[0]]) 
                     else (sum([self.utr_len[0], self.seq_len*3])+int((n-sum([self.seq_len,self.utr_pool_size[0]]))/self.utr_pool) - (self.utr_len[0]-regsizes[0]) - ((self.seq_len*3)-regsizes[1])) for n in range(int(sum([self.seq_len, self.utr_pool_size[0], self.utr_pool_size[1]])+1))}
        A_pad = np.array([[1, 0, 0, 0]], dtype=np.float32)  # 'A' as padding nucleotide
        # float32 throughout: source parquet scores are float32
        windows = {cond:np.full((len(pos),np.sum(2*pwind),4), np.nan, dtype=np.float32) for cond in conds}
        oheall = np.zeros((len(pos),np.sum(2*pwind),4), dtype=np.float32)
        for cond,col in ddprofile.items():
            nt_len = self.utr_len[0] + self.seq_len * 3 + self.utr_len[1]
            model_len = sum([self.seq_len, self.utr_pool_size[0], self.utr_pool_size[1]])
            sis = np.full((nt_len, model_len), np.nan, dtype=np.float32)
            sis[col.index,:] = np.stack(col)
            for n,i in enumerate(pos):
                seq_pos = int(seqmapper[i])  # i: codon-resolution profile position -> seq_pos: mutation-index position
                win_start = seq_pos - pwind
                win_end = seq_pos + pwind
                clip_start = max(0, win_start)
                clip_end = min(len(seq), win_end)
                if clip_start >= clip_end:
                    continue  # peak center outside transcript; oheall stays zeros → filtered by empty_idx
                left_pad = clip_start - win_start  # >= 0, positions to pad on the left
                right_pad = win_end - clip_end      # >= 0, positions to pad on the right
                n_valid = clip_end - clip_start
                subseq = seq[clip_start:clip_end]
                sub_ohe = pd.get_dummies(list(subseq))
                for b in [s for s in nucl if s not in sub_ohe.columns]:
                    sub_ohe[b] = False
                sub_ohe = sub_ohe[nucl].values  # (n_valid, 4)
                # Padded OHE: out-of-transcript positions use A ([1,0,0,0]); scores stay NaN there
                full_ohe = np.concatenate([
                    np.tile(A_pad, (left_pad, 1)),
                    sub_ohe,
                    np.tile(A_pad, (right_pad, 1))
                ])  # (2*pwind, 4)
                oheall[n, :, :] = full_ohe
                # Assign scores for valid positions; padded positions keep NaN from initialization
                valid_sis = sis[clip_start:clip_end, i]  # (n_valid,)
                windows[cond][n, left_pad:left_pad + n_valid][sub_ohe.astype(bool)] = valid_sis
        return windows, oheall

    def _sites_scores_1gene_pausing_nt(self, genepeaks, pwind, conds, ctrlcond, nucl, subdir="pausing"):
        '''
        Extract ribosome-pausing importance scores around differential sites for a single
        nucleotide resolution gene.

        This mirrors :meth:`_sites_scores_1gene_pausing_codon` exactly, the only differences being
        that (i) the peaks are already nucleotide positions (first nt of a canonical codon), so the
        sequence-to-profile map is the identity, and (ii) the per-position score is the codon
        occupancy, i.e. the sum of the three in-codon profile columns ``i``, ``i+1`` and ``i+2``.
        Contrasted against ``ctrlcond`` if given, else against the mean.

        Parameters
        ----------
        genepeaks : tuple
            Tuple of (gene_name, peak_positions).
        pwind : int
            Window size around each peak.
        conds : list of str
            List of condition names.
        ctrlcond : str or None
            Control condition name.
        nucl : list of str
            List of nucleotides.
        subdir : str, default="pausing"
            Sub-directory under the condition-set distribution directory to read the windowed
            ``diffsites_scores.parquet`` from (ignored when ``save_distribution`` is True).

        Returns
        -------
        tuple
            (windows_dict, one_hot_encoded_sequences)
            windows_dict has structure {cond: ndarray} (identical shape to the codon pausing case).
        '''
        # Find batch
        gene, pos = genepeaks
        if self.save_distribution:
            batch_num = int([k for k,v in self.batch_splits.items() if gene in v][0])
            # Upload contrasts
            contrasts = pd.read_parquet(os.path.join(self.interpretation_dir, "full_contrasts", "contrasts_batch_%i.parquet" % batch_num), filters=[("gene", "==", gene)])
        else:
            ctrlname = ctrlcond if ctrlcond is not None else "mean"
            sitedir = os.path.join(self.interpretation_dir,"distribution","-".join(conds)+"_vs_"+ctrlname)
            if subdir:
                sitedir = os.path.join(sitedir, subdir)
            contrasts = pd.read_parquet(os.path.join(sitedir, "diffsites_scores.parquet"), filters=[("gene", "==", gene)])
        contrast_per_pos = contrasts.loc[(ser:=contrasts["pos_seq"].explode()).index].assign(poslong=contrasts["ori_seq"]+ser.astype(str)+contrasts["to_seq"])
        dprofile = contrast_per_pos.loc[(ser:=contrast_per_pos["dprofile"].explode()).index].assign(dprofval=ser, cond=list(self.y_raw["cds"].columns)*contrast_per_pos.shape[0]).pivot_table(index="poslong",columns="cond",values="dprofval",dropna=False)
        if ctrlcond is not None:
            ddprofile = dprofile[conds].sub(dprofile[ctrlcond], axis=0).assign(pos=[int(s[1:-1]) for s in dprofile.index]).pivot_table(index="pos", aggfunc=lambda x: np.mean(np.stack(x), axis=0),dropna=False)
        else:
            avgprof = dprofile[conds].apply(lambda x: np.mean(np.stack(x, axis=-1), axis=-1), axis=1)
            ddprofile = dprofile[conds].sub(avgprof, axis=0).assign(pos=[int(s[1:-1]) for s in dprofile.index]).pivot_table(index="pos", aggfunc=lambda x: np.mean(np.stack(x), axis=0), dropna=False)
        # Record seq (one nucleotide per scored transcript position, indexed by absolute position)
        seq = contrast_per_pos["poslong"].apply(lambda x: [x[:1],x[1:-1]]).drop_duplicates().apply(lambda x: x[0]).values
        A_pad = np.array([[1, 0, 0, 0]], dtype=np.float32)  # 'A' as padding nucleotide
        # float32 throughout: source parquet scores are float32
        windows = {cond:np.full((len(pos),np.sum(2*pwind),4), np.nan, dtype=np.float32) for cond in conds}
        oheall = np.zeros((len(pos),np.sum(2*pwind),4), dtype=np.float32)
        # mutation axis and profile axis are both the (padded) nucleotide length
        nt_rows = self.seq_len
        for cond,col in ddprofile.items():
            sis = np.full((nt_rows, self.seq_len), np.nan, dtype=np.float32)
            sis[col.index,:] = np.stack(col)
            for n,i in enumerate(pos):
                seq_pos = int(i)  # identity map: peaks are already nt codon-start positions
                if seq_pos + 2 >= self.seq_len:
                    continue  # codon triplet outside model window; oheall stays zeros
                win_start = seq_pos - pwind
                win_end = seq_pos + pwind
                clip_start = max(0, win_start)
                clip_end = min(len(seq), win_end)
                if clip_start >= clip_end:
                    continue  # peak center outside transcript; oheall stays zeros → filtered by empty_idx
                left_pad = clip_start - win_start  # >= 0, positions to pad on the left
                right_pad = win_end - clip_end      # >= 0, positions to pad on the right
                n_valid = clip_end - clip_start
                subseq = seq[clip_start:clip_end]
                sub_ohe = pd.get_dummies(list(subseq))
                for b in [s for s in nucl if s not in sub_ohe.columns]:
                    sub_ohe[b] = False
                sub_ohe = sub_ohe[nucl].values  # (n_valid, 4)
                # Padded OHE: out-of-transcript positions use A ([1,0,0,0]); scores stay NaN there
                full_ohe = np.concatenate([
                    np.tile(A_pad, (left_pad, 1)),
                    sub_ohe,
                    np.tile(A_pad, (right_pad, 1))
                ])  # (2*pwind, 4)
                oheall[n, :, :] = full_ohe
                # Codon occupancy = sum of the three in-codon profile columns (triplet sum)
                valid_sis = sis[clip_start:clip_end, seq_pos] + sis[clip_start:clip_end, seq_pos + 1] + sis[clip_start:clip_end, seq_pos + 2]  # (n_valid,)
                windows[cond][n, left_pad:left_pad + n_valid][sub_ohe.astype(bool)] = valid_sis
        return windows, oheall

    def _extract_sites_scores(self, p_per_gene, pwind, conds, ctrlcond, nthreads, analysis="pausing", nucl=["A","C","G","T"]):
        '''
        Extract importance scores around differential sites for all genes.

        Parameters
        ----------
        p_per_gene : dict
            Dictionary mapping gene names to peak positions.
        pwind : int
            Window size around each peak.
        conds : list of str
            List of condition names.
        ctrlcond : str or None
            Control condition name.
        nthreads : int
            Number of threads for parallel processing.
        analysis : str, default="pausing"
            Which analysis to extract scores for: "pausing" (both resolutions) or "frameshift"
            (nucleotide resolution only).
        nucl : list of str, default=["A","C","G","T"]
            List of nucleotides.

        Returns
        -------
        tuple
            If analysis == "pausing": (peak_windows, one_hot_encoded_sequences)
                where peak_windows is {cond: array}
            If analysis == "frameshift": (frame_peak_windows, one_hot_encoded_sequences)
                where frame_peak_windows is {cond: {frame: array}} for frames score_0, score_1, score_2
        '''
        if analysis == "frameshift":
            if self.seq_resolution != "nucleotide":
                raise ValueError("Frameshift analysis is only defined at nucleotide resolution.")
            subdir = "frameshift"
            sites_scores_fn = self._sites_scores_1gene_frameshift
        elif analysis == "pausing":
            # codon keeps subdir="" so its paths stay byte-identical to previous runs
            subdir = "" if self.seq_resolution == "codon" else "pausing"
            sites_scores_fn = self._sites_scores_1gene_pausing_codon if self.seq_resolution == "codon" else self._sites_scores_1gene_pausing_nt
        else:
            raise ValueError("analysis must be 'pausing' or 'frameshift', got %r." % analysis)
        is_frameshift = analysis == "frameshift"
        ctrlname = ctrlcond if ctrlcond is not None else "mean"
        sitedir = os.path.join(self.interpretation_dir,"distribution","-".join(conds)+"_vs_"+ctrlname)
        if subdir:
            sitedir = os.path.join(sitedir, subdir)
            os.makedirs(sitedir, exist_ok=True)
        parquet_path = os.path.join(sitedir, "diffsites_scores.parquet")
        if not self.save_distribution and os.path.exists(parquet_path):
            scored_genes = set(pd.read_parquet(parquet_path, columns=["gene"])["gene"].unique())
            required_genes = set(p_per_gene.keys())
            if not required_genes.issubset(scored_genes):
                missing = required_genes - scored_genes
                log.warning("diffsites_scores.parquet is stale: %d gene(s) missing (e.g. %s). Recomputing.", len(missing), next(iter(missing)))
                os.remove(parquet_path)
        cache_path = os.path.join(sitedir, f"sites_scores_pwind{pwind}.npz")
        if not self.save_distribution and not os.path.exists(parquet_path):
            allpos = {}
            for g, pos in p_per_gene.items():
                # `pos` here are diffsites/PROFILE positions (see _get_diffsites); they must be
                # converted to ISM mutation-index positions before being used as `gene_pos` in
                # mutagenize (which filters on the mutation-index space), otherwise the wrong
                # window of the transcript would be re-scored around each differential-site peak.
                regsizes = [self.x_raw.loc[g,"utr5_len"], self.x_raw.loc[g,"cds_len"], self.x_raw.loc[g,"utr3_len"]]
                if self.seq_resolution == "codon":
                    # map from (codon-resolution) profile position to mutation-index position
                    seqmapper = {n:int(int(n/self.utr_pool) - (self.utr_len[0]-regsizes[0])) if n<self.utr_pool_size[0] 
                        else int(self.utr_len[0]+int((n-self.utr_pool_size[0])*3) - (self.utr_len[0]-regsizes[0])) if n<sum([self.seq_len,self.utr_pool_size[0]]) 
                        else int(sum([self.utr_len[0], self.seq_len*3])+int((n-sum([self.seq_len,self.utr_pool_size[0]]))/self.utr_pool) - (self.utr_len[0]-regsizes[0]) - ((self.seq_len*3)-regsizes[1])) for n in range(int(sum([self.seq_len, self.utr_pool_size[0], self.utr_pool_size[1]])+1))}
                else:
                    # nucleotide-resolution profile position == mutation-index position (identity)
                    seqmapper = range(self.seq_len)
                allpos[g] = [i for p in pos for i in range((seqmapper[p] - pwind),(seqmapper[p] + pwind)) if i>=0 and i<=sum(regsizes)]
            log.info("Compute and save distribution scores around identified differential sites.")
            self._score(gene_dict=allpos, filename=parquet_path)
        if os.path.exists(cache_path):
            log.info("Using cached site score windows from %s.", cache_path)
            # Materialize arrays into owned memory and close the file handle, so the
            # returned arrays do not stay backed by the open NpzFile zip buffers.
            with np.load(cache_path, allow_pickle=False) as cached:
                if is_frameshift:
                    required = ["ohe"] + [f"winds_{cond}_{frame}" for cond in conds
                                          for frame in ["score_0", "score_1", "score_2"]]
                else:
                    required = ["ohe"] + [f"winds_{cond}" for cond in conds]
                missing = [k for k in required if k not in cached.files]
                if missing:
                    log.warning("Cached site score windows at %s are stale (missing %s). Recomputing.",
                                cache_path, ", ".join(missing))
                else:
                    ohe = np.array(cached["ohe"])
                    # Guard against a cache whose peak count no longer matches the current
                    # differential sites (e.g. diffsites.csv changed since the cache was written).
                    n_peaks = sum(len(p) for p in p_per_gene.values())
                    if ohe.shape[0] != n_peaks:
                        log.warning("Cached site score windows at %s are stale (%d rows vs %d peaks). Recomputing.",
                                    cache_path, ohe.shape[0], n_peaks)
                    elif is_frameshift:
                        frame_winds = {cond: {frame: np.array(cached[f"winds_{cond}_{frame}"]) for frame in ["score_0", "score_1", "score_2"]} for cond in conds}
                        return frame_winds, ohe
                    else:
                        peak_winds = {cond: np.array(cached[f"winds_{cond}"]) for cond in conds}
                        return peak_winds, ohe
        frozen_fun = partial(
            sites_scores_fn,
            pwind=pwind,
            conds=conds,
            ctrlcond=ctrlcond,
            nucl=nucl,
            subdir=subdir
            )
        poolwrap, ohe = zip(*parallel_map(frozen_fun, p_per_gene.items(), threads=nthreads, chunksize=10,
                                          desc="Extracting importance scores around differential sites"))

        ohe = np.concatenate(ohe)

        # Flatten gene lists and cache to disk
        if is_frameshift:
            frame_winds = {k: {frame: np.concatenate([g[k][frame] for g in poolwrap]) for frame in ["score_0", "score_1", "score_2"]} for k in poolwrap[0].keys()}
            log.info("Saving site score windows to %s.", cache_path)
            save_dict = {"ohe": ohe}
            for cond, frames in frame_winds.items():
                for frame, arr in frames.items():
                    save_dict[f"winds_{cond}_{frame}"] = arr
            np.savez(cache_path, **save_dict)
            return frame_winds, ohe
        else:
            peak_winds = {k:np.concatenate([g[k] for g in poolwrap]) for k in poolwrap[0].keys()}
            log.info("Saving site score windows to %s.", cache_path)
            save_dict = {"ohe": ohe}
            for cond, arr in peak_winds.items():
                save_dict[f"winds_{cond}"] = arr
            np.savez(cache_path, **save_dict)
            return peak_winds, ohe

    def plot_sites_average(self, peak_avg, conds, ctrlcond, pwind, zoom=[], nucl=["A","C","G","T"], title=None, lims=None):
        '''
        Plot average importance scores around differential sites as sequence logos.

        Parameters
        ----------
        peak_avg : dict
            Dictionary of average scores per condition.
        conds : list of str
            List of condition names.
        ctrlcond : str or None
            Control condition name.
        pwind : int
            Window size around peaks.
        zoom : list, default=[]
            Zoom range [start, end] for the plot.
        nucl : list of str, default=["A","C","G","T"]
            List of nucleotides.
        title : str, optional
            Optional title for the sequence logo plot.
        lims : list, optional
            Optional y-axis limits [ymin, ymax] for the plots. If None, limits are determined automatically.

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure.
        '''
        plotconds  = conds if ctrlcond is None else [s for s in conds if s!=ctrlcond]
        width = pwind if not zoom else zoom[1]-zoom[0]
        ylims = [[],[]]
        fig, ax = plt.subplots(nrows=len(plotconds), figsize=(0.3*width, 1.3*len(plotconds)))
        for n,cond in enumerate(plotconds):
            meansis = pd.DataFrame(peak_avg[cond], index=list(range(-pwind, pwind)), columns = nucl).fillna(0.0)
            if zoom:
                meansis = meansis.iloc[zoom[0]:zoom[1]]
            axtemp = ax[n] if len(plotconds)>1 else ax
            # Display as RNA: relabel T->U for the logo only (colorblind_safe maps U to the
            # same color as T). Scores/columns stay DNA; this is purely a plotting change.
            nucl_logo = logomaker.Logo(meansis.rename(columns={"T": "U"}), color_scheme='colorblind_safe', ax=axtemp)
            nucl_logo.style_spines(visible=True)
            nucl_logo.ax.set_ylabel(cond, labelpad=-1)
            nucl_logo.ax.set_xlabel("Position relative to differential site (nt)")
            if title and n == 0:
                fig.suptitle(title)
            # Fix y axis to min max
            ylims[0].append(nucl_logo.ax.get_ylim()[0])
            ylims[1].append(nucl_logo.ax.get_ylim()[1])
            if n < (len(plotconds)-1):
                axtemp.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
                axtemp.set_xlabel("")
        # Set ylims and highlight regions
        if lims is None:
            lims = [min(ylims[0]), max(ylims[1])]
        if len(plotconds)>1:
            for a in ax:
                a.set_ylim(lims)
                Esite = Rectangle(xy=(-3.5, lims[0]), width=3, height=lims[1]-lims[0],
                                  facecolor="#ede7bc", zorder=-2, alpha=0.6)
                Psite = Rectangle(xy=(-0.5, lims[0]), width=3, height=lims[1]-lims[0],
                                  facecolor="#B6D0E2", zorder=-2, alpha=0.6)
                Asite = Rectangle(xy=(2.5, lims[0]), width=3, height=lims[1]-lims[0],
                                  facecolor="#f7d9d3", zorder=-2, alpha=0.6)
                a.add_patch(Esite)
                a.add_patch(Psite)
                a.add_patch(Asite)
        else:
            ax.set_ylim(lims)
            Esite = Rectangle(xy=(-3.5, lims[0]), width=3, height=lims[1]-lims[0],
                              facecolor="#ede7bc", zorder=-2, alpha=0.6)
            Psite = Rectangle(xy=(-0.5, lims[0]), width=3, height=lims[1]-lims[0],
                              facecolor="#B6D0E2", zorder=-2, alpha=0.6)
            Asite = Rectangle(xy=(2.5, lims[0]), width=3, height=lims[1]-lims[0],
                              facecolor="#f7d9d3", zorder=-2, alpha=0.6)
            ax.add_patch(Esite)
            ax.add_patch(Psite)
            ax.add_patch(Asite)
        fig.tight_layout()
        return fig

    def _compute_codon_aa_avg(self, peak_winds, ohe, nucl=["A", "C", "G", "T"]):
        '''
        Compute codon and amino acid level average ISM scores from nucleotide-level windows.

        Parameters
        ----------
        peak_winds : dict
            Dictionary of {cond: np.ndarray of shape (n_peaks, 2*pwind, 4)} containing
            nucleotide-level ISM scores (NaN where not the original nucleotide).
        ohe : np.ndarray
            One-hot encoded sequences of shape (n_peaks, 2*pwind, 4).
        nucl : list of str, default=["A","C","G","T"]
            Ordered list of nucleotides matching the 4-channel axis.

        Returns
        -------
        tuple of (dict, dict)
            codon_avg: {cond: DataFrame(cod_pos x codon)} - mean ISM score per codon
            aa_avg: {cond: DataFrame(cod_pos x amino_acid)} - mean ISM score per amino acid
        '''
        n_peaks, n_pos, _ = ohe.shape
        n_cod_eff = n_pos // 3
        n_pos_eff = n_cod_eff * 3

        # Index of original nucleotide at each (peak, nt_position)
        orig_nuc_idx = np.argmax(ohe, axis=-1)  # (n_peaks, n_pos)
        nucl_arr = np.array(nucl)

        # Codon indices and chars: (n_peaks, n_cod_eff, 3)
        orig_idx_cod = orig_nuc_idx[:, :n_pos_eff].reshape(n_peaks, n_cod_eff, 3)
        cod_chars = nucl_arr[orig_idx_cod]  # (n_peaks, n_cod_eff, 3) dtype str

        # ohe validity: a position is valid if any nt is 1 (ohe row is not all-zero)
        ohe_valid = (ohe[:, :n_pos_eff, :].sum(axis=-1) > 0).reshape(n_peaks, n_cod_eff, 3)
        # Codon fully valid only if all 3 nt positions are valid
        cod_valid_mask = ohe_valid.all(axis=-1)  # (n_peaks, n_cod_eff)

        codon_avg = {}
        aa_avg = {}

        for cond, winds in peak_winds.items():
            # Pick the score at the original nucleotide for each position
            nt_scores = np.take_along_axis(winds, orig_nuc_idx[:, :, np.newaxis], axis=-1).squeeze(-1)  # (n_peaks, n_pos)

            # Sum nt scores over the 3 positions of each codon
            nt_scores_cod = nt_scores[:, :n_pos_eff].reshape(n_peaks, n_cod_eff, 3)  # (n_peaks, n_cod_eff, 3)
            nan_mask = np.isnan(nt_scores_cod).any(axis=-1)  # (n_peaks, n_cod_eff)
            cod_scores = np.nansum(nt_scores_cod, axis=-1)  # (n_peaks, n_cod_eff)
            cod_scores[nan_mask | ~cod_valid_mask] = np.nan

            # Flatten for groupby
            cod_pos_flat = np.tile(np.arange(n_cod_eff), n_peaks)
            score_flat = cod_scores.flatten()
            codon_flat = np.array([''.join(cod_chars[n, c, :]) for n in range(n_peaks) for c in range(n_cod_eff)])
            aa_flat = np.array([Cdict.get(cod, '0') for cod in codon_flat])

            valid = ~np.isnan(score_flat) & (aa_flat != '0')
            df = pd.DataFrame({
                'cod_pos': cod_pos_flat[valid],
                'codon': codon_flat[valid],
                'aa': aa_flat[valid],
                'score': score_flat[valid]
            })

            codon_avg[cond] = df.groupby(['cod_pos', 'codon'])['score'].mean().unstack(fill_value=0.0)
            aa_avg[cond] = df.groupby(['cod_pos', 'aa'])['score'].mean().unstack(fill_value=0.0)

        return codon_avg, aa_avg

    def plot_sites_average_codon(self, codon_avg, conds, ctrlcond, pwind, zoom=[], title=None, lims=None):
        '''
        Plot average codon-level ISM scores around differential sites as stacked glyph logos.

        Each codon at each window position is drawn as a stacked letter (positive scores
        above the axis, negative below). Colors reflect amino acid chemical properties.

        Parameters
        ----------
        codon_avg : dict
            Dictionary {cond: DataFrame(cod_pos x codon)} of mean codon-level ISM scores,
            as returned by _compute_codon_aa_avg.
        conds : list of str
            List of condition names.
        ctrlcond : str or None
            Control condition name (excluded from plotting).
        pwind : int
            Nucleotide half-window size used to define E/P/A-site highlight positions.
        zoom : list, optional
            Optional x-axis limits [xmin, xmax] for the plots.
        title : str, optional
            Optional title for the codon logo plot.
        lims : list, optional
            Optional y-axis limits [ymin, ymax] for the plots. If None, limits are determined automatically.

        Returns
        -------
        matplotlib.figure.Figure
        '''
        codon_color_palette = {
            'GCT': 'black', 'GCC': 'black', 'GCA': 'black', 'GCG': 'black',
            'TTT': 'black', 'TTC': 'black', 'ATT': 'black', 'ATC': 'black', 'ATA': 'black',
            'TTA': 'black', 'TTG': 'black', 'CTT': 'black', 'CTC': 'black', 'CTA': 'black', 'CTG': 'black',
            'ATG': 'black', 'CCT': 'black', 'CCC': 'black', 'CCA': 'black', 'CCG': 'black',
            'TGG': 'black', 'GTT': 'black', 'GTC': 'black', 'GTA': 'black', 'GTG': 'black',
            'TGT': 'green', 'TGC': 'green', 'GGT': 'green', 'GGC': 'green', 'GGA': 'green', 'GGG': 'green',
            'TCT': 'green', 'TCC': 'green', 'TCA': 'green', 'TCG': 'green', 'AGT': 'green', 'AGC': 'green',
            'ACT': 'green', 'ACC': 'green', 'ACA': 'green', 'ACG': 'green', 'TAT': 'green', 'TAC': 'green',
            'GAT': 'red', 'GAC': 'red', 'GAA': 'red', 'GAG': 'red',
            'CAT': 'blue', 'CAC': 'blue', 'AAA': 'blue', 'AAG': 'blue',
            'CGT': 'blue', 'CGC': 'blue', 'CGA': 'blue', 'CGG': 'blue', 'AGA': 'blue', 'AGG': 'blue',
            'AAT': 'purple', 'AAC': 'purple', 'CAA': 'purple', 'CAG': 'purple',
            'TAA': 'peru', 'TAG': 'peru', 'TGA': 'peru',
        }

        plotconds = conds if ctrlcond is None else [s for s in conds if s != ctrlcond]
        pwind_cod = pwind // 3

        # Number of codon positions from any condition's DataFrame
        sample_df = next(iter(codon_avg.values()))
        n_cod = len(sample_df)

        fig, ax = plt.subplots(nrows=len(plotconds), figsize=(max(4, 0.35 * n_cod), 1.5 * len(plotconds)))
        if title:
            fig.suptitle(title)
        all_ymins, all_ymaxs = [], []

        for n, cond in enumerate(plotconds):
            axtemp = ax[n] if len(plotconds) > 1 else ax
            meansis = codon_avg[cond].copy()
            meansis.index = meansis.index - pwind_cod
            if zoom:
                meansis = meansis.iloc[zoom[0]:zoom[1]]

            melted = meansis.reset_index().melt(id_vars='cod_pos', var_name='codon', value_name='value')
            melted['value'] = melted['value'].fillna(0.0)
            melted['abs'] = melted['value'].abs()
            melted = melted.sort_values('abs')

            zoomed_positions = sorted(meansis.index.tolist())
            n_zoomed = len(zoomed_positions)
            pos_to_idx = {p: i for i, p in enumerate(zoomed_positions)}  # <-- replaces p + pwind_cod

            posmat = np.zeros((n_zoomed, 2))  # sized to zoom, not full n_cod

            for _, row in melted.iterrows():
                p = int(row['cod_pos'])
                pidx = pos_to_idx.get(p)
                if pidx is None:
                    continue                   # skip positions outside zoom
                v = float(row['value'])
                if v == 0.0:
                    continue
                color = codon_color_palette.get(row['codon'], 'gray')
                # Display as RNA: draw the codon with T->U (color is still looked up on the
                # DNA codon above, so nothing but the glyph text changes).
                codon_disp = row['codon'].replace('T', 'U')
                if v > 0:
                    logomaker.Glyph(c=codon_disp, p=p, width=1.0,
                                    floor=posmat[pidx, 0], ceiling=posmat[pidx, 0] + v,
                                    ax=axtemp, color=color)
                    posmat[pidx, 0] += v
                else:
                    logomaker.Glyph(c=codon_disp, p=p, width=1.0,
                                    floor=posmat[pidx, 1] + v, ceiling=posmat[pidx, 1],
                                    ax=axtemp, color=color, flip=True, mirror=True)
                    posmat[pidx, 1] += v

            x_lo = zoomed_positions[0] - 0.5
            x_hi = zoomed_positions[-1] + 0.5
            axtemp.set_xlim([x_lo, x_hi])
            axtemp.set_ylabel(cond, labelpad=-1)
            all_ymins.append(np.min(posmat[:, 1]))
            all_ymaxs.append(np.max(posmat[:, 0]))
            if n < len(plotconds) - 1:
                axtemp.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
                axtemp.set_xlabel('')
            else:
                axtemp.set_xlabel("Position relative to P site (codon)")

        # Unified ylim and E/P/A site highlights
        ymin = min(all_ymins) if all_ymins else -1
        ymax = max(all_ymaxs) if all_ymaxs else 1
        if lims is not None:
            ymin, ymax = lims
        eh = ymax - ymin
        axes_iter = list(ax) if len(plotconds) > 1 else [ax]
        for a in axes_iter:
            a.set_ylim([ymin, ymax])
            a.add_patch(Rectangle(xy=(-1.5, ymin), width=1, height=eh,
                                  facecolor='#ede7bc', zorder=-2, alpha=0.6))
            a.add_patch(Rectangle(xy=(-0.5, ymin), width=1, height=eh,
                                  facecolor='#B6D0E2', zorder=-2, alpha=0.6))
            a.add_patch(Rectangle(xy=(0.5, ymin), width=1, height=eh,
                                  facecolor='#f7d9d3', zorder=-2, alpha=0.6))
            ticks = a.get_xticks()
            a.set_xticks(ticks)
            a.set_xticklabels(['E' if int(round(t)) == -1 else 'P' if int(round(t)) == 0 else 'A' if int(round(t)) == 1 else str(int(round(t))) for t in ticks])
        fig.tight_layout()
        return fig

    def plot_sites_average_aa(self, aa_avg, conds, ctrlcond, pwind, zoom=[], title=None, lims=None):
        '''
        Plot average amino-acid-level ISM scores around differential sites as sequence logos.

        Parameters
        ----------
        aa_avg : dict
            Dictionary {cond: DataFrame(cod_pos x amino_acid)} of mean AA-level ISM scores,
            as returned by _compute_codon_aa_avg.
        conds : list of str
            List of condition names.
        ctrlcond : str or None
            Control condition name (excluded from plotting).
        pwind : int
            Nucleotide half-window size used to define E/P/A-site highlight positions.
        zoom : list, optional
            Optional x-axis limits [xmin, xmax] for the plots.
        title : str, optional
            Optional title for the amino acid logo plot.
        lims : list, optional
            Optional y-axis limits [ymin, ymax] for the plots. If None, limits are determined automatically.

        Returns
        -------
        matplotlib.figure.Figure
        '''
        aa_color_palette = {
            'A': 'black', 'F': 'black', 'I': 'black', 'L': 'black', 'M': 'black',
            'P': 'black', 'V': 'black', 'W': 'black',
            'C': 'green', 'G': 'green', 'S': 'green', 'T': 'green', 'Y': 'green',
            'D': 'red', 'E': 'red',
            'H': 'blue', 'K': 'blue', 'R': 'blue',
            'N': 'purple', 'Q': 'purple',
            '_': 'peru',
        }

        plotconds = conds if ctrlcond is None else [s for s in conds if s != ctrlcond]
        pwind_cod = pwind // 3

        sample_df = next(iter(aa_avg.values()))
        n_cod = len(sample_df)

        fig, ax = plt.subplots(nrows=len(plotconds), figsize=(max(4, 0.35 * n_cod), 1.5 * len(plotconds)))
        if title:
            fig.suptitle(title)
        ylims = [[], []]

        for n, cond in enumerate(plotconds):
            axtemp = ax[n] if len(plotconds) > 1 else ax
            meansis = aa_avg[cond].copy()
            meansis.index = meansis.index - pwind_cod
            if zoom:
                meansis = meansis.iloc[zoom[0]:zoom[1]]
            color_scheme = {aa: aa_color_palette.get(aa, 'gray') for aa in meansis.columns}
            aa_logo = logomaker.Logo(meansis, color_scheme=color_scheme, ax=axtemp)
            aa_logo.style_spines(visible=True)
            aa_logo.ax.set_ylabel(cond, labelpad=-1)
            ylims[0].append(aa_logo.ax.get_ylim()[0])
            ylims[1].append(aa_logo.ax.get_ylim()[1])
            if n < len(plotconds) - 1:
                axtemp.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
                axtemp.set_xlabel('')
            else:
                axtemp.set_xlabel("Position relative to P site (codon)")

        if lims is None:
            lims = [min(ylims[0]), max(ylims[1])]
        eh = lims[1] - lims[0]
        axes_iter = list(ax) if len(plotconds) > 1 else [ax]
        for a in axes_iter:
            a.set_ylim(lims)
            a.add_patch(Rectangle(xy=(-1.5, lims[0]), width=1, height=eh,
                                  facecolor='#ede7bc', zorder=-2, alpha=0.6))
            a.add_patch(Rectangle(xy=(-0.5, lims[0]), width=1, height=eh,
                                  facecolor='#B6D0E2', zorder=-2, alpha=0.6))
            a.add_patch(Rectangle(xy=(0.5, lims[0]), width=1, height=eh,
                                  facecolor='#f7d9d3', zorder=-2, alpha=0.6))
            ticks = a.get_xticks()
            a.set_xticks(ticks)
            a.set_xticklabels(['E' if int(round(t)) == -1 else 'P' if int(round(t)) == 0 else 'A' if int(round(t)) == 1 else str(int(round(t))) for t in ticks])
        fig.tight_layout()
        return fig

    def plot_seqlet_sites(self, fhfile, peaksdf):
        '''
        Plot heatmap showing where seqlets are located across genomic regions.

        Parameters
        ----------
        fhfile : str
            Path to HDF5 file containing TF-MoDISco results.
        peaksdf : pd.DataFrame
            DataFrame containing peak positions.

        Returns
        -------
        matplotlib.figure.Figure
            The generated heatmap figure.
        '''
        fh = h5.File(fhfile)
        type_patterns = [p for p in ["pos_patterns","neg_patterns"] if fh[p].keys()]
        if not type_patterns:
            fh.close()
            fig, ax = plt.subplots(figsize=(8, 2))
            ax.text(0.5, 0.5, "No patterns found", ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
            return fig
        cmap = LinearSegmentedColormap.from_list('Custom',["#eeeeee","#d0d0d0","#eeeeee","#ff0000"], 4)
        fig, ax = plt.subplots(nrows=len(type_patterns), figsize=(8, 2*len(type_patterns)))
        for ax_n, type_pattern in enumerate(type_patterns):
            subax = ax[ax_n] if len(type_patterns)>1 else ax
            if type_pattern in fh.keys():
                patterns = fh[type_pattern].keys()
                patterns = [p for _,p in sorted(zip([int(s.split("_")[-1]) for s in patterns],patterns))]
                heatdf = pd.DataFrame(np.tile(np.concatenate([np.zeros(self.utr_pool_size[0]),np.full(self.seq_len,1),np.full(self.utr_pool_size[1],2)]),(len(patterns),1)), columns=np.arange(sum([self.seq_len, self.utr_pool_size[0], self.utr_pool_size[1]])), index=patterns)
                for p in patterns:
                    heatdf.loc[p,np.unique(peaksdf.iloc[np.array(fh[type_pattern][p]["seqlets"]["example_idx"]),1].values)] = 3
                sns.heatmap(heatdf, cmap = cmap, ax= subax)
                subax.set_ylabel("Motif pattern")
                subax.set_xlabel("Sequence position along transcript")
                colorbar = subax.collections[0].colorbar
                r = colorbar.vmax - colorbar.vmin
                n=4; colorbar.set_ticks([colorbar.vmin + 0.5 * r / (n) + r * i / (n) for i in range(n)])
                colorbar.set_ticklabels(['UTR5', 'CDS', 'UTR3', 'peaks'])
        fh.close()
        fig.tight_layout()
        return fig
    
    def _run_modisco_distribution(self, p_per_gene, peak_winds, ohe, conds, ctrlname, subdir="", max_seqlets=1000):
        '''
        Run TF-MoDISco on distribution-based importance scores around CDS differential sites.

        Parameters
        ----------
        p_per_gene : dict
            Dictionary mapping gene names to peak positions.
        peak_winds : dict
            Dictionary of importance scores per condition.
        ohe : np.ndarray
            One-hot encoded sequences.
        conds : list of str
            List of condition names.
        ctrlname : str
            Control condition name or "mean".
        subdir : str, default=""
            Sub-directory under the condition-set distribution directory to read/write TF-MoDISco
            results from (e.g. "pausing"). Defaults to "" so codon-resolution paths are unchanged.
        max_seqlets : int, default=1000
            Maximum number of seqlets per metacluster passed to TF-MoDISco.

        Returns
        -------
        None
            Saves TF-MoDISco results and adds plots to report.
        '''
        sitedir = os.path.join(self.interpretation_dir, "distribution", "-".join(conds) + "_vs_" + ctrlname)
        if subdir:
            sitedir = os.path.join(sitedir, subdir)
        peaksdf = np.array([(k, p) for k in p_per_gene.keys() for p in p_per_gene[k]])
        peaksdf = pd.DataFrame(np.tile(peaksdf, (len(peak_winds), 1)), columns=["gene", "pos"]).astype(dtype={"gene": "str", "pos": "int"})
        # Prepare data
        snp_attr = np.nan_to_num(np.concatenate(list(peak_winds.values())))
        snp_seq = np.tile(ohe,(len(conds),1,1))
        if not all([os.path.exists(os.path.join(sitedir, "modisco_results.h5")),
                    os.path.exists(os.path.join(sitedir, "modisco_report"))]):
            log.info("Running TF-MoDISco for distribution analysis (%s).", "-".join(conds) + "_vs_" + ctrlname)
            # Run modisco
            sliding_window_size = 11
            max_seqlets_per_metacluster = max_seqlets
            def seqlet_selection_fn(seqlets: list[Seqlet]) -> list[Seqlet]:
                scores = []; indices = []
                for i, seqlet in enumerate(seqlets):
                    flank = int(0.5 * (len(seqlet) - sliding_window_size))
                    scores.append(np.abs(np.sum(seqlet.contrib_scores[flank:-flank])))
                    indices.append(i)
                scores = np.array(scores)
                indices = np.array(indices)
                ordered = indices[np.argsort(scores)[::-1]]
                selection = ordered[:max_seqlets_per_metacluster]
                return [seqlets[i] for i in selection]
            pos_patterns, neg_patterns = TFMoDISco(snp_seq, snp_attr, n_leiden_runs=10, sliding_window_size=sliding_window_size,
                                                flank_size=5, final_min_cluster_size=10, verbose=True, revcomp=False,
                                                max_seqlets_per_metacluster=max_seqlets_per_metacluster, seqlet_selection_fn=seqlet_selection_fn)
            # Save results
            if pos_patterns is None:
                pos_patterns = []
            if neg_patterns is None:
                neg_patterns = []
            log.info(("-".join(conds)+"_vs_"+ctrlname)+ f": Found {len(pos_patterns)} positive patterns" f" and {len(neg_patterns)} negative patterns.")
            
            if not os.path.exists(sitedir):
                os.makedirs(sitedir, exist_ok=True)
            save_hdf5(os.path.join(sitedir, "modisco_results.h5"), pos_patterns, neg_patterns, snp_seq.shape[1])
            # Generate report
            report_motifs(os.path.join(sitedir, "modisco_results.h5"),
                        os.path.join(sitedir, "modisco_report"),
                        img_path_suffix=os.path.join(sitedir, "modisco_report/"),
                        meme_motif_db=None, is_writing_tomtom_matrix=False, revcomp=False)
        else:
            log.info("Using cached TF-MoDISco distribution results for %s.", "-".join(conds) + "_vs_" + ctrlname)
        # Print
        add_text("The TF-MoDISco motif report below shows sequence patterns discovered around differential sites. "
                 "Each motif represents a recurring sequence pattern associated with condition-specific ribosome occupancy changes.")
        add_html(os.path.join(sitedir, "modisco_report","motifs.html"))
        add_text("The seqlet position plot visualizes where discovered motifs occur across the sequence windows around differential sites. "
                 "Each row corresponds to one seqlet (motif instance).")
        fig = plot_pos_seqlets(os.path.join(sitedir, "modisco_results.h5"), snp_seq.shape[1])
        fig.savefig(os.path.join(sitedir, "modisco_report","seqlet_window_positions.pdf"))
        add_plot(fig)
        plt.close()
        add_text("The heatmap below shows the location of identified seqlets along the trancripts. "
                 "Red positions indicate where differential sites with the corresponding motif patterns are located. "
                 "This reveals whether motifs are enriched in specific genomic regions (5'UTR, CDS, 3'UTR).")
        fig = self.plot_seqlet_sites(os.path.join(sitedir, "modisco_results.h5"), peaksdf)
        fig.savefig(os.path.join(sitedir, "modisco_report","seqlet_transcript_sites.pdf"))
        add_plot(fig)
        plt.close()
        return

    def _load_diffsites(self, c, diffsites, threads, sitedir, analysis):
        '''
        Load cached differential sites or compute them for a given analysis.

        Parameters
        ----------
        c : list of str
            Condition names in this condition set.
        diffsites : int or float
            Number of top sites (>1) or p-value threshold (<=1).
        threads : int
            Number of threads for parallel processing.
        sitedir : str
            Directory in which ``diffsites.csv`` is cached.
        analysis : str
            "pausing" or "frameshift".

        Returns
        -------
        tuple
            (p_idx DataFrame, p_per_gene dict).
        '''
        os.makedirs(sitedir, exist_ok=True)
        csv_path = os.path.join(sitedir, "diffsites.csv")
        if not os.path.exists(csv_path):
            p_idx = self._get_diffsites(c, diffsites, threads, analysis=analysis)
            p_idx.to_csv(csv_path, header=False, index=False)
        else:
            log.info("Using cached differential sites from %s.", csv_path)
            p_idx = pd.read_csv(csv_path, header=None)
        p_per_gene = p_idx.pivot_table(index=0, aggfunc=lambda x: list(x), dropna=False)[1].to_dict()
        return p_idx, p_per_gene

    def _plot_diffsites_histogram(self, p_idx, sitedir):
        '''
        Add the CDS-position histogram of differential sites to the report and save it.

        Parameters
        ----------
        p_idx : pd.DataFrame
            Differential sites (gene in column 0, position in column 1).
        sitedir : str
            Directory in which to save the histogram PDF.
        '''
        add_text("The histogram below shows the percentage of differential ribosome occupancy sites "
                 "per CDS position bin, corrected for the number of genes of at least that length "
                 "(position 0 = CDS start).")
        fig, ax = plt.subplots(figsize=(8, 4))
        if self.seq_resolution == "nucleotide":
            p_idx_ser = p_idx.set_index(0)[1].sub(self.x_raw["utr5_len"])
            gene_lengths = self.x_raw["cds_len"]
        else:
            p_idx_ser = p_idx[1] - self.utr_pool_size[0]
            gene_lengths = self.x_raw["cds_len"] / 3
        counts, bin_edges = np.histogram(p_idx_ser.dropna(), bins=50)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        n_genes_at_pos = np.array([(gene_lengths >= pos).sum() for pos in bin_centers])
        pct = np.where(n_genes_at_pos > 0, counts / n_genes_at_pos * 100, 0)
        ax.bar(bin_centers, pct, width=np.diff(bin_edges), color='#3d6758', align='center')
        ax.set_xlabel("Position relative to CDS start")
        ax.set_ylabel("% of differential sites")
        fig.savefig(os.path.join(sitedir, "diffsites_histogram.pdf"))
        add_plot(fig)
        plt.close()

    def _distribution_pausing_block(self, c, ctrl, ctrlname, base_sitedir, subdir, diffsites, pwind, threads, max_seqlets):
        '''
        Run the ribosome-pausing distribution analysis for one condition set: differential codon
        occupancy sites, nucleotide/codon/amino-acid metagene logos and TF-MoDISco.

        Works for both resolutions: codon (``subdir=""``, paths unchanged) and nucleotide
        (``subdir="pausing"``); the peak windows have identical structure in both cases.
        '''
        sitedir = os.path.join(base_sitedir, subdir) if subdir else base_sitedir
        p_idx, p_per_gene = self._load_diffsites(c, diffsites, threads, sitedir, analysis="pausing")
        log.info("Extracting scores around identified differential occupancy sites.")
        peak_winds, ohe = self._extract_sites_scores(p_per_gene, pwind, c, ctrl, threads, analysis="pausing")
        # Histogram of CDS differential sites relative to CDS start
        self._plot_diffsites_histogram(p_idx, sitedir)
        # Nucleotide-level average scores
        peak_avg = {k: np.nanmean(v, axis=0) for k, v in peak_winds.items()}
        with pd.ExcelWriter(os.path.join(sitedir, "diffsites_average_logo.xlsx"), engine="openpyxl") as writer:
            for sheet_name in c:
                df = pd.DataFrame(peak_avg[sheet_name], index=list(range(-pwind, pwind)), columns=["A","C","G","T"]).fillna(0.0)
                df.to_excel(writer, sheet_name=sheet_name)
        add_subheading("Average nucleotide scores among differential sites.")
        add_text("The sequence logo plots below show the average nucleotide importance scores in windows around differential sites. "
                 "The height of each letter indicates how strongly that nucleotide at that position influences ribosome occupancy. "
                 "Highlighted regions indicate codon positions (-3 nt for E site, +3 nt for A site) relative to the differential site center (P site).")
        fig = self.plot_sites_average(peak_avg, c, ctrl, pwind)
        fig.savefig(os.path.join(sitedir, "diffsites_average_logo.pdf"))
        add_plot(fig)
        plt.close()
        # Codon and amino acid level plots
        codon_avg, aa_avg = self._compute_codon_aa_avg(peak_winds, ohe)
        with pd.ExcelWriter(os.path.join(sitedir, "diffsites_average_codon_logo.xlsx"), engine="openpyxl") as writer:
            for sheet_name in c:
                df = codon_avg[sheet_name]
                df.to_excel(writer, sheet_name=sheet_name)
        add_subheading("Average codon scores among differential sites.")
        add_text("The codon logo plots below show the average codon-level importance scores summed over all three nucleotide positions of each codon. "
                    "Each codon is drawn as a stacked letter; positive heights indicate codons that increase ribosome occupancy at the differential site, "
                    "negative heights indicate codons that decrease it. "
                    "Colors reflect amino acid chemical properties: black = hydrophobic, green = polar/small, red = acidic, blue = basic, purple = amide, brown = stop. "
                    "Highlighted regions mark E (yellow), P (blue), and A (red) ribosome sites.")
        fig = self.plot_sites_average_codon(codon_avg, c, ctrl, pwind)
        fig.savefig(os.path.join(sitedir, "diffsites_average_codon_logo.pdf"))
        add_plot(fig)
        plt.close()
        with pd.ExcelWriter(os.path.join(sitedir, "diffsites_average_aa_logo.xlsx"), engine="openpyxl") as writer:
            for sheet_name in c:
                df = aa_avg[sheet_name]
                df.to_excel(writer, sheet_name=sheet_name)
        add_subheading("Average amino acid scores among differential sites.")
        add_text("The amino acid logo plots below show the average per-codon ISM scores aggregated by amino acid identity. "
                    "The score for each amino acid at each codon position is the mean over all codons encoding that amino acid. "
                    "Colors follow the same chemical-property scheme as for codons. "
                    "Highlighted regions mark E (yellow), P (blue), and A (red) ribosome sites.")
        fig = self.plot_sites_average_aa(aa_avg, c, ctrl, pwind)
        fig.savefig(os.path.join(sitedir, "diffsites_average_aa_logo.pdf"))
        add_plot(fig)
        plt.close()
        # TF-MoDISco
        log.info("Running TF-MoDISco for %s (pausing)." % ("-".join(c)))
        add_subheading("TF-MoDISco among differential sites.")
        add_text("TF-MoDISco analysis identifies recurring sequence motifs in the windows around differential sites. "
                "This reveals specific sequence patterns that are enriched at positions where ribosome occupancy differs between conditions, "
                "suggesting these motifs may mediate condition-specific regulation.")
        self._run_modisco_distribution(p_per_gene, peak_winds, ohe, c, ctrlname, subdir=subdir, max_seqlets=max_seqlets)

    def _distribution_frameshift_block(self, c, ctrl, ctrlname, base_sitedir, subdir, diffsites, pwind, threads):
        '''
        Run the frameshift distribution analysis for one condition set (nucleotide resolution only):
        differential in- vs out-of-frame sites and per-frame (F0/F1/F2) nucleotide/codon/amino-acid
        metagene logos. No TF-MoDISco is run for frameshift.
        '''
        sitedir = os.path.join(base_sitedir, subdir)
        p_idx, p_per_gene = self._load_diffsites(c, diffsites, threads, sitedir, analysis="frameshift")
        log.info("Extracting scores around identified frameshift sites.")
        frame_winds, ohe = self._extract_sites_scores(p_per_gene, pwind, c, ctrl, threads, analysis="frameshift")
        # Histogram of CDS differential sites relative to CDS start
        self._plot_diffsites_histogram(p_idx, sitedir)
        frame_labels = [("score_0", "F0"), ("score_1", "F1"), ("score_2", "F2")]
        # Nucleotide-level average scores, per frame
        for score_key, label in frame_labels:
            frame_avg = {k: np.nanmean(v[score_key], axis=0) for k, v in frame_winds.items()}
            add_subheading(f"Average nucleotide scores among differential sites ({label}).")
            add_text(f"The sequence logo plot below shows the average nucleotide importance scores in windows around differential sites for the {label} frame. "
                     "The height of each letter indicates how strongly that nucleotide at that position influences ribosome occupancy in this frame. "
                     "Highlighted regions indicate codon positions (-3 nt for E site, +3 nt for A site) relative to the differential site center (P site).")
            fig = self.plot_sites_average(frame_avg, c, ctrl, pwind, title=f"{label} frame")
            fig.savefig(os.path.join(sitedir, f"diffsites_average_logo_{score_key}.pdf"))
            add_plot(fig)
            plt.close()
        # Codon and amino acid level plots, per frame
        for score_key, label in frame_labels:
            frame_winds_dict = {k: v[score_key] for k, v in frame_winds.items()}
            codon_avg, aa_avg = self._compute_codon_aa_avg(frame_winds_dict, ohe)
            add_subheading(f"Average codon scores among differential sites ({label}).")
            add_text(f"The codon logo plots below show the average codon-level importance scores summed over all three nucleotide positions of each codon for the {label} frame. "
                        "Each codon is drawn as a stacked letter; positive heights indicate codons that increase ribosome occupancy at the differential site, "
                        "negative heights indicate codons that decrease it. "
                        "Colors reflect amino acid chemical properties: black = hydrophobic, green = polar/small, red = acidic, blue = basic, purple = amide, brown = stop. "
                        "Highlighted regions mark E (yellow), P (blue), and A (red) ribosome sites.")
            fig = self.plot_sites_average_codon(codon_avg, c, ctrl, pwind, title=f"{label} frame")
            fig.savefig(os.path.join(sitedir, f"diffsites_average_codon_logo_{score_key}.pdf"))
            add_plot(fig)
            plt.close()
            add_subheading(f"Average amino acid scores among differential sites ({label}).")
            add_text(f"The amino acid logo plots below show the average per-codon ISM scores aggregated by amino acid identity for the {label} frame. "
                        "The score for each amino acid at each codon position is the mean over all codons encoding that amino acid. "
                        "Colors follow the same chemical-property scheme as for codons. "
                        "Highlighted regions mark E (yellow), P (blue), and A (red) ribosome sites.")
            fig = self.plot_sites_average_aa(aa_avg, c, ctrl, pwind, title=f"{label} frame")
            fig.savefig(os.path.join(sitedir, f"diffsites_average_aa_logo_{score_key}.pdf"))
            add_plot(fig)
            plt.close()

    def extract_distribution_motifs(self, conds=None, ctrl_cond=None, diffsites=5000, pwind=30, threads=1, max_seqlets=1000, filename=None): #meta + modisco
        '''
        Extract motifs based on distribution predictions around differential sites.

        Parameters
        ----------
        conds : list of str, optional
            List of condition names. If None, uses all conditions.
        ctrl_cond : str, dict, optional
            Control condition specification.
        diffsites : int, float, default=5000
            If int >1, number of top differential sites to analyze.
            If float 0-1, p-value threshold for differential sites.
        pwind : int, default=30
            Nucleotide window size around each differential site.
        threads : int, default=1
            Number of threads for multiprocessing.
        max_seqlets : int, default=1000
            Maximum number of seqlets per metacluster passed to TF-MoDISco.
        filename : str, optional
            Output filename for the report.

        Returns
        -------
        None
            Saves motif analysis results and adds plots to report.
        '''
        # Some checks
        if filename:
            create_report(os.path.join(self.interpretation_dir, filename), title="RiboInterpretation Distribution")
            # Add introduction
            if self.scoring=="ISM":
                score_expl = "ISM systematically mutates each nucleotide position and measures the resulting change in model predictions, revealing which sequence positions are most important for determining ribosome counts and distribution."
            else:
                score_expl = ""
            add_text("This report presents the interpretation of the RiboNet model. "
                    f"The analysis scores sequence features that influence ribosome occupancy predictions using {self.scoring} importance scores. "
                    f"{score_expl}")
            add_page_break()
            add_toc()
        if not os.path.exists(os.path.join(self.interpretation_dir,"distribution")):
            os.makedirs(os.path.join(self.interpretation_dir,"distribution"),exist_ok=True)
        # Define control conditions if not done before
        ctrl_cond = self._define_ctrl_conditions(ctrl_cond, conds=conds)
        if conds is None:
            conds = list(self.y_raw["cds"].columns)
        # Define condition sets
        if ctrl_cond is None:
            condsets = [(conds, ctrl_cond)]
        else:
            condsets = [([k for k,v in ctrl_cond.items() if c==v],c) for c in list(set(ctrl_cond.values()))]
        if pwind % 3 != 0:
            log.warning("pwind=%d is not a multiple of 3; codon/amino-acid metagene logos will not "
                        "be aligned to the canonical reading frame (P site). Use a multiple of 3.", pwind)
        for c,ctrl in condsets:
            ctrlname = ctrl if ctrl is not None else "mean"
            add_heading("Distribution differential sites vs %s" % ctrlname)
            add_text("This section identifies sites of differential ribosome occupancy between conditions and analyzes sequence features around these sites. "
                     "Differential sites are positions where ribosome distribution patterns differ significantly between conditions, potentially indicating "
                     "condition-specific regulatory elements or pausing sites.")
            log.info("Identifying differential distribution sites among %s." % ("-".join(c)))
            base_sitedir = os.path.join(self.interpretation_dir,"distribution","-".join(c)+"_vs_"+ctrlname)
            if self.seq_resolution == "nucleotide":
                # Frameshift and ribosome pausing are different biology and are analysed separately.
                # Frameshift only makes sense at nucleotide resolution and gets metagene plots only.
                add_subheading("Frameshift analysis")
                add_text("The frameshift analysis identifies sites where the in-frame vs out-of-frame ribosome occupancy "
                         "changes between conditions, and characterises the sequence features around them separately for "
                         "each reading frame (F0 = canonical/in-frame, F1 and F2 = the two shifted frames).")
                self._distribution_frameshift_block(c, ctrl, ctrlname, base_sitedir, "frameshift", diffsites, pwind, threads)
                # Ribosome pausing runs at nucleotide resolution too, adapting the codon pipeline
                # (differential codon occupancy from summed nt triplets), plus TF-MoDISco.
                add_subheading("Ribosome pausing analysis")
                add_text("The ribosome pausing analysis identifies sites of differential codon occupancy (summing the three "
                         "nucleotides of each codon) and characterises the surrounding sequence features, mirroring the "
                         "codon-resolution pausing pipeline, including TF-MoDISco motif discovery.")
                self._distribution_pausing_block(c, ctrl, ctrlname, base_sitedir, "pausing", diffsites, pwind, threads, max_seqlets)
            else:
                # Codon resolution: ribosome pausing only (paths unchanged, subdir="").
                self._distribution_pausing_block(c, ctrl, ctrlname, base_sitedir, "", diffsites, pwind, threads, max_seqlets)
            add_page_break()
            if filename:
                gc.collect()
                add_interactive_guide()
                finalize_report()
        return

    def generate_report(self, ctrl_cond=None, threads=1, motif_db="default", diffsites=5000, pwind=30, max_seqlets=1000, filename="interpretation.pdf"):
        '''
        Generate a pdf document summarizing the interpretation results.
        If ctrl_cond is provided, it is used as the control condition for contrasts.

        Parameters
        ----------
        ctrl_cond : str, dict, optional
            Control condition.
            If str refers to a track name in the sample file, used for all tracks.
            If str refers to a condition name in the sample file, used for all tracks with the matching sample readlength.
            If dict, should contain track names as keys and control condition names as values.
            If not provided, the average across all conditions is used as control.
        threads : int, default=1
            Number of threads to use in multiprocessing.
        motif_db : str, None, default="default"
            If None, no database search is performed.
            If str, path to the meme file should be provided.
        diffsites : int, float, default=5000
            If int >1, take top most differential number of sites.
            If float between 0-1, take differential sites below this p-value.
        pwind : int, default=30
            Nucleotide window to take at each side of differential sites.
        max_seqlets : int, default=1000
            Maximum number of seqlets per metacluster passed to TF-MoDISco (counts and distribution).

        Returns
        -------
        Saved pdf results in interpretation directory.
        '''
        # Define control conditions
        ctrl_cond = self._define_ctrl_conditions(ctrl_cond)
        
        # Initialize report
        log.info("Initalizing RiboInterpretation report.")
        create_report(os.path.join(self.interpretation_dir, filename), title="RiboInterpretation Report")
        
        # Add introduction
        if self.scoring=="ISM":
            score_expl = "ISM systematically mutates each nucleotide position and measures the resulting change in model predictions, revealing which sequence positions are most important for determining ribosome counts and distribution."
        else:
            score_expl = ""
        add_text("This report presents the interpretation of RiboNet model predictions through feature importance analysis. "
                 f"The analysis identifies sequence features that influence ribosome occupancy predictions. "
                 f"{score_expl}")
        add_text("The report is organized into two main sections:")
        add_text("1. Count-based motif analysis: Identifies sequence motifs that predict absolute ribosome counts and relative changes between conditions.")
        add_text("2. Distribution-based motif analysis: Focuses on sites with differential ribosome occupancy patterns and identifies "
                 "sequence features associated with condition-specific changes in ribosome distribution.")
        add_page_break()
        add_toc()

        # Run counts motifs analysis if counts aren't suppressed
        if self.alpha != 0:
            self.extract_counts_motifs(ctrl_cond=ctrl_cond, threads=threads, motif_db=motif_db, max_seqlets=max_seqlets)
            gc.collect()
        # Run distribution motifs analysis if distributions aren't suppressed
        if self.beta != 0:
            self.extract_distribution_motifs(ctrl_cond=ctrl_cond, diffsites=diffsites, pwind=pwind, threads=threads, max_seqlets=max_seqlets)
            gc.collect()
        add_interactive_guide()
        finalize_report()
        return

    def extract_gene_scores(self, gene, conds=None, ctrlcond=None, countsonly=True):
        '''
        Extract importance scores for a specific gene.

        Parameters
        ----------
        gene : str
            Gene identifier.
        conds : list of str, optional
            List of condition names. If None, uses all conditions.
        ctrlcond : str, optional
            Control condition name for differential scoring.
        countsonly : bool, default=True
            If True, only extract count-based scores.
            If False, also extract distribution scores.

        Returns
        -------
        tuple
            Two tuples: (absolute_scores, differential_scores)
            Each contains [dcounts, dprofile, region_sizes]
        '''
        if countsonly or self.save_distribution:
            batch_num = int([k for k,v in self.batch_splits.items() if gene in v][0])
            # Upload contrasts
            contrasts = pd.read_parquet(os.path.join(self.interpretation_dir, "full_contrasts", "contrasts_batch_%i.parquet" % batch_num), filters=[("gene", "==", gene)])
        else:
            contrasts = self._score(gene_dict={gene:None}, onthefly=True)
        # Process contrasts. The resulting "pos" index (parsed from pos_seq) is the ISM
        # mutation-index position (see mutagenize); this is the coordinate space that ntstart/ntend
        # in plot_matrix/plot_logo are given in, and that dprofile's PROFILE-position columns still
        # need `ntmapper`/profmask (see plot_matrix) to be aligned with for codon-resolution models.
        contrast_per_pos = contrasts.loc[(ser:=contrasts["pos_seq"].explode()).index].assign(poslong=contrasts["ori_seq"]+ser.astype(str)+contrasts["to_seq"])
        dcounts = contrast_per_pos.loc[(ser:=contrast_per_pos["dcount"].explode()).index].assign(dcountval=ser, cond=list(self.y_raw["cds"].columns)*contrast_per_pos.shape[0]).pivot_table(index="poslong",columns="cond",values="dcountval",dropna=False).astype(np.float32)
        if not countsonly:
            dprofile = contrast_per_pos.loc[(ser:=contrast_per_pos["dprofile"].explode()).index].assign(dprofval=ser, cond=list(self.y_raw["cds"].columns)*contrast_per_pos.shape[0]).pivot_table(index="poslong",columns="cond",values="dprofval", dropna=False)
        if conds is None:
            conds = list(self.y_raw["cds"].columns)
        if ctrlcond is not None:
            ddcounts = dcounts[conds].sub(dcounts[ctrlcond], axis=0).assign(pos=[int(s[1:-1]) for s in dcounts.index]).pivot_table(index="pos", aggfunc=lambda x: np.mean(x, axis=0),dropna=False)
            if not countsonly:
                ddprofile = dprofile[conds].sub(dprofile[ctrlcond], axis=0).assign(pos=[int(s[1:-1]) for s in dprofile.index]).pivot_table(index="pos", aggfunc=lambda x: np.mean(np.stack(x), axis=0), dropna=False)
        else:
            ddcounts = dcounts[conds].sub(dcounts[conds].mean(axis=1), axis=0).assign(pos=[int(s[1:-1]) for s in dcounts.index]).pivot_table(index="pos", aggfunc=lambda x: np.mean(x, axis=0),dropna=False)
            if not countsonly:
                ddprofile = dprofile[conds].apply(lambda row: row.apply(lambda x: x - np.mean(np.stack(row.values), axis=0)), axis=1).assign(pos=[int(s[1:-1]) for s in dprofile.index]).pivot_table(index="pos", aggfunc=lambda x: np.mean(np.stack(x), axis=0), dropna=False)
        dcounts = dcounts[conds].assign(pos=[int(s[1:-1]) for s in dcounts.index]).pivot_table(index="pos", aggfunc=lambda x: np.max(np.abs(x), axis=0),dropna=False)
        if not countsonly:
            dprofile = dprofile[conds].assign(pos=[int(s[1:-1]) for s in dprofile.index]).pivot_table(index="pos", aggfunc=lambda x: np.max(np.abs(np.stack(x)), axis=0), dropna=False)
        # Record regsizes
        regsizes = [self.x_raw.loc[gene,"utr5_len"], self.x_raw.loc[gene,"cds_len"], self.x_raw.loc[gene,"utr3_len"]]
        if countsonly:
            dprofile = None; ddprofile = None
        return [dcounts, dprofile, regsizes], [ddcounts, ddprofile, regsizes]
    
    def plot_matrix(self, gene, cond, ctrl_cond=None, ntstart=0, ntend=None):
        '''
        Plot heatmap of importance scores along a gene sequence.

        Parameters
        ----------
        gene : str
            Gene identifier.
        cond : str
            Condition name to plot.
        ctrl_cond : str, optional
            Control condition for differential scoring.
        ntstart : int, default=0
            Starting nucleotide position.
        ntend : int, optional
            Ending nucleotide position. If None, uses full sequence.

        Returns
        -------
        None
            Displays the heatmap plot.
        '''
        # Load data
        sis, dsis = self.extract_gene_scores(gene, conds=[cond], ctrlcond=ctrl_cond, countsonly=False)
        sisdf = sis if ctrl_cond is None else dsis
        xcgene = self.x_raw.loc[gene]
        # Get infos
        ntend= sisdf[1].shape[0] if ntend is None else min(ntend, sisdf[1].shape[0])
        utr5len=min(self.utr_len[0], xcgene.loc["utr5_len"])
        utr5pool=int(utr5len*self.utr_pool)
        utr3len=min(self.utr_len[1], xcgene.loc["utr3_len"])
        utr3pool=int(utr3len*self.utr_pool)
        cdslen=int(xcgene.loc["cds_len"]/3)
        profmask=np.concatenate([[False]*(self.utr_pool_size[0]-utr5pool)+[True]*utr5pool,
                                [True]*(cdslen)+[False]*(self.seq_len-cdslen),
                                [True]*utr3pool+[False]*(self.utr_pool_size[1]-utr3pool)])
        # Print seq
        # ntmapper converts n, an ISM mutation-index nt position (same space as ntstart/ntend and as
        # sisdf's "pos" index, both anchored at this gene's annotated UTR5 start), into the index of
        # the corresponding column of the profmask-filtered profile below (i.e. the model's codon-
        # resolution profile position, but with invalid/padding bins already dropped by profmask).
        ntmapper = {n:int(n*self.utr_pool) if n<utr5len else utr5pool+int((n-utr5len)/3) if n<sum([utr5len,cdslen*3]) else sum([utr5pool,cdslen])+int((n-sum([utr5len,cdslen*3]))*self.utr_pool) for n in range(int(sum([utr5len,cdslen*3,utr3len])+1))}
        _ = print_seq(xcgene, sisdf[2])
        # Plot
        df1 = np.stack(sisdf[1][cond])[ntstart:ntend,profmask]
        df2 = sisdf[0][[cond]].values[ntstart:ntend]
        fig = plt.figure(figsize=(14, 11))
        ax0 = plt.subplot2grid(shape=(28, 120), loc=(0, 0), rowspan=2, colspan=90)
        ax1 = plt.subplot2grid(shape=(28, 120), loc=(2, 0), rowspan=26, colspan=90)
        ax2 = plt.subplot2grid(shape=(28, 120), loc=(2, 91), rowspan=26, colspan=1)
        ax3 = plt.subplot2grid(shape=(28, 120), loc=(2, 99), rowspan=26, colspan=3)
        ax4 = plt.subplot2grid(shape=(28, 120), loc=(2, 103), rowspan=26, colspan=1)
        # profile
        ymatgene = np.mean(self.y_mat[self.x_raw.index.get_loc(gene),self.y_raw["cds"].columns.get_loc(cond),:,:], axis=-1)
        tempdf = pd.DataFrame({"pos":np.arange(sum(profmask)),"counts":np.nan_to_num(ymatgene[profmask])}).iloc[ntmapper[ntstart]:ntmapper[ntend]]
        sns.lineplot(tempdf, x="pos", y="counts", color="k", linewidth=1, errorbar=None, ax=ax0)
        ax0.set_xticks([])
        ax0.xaxis.set_visible(False)
        ax0.margins(x=0)
        # heatmap
        df1 = df1[:,ntmapper[ntstart]:ntmapper[ntend]]
        if not ctrl_cond:
            sns.heatmap(df1, xticklabels=50, yticklabels=100, cbar=False, ax=ax1, cmap="viridis", vmin=0.0, robust=True)
        else:
            sns.heatmap(df1, xticklabels=50, yticklabels=100, cbar=False, ax=ax1, cmap="icefire", center=0.0, robust=True)
        ax1.hlines([sisdf[2][0]-ntstart, sum([sisdf[2][0],sisdf[2][1]])-ntstart], *ax1.get_xlim(), colors="w", linewidth=0.5)
        ax1.vlines([sum(profmask[:self.utr_pool_size[0]])-ntmapper[ntstart], sum(profmask[:sum([self.utr_pool_size[0], self.seq_len])])-ntmapper[ntstart]], *ax1.get_ylim(), colors="w", linewidth=0.5)
        fig.colorbar(ax1.collections[0], cax=ax2)
        if not ctrl_cond:
            sns.heatmap(df2, yticklabels=100, xticklabels=["counts"], cbar=False, ax=ax3, cmap="viridis", robust=True)
        else:
            sns.heatmap(df2, yticklabels=100, xticklabels=["counts"], cbar=False, ax=ax3, cmap="icefire", center=0.0, robust=True)
        fig.colorbar(ax3.collections[0], cax=ax4)
        return fig
    
    def plot_logo(self, gene, conds, ctrl_cond=None, ntstart=0, ntend=None, lims=[]): 
        '''
        Plot sequence logo showing nucleotide importance scores.

        Parameters
        ----------
        gene : str
            Gene identifier.
        conds : list of str
            List of condition names to plot.
        ctrl_cond : str, optional
            Control condition for differential scoring.
        ntstart : int, default=0
            Starting nucleotide position.
        ntend : int, optional
            Ending nucleotide position. If None, uses full sequence.
        lims : list, default=[]
            Y-axis limits for the plots.

        Returns
        -------
        None
            Displays the sequence logo plots.
        '''
        # Load data
        sis, dsis = self.extract_gene_scores(gene, conds=conds, ctrlcond=ctrl_cond, countsonly=True)
        sisdf = sis if ctrl_cond is None else dsis
        xcgene = self.x_raw.loc[gene]
        # Get infos
        ntend= sisdf[0].shape[0] if ntend is None else min(ntend, sisdf[0].shape[0])
        utr5len=min(self.utr_len[0], xcgene.loc["utr5_len"])
        cdslen=int(xcgene.loc["cds_len"])
        # Print seq
        seq = print_seq(xcgene, sisdf[2])
        # Make logos
        nucl = ["A","C","G","T"]
        ohe = pd.get_dummies(list(seq[ntstart:ntend]))
        for b in [s for s in nucl if s not in ohe.columns]:
            ohe[b] = False
        ohe = ohe[nucl].values
        # Plot
        limsmax = [[],[]]
        fig, ax = plt.subplots(nrows=len(conds), figsize=(10, 1.4*len(conds)))
        for n,cond in enumerate(conds):
            axtemp = ax[n] if len(conds)>1 else ax
            # Logo
            df = np.zeros(ohe.shape)
            df[ohe] = sisdf[0][[cond]].values[ntstart:ntend].flatten()
            tempsis = pd.DataFrame(df, columns=nucl, index = np.arange(ntstart,ntend))
            # Display as RNA: relabel T->U for the logo only (colorblind_safe maps U to the
            # same color as T). Scores/columns stay DNA; this is purely a plotting change.
            nucl_logo = logomaker.Logo(tempsis.rename(columns={"T": "U"}), color_scheme='colorblind_safe', ax=axtemp)
            nucl_logo.style_spines(visible=True)
            # Fix lims to min max
            limsmax[0].append(nucl_logo.ax.get_ylim()[0])
            limsmax[1].append(nucl_logo.ax.get_ylim()[1])
            nucl_logo.ax.set_ylabel(cond, labelpad=-1)
            # Shade CDS
            if n <= len(conds):
                axtemp.set_xticks([])
                axtemp.xaxis.set_visible(False)
        # Set ylims
        lims = lims if lims else [min(limsmax[0]), max(limsmax[1])]
        if len(conds)>1:
            for a in ax:
                a.set_ylim(lims)
                cdspatch = Rectangle(xy=(utr5len - 0.5, lims[0]), width=cdslen, height=lims[1]-lims[0],
                          facecolor="#eeeeee", zorder=-2)
                a.add_patch(cdspatch)
        else:
            ax.set_ylim(lims)
            cdspatch = Rectangle(xy=(utr5len - 0.5, lims[0]), width=cdslen, height=lims[1]-lims[0],
                          facecolor="#eeeeee", zorder=-2)
            ax.add_patch(cdspatch)
        fig.tight_layout()
        return fig

def add_interactive_guide():
    # Add interactive analysis guide
    add_heading("Interactive analysis guide")
    add_text("After generating this report, you can interactively explore the results using the following methods:")
            
    add_subsubheading("1. get_genes_pattern(pattern, typepat, arr, conds, region)")
    add_text("Extract genes associated with a specific discovered motif pattern. This is useful for identifying which genes contain a motif of interest.")
    add_text("Parameters:")
    add_text("  - pattern: Pattern identifier (e.g., 'pattern_0', 'pattern_1')<br/>"
             "  - typepat: Pattern type ('pos_patterns' for positive patterns, 'neg_patterns' for negative patterns)<br/>"
             "  - arr: Array type ('dcounts' for absolute scores, 'ddcounts' for differential scores)<br/>"
             "  - conds: '-'-joined condition names as they appear in the report headings<br/>"
             "  - region: Genomic region ('5UTR', 'CDS', or '3UTR')")
    add_text("Returns: DataFrame with gene names, occurrence counts, maximum attribution scores, and average motif locations.")
    add_text("Example: genes_df = interpreter.get_genes_pattern('pattern_0', 'pos_patterns', 'dcounts', 'condition1-condition2', 'CDS')")
            
    add_subsubheading("2. plot_matrix(gene, cond, ctrl_cond=None, ntstart=0, ntend=None)")
    add_text("Visualize both counts and distribution importance scores as a heatmap.")
    add_text("Parameters:")
    add_text("  - gene: Gene identifier to visualize<br/>"
             "  - cond: Condition name to plot<br/>"
             "  - ctrl_cond: Optional control condition for differential analysis<br/>"
             "  - ntstart: Starting nucleotide position (default: 0)<br/>"
             "  - ntend: Ending nucleotide position (default: full sequence)")
    add_text("Heatmap shows how the sequence in each position of the transcript (rows) affect the distribution of ribosomes on every other position (columns). "
             "Brighter colors indicate nucleotide positions with higher importance for predicting ribosome occupancy. "
             "The right panel shows importance scores for total ribosome counts predictions. "
             "Lines separate the 5'UTR, CDS, and 3'UTR regions. The line plot on the top shows the actual ribosome occupancy.")
    add_text("Example: interpreter.plot_matrix('ENSG00000123456', 'condition1', ntstart=100, ntend=500)")
            
    add_subsubheading("3. plot_logo(gene, conds, ctrl_cond=None, ntstart=0, ntend=None, lims=[])")
    add_text("Display sequence logos showing nucleotide-level importance scores for one or more conditions. "
             "The height of each nucleotide letter represents its importance score at that position.")
    add_text("Parameters:")
    add_text("  - gene: Gene identifier to visualize<br/>"
             "  - conds: List of condition names to plot<br/>"
             "  - ctrl_cond: Optional control condition for differential analysis<br/>"
             "  - ntstart: Starting nucleotide position (default: 0)<br/>"
             "  - ntend: Ending nucleotide position (default: full sequence)<br/>"
             "  - lims: Y-axis limits [min, max] (default: auto-scaled)")
    add_text("The shaded region indicates the CDS. If a control condition is provided, the logos show differential importance scores between the specified conditions and the control. ")
    add_text("Example: interpreter.plot_logo('ENSG00000123456', ['condition1', 'condition2'], ntstart=0, ntend=300)")