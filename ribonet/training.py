import logging
import os
import torch
import json
try:
    import wandb
except ImportError:  # optional, only needed with use_wandb=True
    wandb = None
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from functools import partial
from torch.utils.data import DataLoader
import torch.optim as optim
from .utils.loss_utils import total_bpnet_loss
from .models.ribodataset import RiboDataset_cod, RiboDataset_nt
from .models.codon_model import RiboNet as CodonRiboNet
from .models.nucleotide_model import RiboNet as NtRiboNet
from .utils.training_utils import train_epoch, evaluate, make_prediction
from .utils.seq_utils import reshape_training_data, count_library_reads, STOP_CODONS, pad_array, print_seq
from .utils.plot_utils import color_map
from .utils.model_utils import load_lm, number_speciesLM, autoselect_workers
from .config import Nuc_Config, CDS_Config
from .utils.eval_utils import eval_prof_corr, eval_count_pred, eval_explvar
from .utils.report_utils import *


os.environ["TOKENIZERS_PARALLELISM"] = "false" # Explicitly disable tokenizers parallelism to avoid warnings and implicit disabling

log = logging.getLogger(__name__)

class RiboTraining():
    def __init__(self, project_dir, seq_resolution, cache_dir=None, model_dir=None, lm=None, species_proxy=None):
        '''
        Initialize RiboTraining class for training ribosome profiling models.

        Parameters
        ----------
        project_dir : str
            Path to the project directory containing preprocessed data.
        seq_resolution : str
            Sequence resolution for the model. Valid options: 'codon' or 'nucleotide'.
        cache_dir : str, optional
            Directory for caching language model files.
        model_dir : str, optional
            Directory containing pre-trained language models.
        lm : str, dict, or list, optional
            Language model specification. Can be a model name, dictionary of models for different regions,
            or list of model parameters.
        species_proxy : str, optional
            Proxy species identifier required when using SpeciesLM models.

        Returns
        -------
        None

        Raises
        ------
        NotADirectoryError
            If project_dir or training directory does not exist.
        ValueError
            If SpeciesLM is used without species_proxy or vice versa.
        Exception
            If seq_resolution is not valid.
        '''
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
        os.makedirs(self.training, exist_ok=True)

        x_inputs_path = os.path.join(self.preprocess_dir, "x_inputs.parquet")
        y_outputs_path = os.path.join(self.preprocess_dir, "y_outputs.parquet")
        if not os.path.isfile(x_inputs_path):
            err_message = (f"Preprocessed input file not found: {x_inputs_path}. "
                           f"Run the 'preprocess' step first to generate the training data.")
            log.error(err_message)
            raise FileNotFoundError(err_message)
        if not os.path.isfile(y_outputs_path):
            err_message = (f"Preprocessed output file not found: {y_outputs_path}. "
                           f"Run the 'preprocess' step first to generate the training data.")
            log.error(err_message)
            raise FileNotFoundError(err_message)

        self.x_c = pd.read_parquet(x_inputs_path)
        self.y_c = pd.read_parquet(y_outputs_path)

        if self.x_c.empty or self.y_c.empty:
            err_message = ("Preprocessed training data is empty. "
                           "Re-run preprocessing or check that genes passed the filtering criteria.")
            log.error(err_message)
            raise RuntimeError(err_message)
        self.seq_resolution = seq_resolution
        self.cache_dir = cache_dir
        self.model_dir = model_dir
        self.lm = lm
        self.species_proxy = species_proxy 
        
        # proxy species is required iff SpeciesLM model is used
        if number_speciesLM(self.lm) == 0 and self.species_proxy is not None:
            log.warning("A proxy species is specified, but no SpeciesLM model is used. The proxy species will be ignored.")
        if number_speciesLM(self.lm) > 0 and self.species_proxy is None:
            err_message = "A SpeciesLM model is used, but no proxy species is specified. Please provide a proxy species."
            log.error(err_message)
            raise ValueError(err_message)

        possible_seq_res = ['codon', 'nucleotide']
        if not seq_resolution.lower() in possible_seq_res:
            err_message = f"{seq_resolution} is not a valid resolution. Valid are: {possible_seq_res}"
            log.error(err_message)
            raise Exception(err_message)
        else:
            self.seq_resolution = seq_resolution.lower()
        
        if self.lm is None and self.seq_resolution == 'codon':
            self.lm = {"UTR5": None, "UTR3": None}
        self.lm = load_lm(self.lm, cache_dir=cache_dir, model_dir=self.model_dir, species_proxy=self.species_proxy)
    

    def _lm_given(self):
        '''
        Check if a language model has been provided.

        Parameters
        ----------
        None

        Returns
        -------
        bool
            True if a language model is provided, False otherwise.
        '''
        if type(self.lm) == list:
            return True
        elif type(self.lm) == dict:
            return (self.lm["UTR5"] is not None) or (self.lm["UTR3"] is not None)
        else:
            return False
        

    def _model_summary(self, model):
        log.info("{:<50} {:<30} {:<15} {:<15}".format("Layer Name", "Shape", "Parameters", "Trainable"))
        log.info("-" * 20)
        total_params = 0
        total_trainable_params = 0
        lm_params = 0
        lm_trainable_params = 0
        lm_layers = 0
        for name, parameter in model.named_parameters():
            param = parameter.numel()
            total_params += param
            # Check if the parameter is trainable
            trainable = parameter.requires_grad
            trainable_param = param if trainable else 0
            total_trainable_params += trainable_param
            if "lm" == name[:2]:
                lm_layers += 1
                lm_params += param
                lm_trainable_params += param if trainable else 0
            else:
                log.info("{:<50} {:<30} {:<15} {:<15}".format(name, str(parameter.size()), param, trainable_param))
        log.info("{:<50} {:<30} {:<15} {:<15}".format("Selected language model(s): ", str("%i Layers" % lm_layers), lm_params, lm_trainable_params))
        log.info("-" * 20)
        log.info(f"Total Parameters: {total_params}")
        log.info(f"Trainable Parameters: {total_trainable_params}")


    def initialize_training(self,
                            epoch=50, 
                            batch=32, 
                            split=0.7, 
                            learning=0.0005, 
                            weight_decay=0.01, 
                            warmup=5, 
                            alpha=0.1, 
                            beta=None, 
                            utr_pool=0.3,
                            use_decoder=False,
                            use_annotation_emb=True,
                            embedding_dim=None,
                            embeddings=None,
                            id='',
                            ablate_utr5=False,
                            ablate_utr3=False):
        '''Initializes the necessary components for training.

        Parameters
        ----------
        epoch : int, default = 50
            Epoch number for model training.
        batch : int, default = 32
            Batch size for model training.
        split : float, default = 0.7
            Proportion for the training dataset.
        learning : float, default = 0.0005
            Learning rate for model training.
        weight_decay : float, default = 0.01
            Weight decay (L2 penalty).
        warmup : int, default = 5
            Number of warmup epochs
        alpha : float, default = 0.1
            Alpha parameter of the hybrid cost function.
        beta : float, default = None
            Beta parameter of the hybrid cost function.
        utr_pool : float, default = 0.3
            Ignored if RiboTraining was initialized with seq_resolution = 'nucleotide'. Else used to condense UTR regions
        use_decoder : bool, default = False
            Whether to use the CellStateDecoder for conditioning on experimental conditions.
        use_annotation_emb : bool, default = True
            Whether to use the UTR and CDS annotation embeddings for the nucleotide model
        embedding_dim : int or None, default = None
            Ignored if use_decoder=False in initialization of RiboTraining.
            Dimension of the embeddings used in the CellStateDecoder. Required if use_decoder=True.
        embeddings: str or None, default = None
            Ignored if use_decoder=False in initialization of RiboTraining.
            Path to csv file with counts for generating PCA embeddings. 
            If None, learned embeddings are used.
        id: str, default = ''
            A description or ID for the training run.

        Returns
        -------
        None
        '''

        log.info("Initializing training with ID %s", id if id else "<not specified>")

        self.epoch = epoch
        self.batch = batch
        self.split = split
        self.learning = learning
        self.weight_decay = weight_decay
        self.warmup = warmup
        self.alpha = alpha
        if beta is None:
            self.beta = 4.5 if self.seq_resolution == 'nucleotide' else 1.0
        else:
            self.beta = beta
        self.utr_pool = utr_pool
        self.ablate_utr5 = ablate_utr5
        self.ablate_utr3 = ablate_utr3
        self.dataloader_workers = autoselect_workers()
        self.id = id
        self.use_decoder = use_decoder
        self.embedding_dim = embedding_dim

        self.sample_file = pd.read_csv(os.path.join(self.preprocess_dir, "sample_file.txt"), sep="\t")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        log.info('Device used for training: %s', self.device)

        if self.use_decoder and self.embedding_dim is None:
            err_message = "Embedding dimension must be specified when using the CellStateDecoder."
            log.error(err_message)
            raise ValueError(err_message)
        
        if not self.use_decoder and self.embedding_dim is not None:
            log.warning("Embedding dimension is specified, but CellStateDecoder is not used. The embedding dimension will be ignored.")
            self.embedding_dim = None
        if not self.use_decoder and embeddings is not None:
            log.warning("Embeddings path is specified, but CellStateDecoder is not used. The embeddings will be ignored.")
            embeddings = None
        # reshape all data according to the requirements of the models
        log.info("Reshaping training data with respect to the chosen model...")
        x_c, y_mat, seq_args = reshape_training_data(self.x_c, self.y_c, self.seq_resolution, self.lm, self.utr_pool)
        log.info("Reshaping done.")

        args = self.__dict__.copy()
        args["embeddings"] = embeddings

        # Initialize decoder configuration if decoder is used
        if self.use_decoder:
            log.info("Using CellStateDecoder with embedding dimension %d and %s embeddings",
                     self.embedding_dim, "given" if embeddings is not None else "learnable")
            embed_df = pd.read_csv(embeddings, index_col=0).T.sort_index() if embeddings is not None else None
            args["decoder"] = {
                "n_conditions": self.sample_file["condition"].nunique(),
                "embedding_dim": self.embedding_dim,
                "pca_embeddings": embed_df,
            }

        # model configs
        if self.seq_resolution == 'codon':
            seq_len = seq_args["maxcodonlen"]
            self.seq_len = seq_len

            self.utr_pool_size = [seq_args["utr5poolsize"], seq_args["utr3poolsize"]]
            self.utr_len = [seq_args["maxutr5len"], seq_args["maxutr3len"]]

            args["left_special_tokens"] = [self.lm["UTR5"][5] if self.lm["UTR5"] is not None else None, self.lm["UTR3"][5] if self.lm["UTR3"] is not None else None]
            args["right_special_tokens"] = [self.lm["UTR5"][6] if self.lm["UTR5"] is not None else None, self.lm["UTR3"][6] if self.lm["UTR3"] is not None else None]
            args["fixed_length"] = [self.lm["UTR5"][7] if self.lm["UTR5"] is not None else False, self.lm["UTR3"][7] if self.lm["UTR3"] is not None else False]
            # Raise warnings if LM uses fixed-length model but data has more k-mer positions than the LM was trained on
            for lm_key in ["UTR5", "UTR3"]:
                if self.lm["UTR5"] is not None:
                    fl = self.lm[lm_key][7]
                    if fl:
                        tok_size = self.lm[lm_key][4]
                        special_tok = sum([self.lm[lm_key][5],self.lm[lm_key][6]])
                        actual = seq_args["maxkmer5len" if lm_key == "UTR5" else "maxkmer3len"]+ special_tok if seq_args else None
                        if actual is not None and actual != tok_size:
                            log.warning(
                                "%s uses a fixed-length model (tok_size=%d) but data has %d k-mer positions. "
                                "Embeddings will be out-of-distribution.", lm_key, tok_size, actual
                            )
            model_config = CDS_Config(seq_len=seq_args["maxcodonlen"],
                                        utr_len=[seq_args["maxutr5len"], seq_args["maxutr3len"]],
                                        utr_pool_size=self.utr_pool_size,
                                        args=args)
        elif self.seq_resolution == 'nucleotide':
            args["use_annotation_emb"] = use_annotation_emb
            args["window_size"] = self.lm[4] if self._lm_given() else None
            args["overlap"] = 100
            seq_len = seq_args["maxseqlen"]
            self.seq_len = seq_len

            self.utr_pool_size = None
        
            if self._lm_given():
                n = 1
                while seq_len > (n*args["window_size"] - n*args["overlap"] - 2):
                    n += 1
                seq_len = n*args["window_size"] - n*args["overlap"] - 2
            
            args["left_special_tokens"] = self.lm[5] if self._lm_given() else None
            args["right_special_tokens"] = self.lm[6] if self._lm_given() else None
            args["fixed_length"] = self.lm[7] if self._lm_given() else False
            # Raise warnings if LM uses fixed-length model but data has more k-mer positions than the LM was trained on
            if self._lm_given() and args["fixed_length"]:
                tok_size = self.lm[4]
                special_tok = sum([self.lm[5],self.lm[6]])
                actual = seq_args["maxkmerlen"]+ special_tok if seq_args else None
                if actual is not None and actual != tok_size:
                    log.warning(
                        "LM uses a fixed-length model (tok_size=%d) but data sequence length is %d. "
                        "Embeddings will be out-of-distribution.", tok_size, actual
                    )
            model_config = Nuc_Config(seq_len=seq_len,
                                        utr_len=[seq_args["maxutr5len"], seq_args["maxutr3len"]],
                                        args=args)

        # log all model configuration parameters
        log.info("---- Model configuration ----")
        for line in str(model_config).split('\n'):
            log.info(line)
        log.info("---- End of model configuration ----")
        # Strip pca_embeddings (DataFrame, not JSON-serializable) before saving;
        # embedding weights are persisted in ribonet.pt and reloaded at inference.
        config_dict = model_config.__dict__.copy()
        if config_dict.get("decoder") is not None:
            config_dict["decoder"] = {k: v for k, v in config_dict["decoder"].items() if k != "pca_embeddings"}
        config_file = os.path.join(self.project_dir, f"config_ribonet.json")
        trained = all(os.path.isfile(os.path.join(self.training, f)) for f in ("ribonet.pt", "model_prediction.parquet"))
        if trained and os.path.isfile(config_file):
            # Training will be skipped: the saved configuration must match the saved model weights.
            with open(config_file) as f:
                saved_config = json.load(f)
            new_config = json.loads(json.dumps(config_dict))  # same types as the saved JSON
            differing = [k for k, v in new_config.items()
                         if k in saved_config and k not in ("model_dir", "cache_dir") and saved_config[k] != v]
            if differing:
                err_message = (f"A trained model already exists in {self.training} with a different configuration "
                               f"({', '.join(differing)}). Use the same options as for training (see {config_file}), "
                               "or a new project directory to train a new model.")
                log.error(err_message)
                raise ValueError(err_message)
            saved_config.update(new_config)  # keep keys added by later steps (e.g. interpretation settings)
            config_dict = saved_config
        with open(config_file, "w") as f:
            json.dump(config_dict, f, indent=4)
        log.info("Model configuration saved to %s", config_file)

        # split data into train/val/test
        log.info("Splitting dataset into train/val/test...")

        assert self.x_c.shape[0]==self.y_c.shape[0]
        data_size = self.x_c.shape[0]
        num_train = int(args["split"] * data_size)
        num_val = (data_size - num_train) // 2

        indices = np.random.default_rng(seed=42).permutation(data_size)
        training_idx, test_idx, val_idx = np.split(indices, [num_train, num_train + num_val])

        x_train, y_train = x_c.iloc[training_idx], y_mat[training_idx,:,:,:]
        x_val, y_val = x_c.iloc[val_idx], y_mat[val_idx,:,:,:]
        x_test, y_test = x_c.iloc[test_idx], y_mat[test_idx,:,:,:]

        self.x_train, self.x_val, self.x_test = x_train, x_val, x_test
        self.y_train, self.y_val, self.y_test = y_train, y_val, y_test

        log.info(
            "Dataset split done — Training: %d sequences | Validation: %d sequences | Testing: %d sequences",
            len(x_train),
            len(x_val),
            len(x_test)
        )

        # self.run = wandb.init(project="ribonet", group="evaluation", name="glucose", id=id, config=config_dict) #TODO: make wandb name changeable

        extract_model = lambda l: l[2] if l is not None else None
        if self.seq_resolution == 'nucleotide':
            self.model = NtRiboNet(extract_model(self.lm), 
                                   model_config,
                                   n_tracks = y_mat.shape[1])
        else:
            self.model = CodonRiboNet(extract_model(self.lm["UTR5"]), 
                                      extract_model(self.lm["UTR3"]), 
                                      model_config, 
                                      n_tracks = y_mat.shape[1])
        
        self.model.to(self.device)

        # If PCA embeddings were used, save them so that interpretation can reconstruct the exact same model architecture.
        if (self.use_decoder
                and hasattr(self.model, 'cell_state_decoder')
                and hasattr(self.model.cell_state_decoder, 'pca_embeddings')):
            pca_emb_list = self.model.cell_state_decoder.pca_embeddings.cpu().tolist()
            with open(config_file, "r+") as f:
                saved_config = json.load(f)
                saved_config["decoder"]["pca_embeddings"] = pca_emb_list
                f.seek(0)
                json.dump(saved_config, f, indent=4)
                f.truncate()
            log.info("Updated config_ribonet.json with computed PCA embeddings.")

        log.info("---- Model summary ----")
        self._model_summary(self.model)
        log.info("---- End of model summary ----")


        if self.seq_resolution == 'nucleotide':
            train_dataset = RiboDataset_nt(np.stack(x_train["seq"].values),
                                        np.stack(x_train["rna"].values), 
                                        np.stack(x_train["annotation"].values),
                                        y_train)

            val_dataset = RiboDataset_nt(np.stack(x_val["seq"].values),
                                    np.stack(x_val["rna"].values), 
                                    np.stack(x_val["annotation"].values),
                                    y_val)

            test_dataset = RiboDataset_nt(np.stack(x_test["seq"].values),
                                    np.stack(x_test["rna"].values), 
                                    np.stack(x_test["annotation"].values),
                                    y_test)
        else:
            train_dataset = RiboDataset_cod(np.stack(x_train["cds"].values),
                                        np.stack(x_train["utr5"].values),
                                        np.stack(x_train["utr3"].values),
                                        np.stack(x_train["rna"].values),
                                        np.stack(x_train["posemb"].values),
                                        y_train)

            val_dataset = RiboDataset_cod(np.stack(x_val["cds"].values),
                                    np.stack(x_val["utr5"].values),
                                    np.stack(x_val["utr3"].values),
                                    np.stack(x_val["rna"].values),
                                    np.stack(x_val["posemb"].values),
                                    y_val)

            test_dataset = RiboDataset_cod(np.stack(x_test["cds"].values),
                                    np.stack(x_test["utr5"].values),
                                    np.stack(x_test["utr3"].values),
                                    np.stack(x_test["rna"].values),
                                    np.stack(x_test["posemb"].values),
                                    y_test)
        
        self.train_dataloader = DataLoader(train_dataset, batch_size=batch, shuffle=True, pin_memory=True, num_workers=self.dataloader_workers)
        self.val_dataloader = DataLoader(val_dataset, batch_size=batch, pin_memory=True, num_workers=self.dataloader_workers)
        self.test_dataloader = DataLoader(test_dataset, batch_size=batch, pin_memory=True, num_workers=self.dataloader_workers)

        # Define optimizer with a scheduler
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning, weight_decay=weight_decay) #, fused = True
        self.warmup_scheduler = optim.lr_scheduler.LinearLR(self.optimizer, start_factor = 0.001, total_iters = warmup)
        self.train_scheduler = optim.lr_scheduler.LinearLR(self.optimizer, start_factor = 1.0, end_factor = 0.01, total_iters = (epoch - warmup))
        self.scheduler = optim.lr_scheduler.SequentialLR(self.optimizer, [self.warmup_scheduler, self.train_scheduler], [warmup])

        self.tot_reads = count_library_reads(self.y_c, self.sample_file)

        # Define loss criterion
        self.loss_fn = partial(total_bpnet_loss, alpha=self.alpha, beta=self.beta,
                               ablate_utr5=self.ablate_utr5, ablate_utr3=self.ablate_utr3)
    

    def run_training(self):
        '''
        Execute the training loop for the ribosome profiling model.
        Trains the model using the initialized configuration, performs validation,
        implements early stopping, and saves predictions.

        Parameters
        ----------
        None

        Returns
        -------
        None
            Saves trained model weights and predictions to the training output directory.
        '''
        if not (os.path.isfile(os.path.join(self.training, "ribonet.pt")) and os.path.isfile(os.path.join(self.training, "model_prediction.parquet"))):
            best_loss = np.inf
            best_epoch = 0

            log.info("Starting training for %d epochs...", self.epoch)

            tot_reads = torch.tensor(self.tot_reads, dtype=torch.float32).to(self.device)
            for epoch in range(self.epoch):
                train_loss = train_epoch(self.model, self.train_dataloader, self.optimizer, self.loss_fn, tot_reads, self.utr_pool_size, self.device)
                val_loss = evaluate(self.model, self.val_dataloader, self.loss_fn, tot_reads, self.utr_pool_size, self.device)
                self.scheduler.step()

                log.info(f"Epoch {epoch+1}/{self.epoch}")
                log.info(f"Train Loss: {train_loss:.4f}")
                log.info(f"Val Loss: {val_loss:.4f}")

                # save best performance
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_epoch = epoch
                    torch.save(self.model.state_dict(), os.path.join(self.training, "ribonet.pt"))
                
                # early stopping if loss does not improve for a patience of 10 epochs
                if (epoch-best_epoch)==10:
                    log.info(f"Early stopping at epoch {epoch+1}")
                    break
            
            # output all the correlations
            log.info("Saving results")
            model_weights_path = os.path.join(self.training, "ribonet.pt")
            if not os.path.isfile(model_weights_path):
                err_message = (f"Trained model weights not found at {model_weights_path}. "
                            f"Training may have failed before any improvement was found.")
                log.error(err_message)
                raise FileNotFoundError(err_message)
            self.model.load_state_dict(torch.load(model_weights_path))
            out_tensors = make_prediction(self.model, self.test_dataloader, self.device)
            profile_pred, count_pred = [s.numpy(force=True) for s in out_tensors]
            colnames_iter = [self.y_c["cds"].columns, ["profile_pred", "count_pred"]]
            col_index = pd.MultiIndex.from_product(colnames_iter, names=["cond", "info"])
            testdf = pd.DataFrame([[x for c in range(count_pred.shape[1]) for x in [profile_pred[r,c,:], count_pred[r,c]]] for r in range(count_pred.shape[0])],
                                index = self.x_test.index, columns = col_index)
            testdf.to_parquet(os.path.join(self.training, 'model_prediction.parquet'))
            
            train_unshuffled = DataLoader(self.train_dataloader.dataset, batch_size=self.batch, shuffle=False, pin_memory=True, num_workers=self.dataloader_workers)
            out_tensors = make_prediction(self.model, train_unshuffled, self.device)
            profile_pred, count_pred = [s.numpy(force=True) for s in out_tensors]
            traindf = pd.DataFrame([[x for c in range(count_pred.shape[1]) for x in [profile_pred[r,c,:], count_pred[r,c]]] for r in range(count_pred.shape[0])],
                                index = self.x_train.index, columns = col_index)
            traindf.to_parquet(os.path.join(self.training, 'model_trainingset.parquet'))
            
            log.info("Finished training and saved predictions as parquet files in %s", self.training)
        
        else:
            log.info("Trained model weights and predictions already exist in %s. Skipping training and loading existing predictions.", self.training)

        self.evaluate_model() # TODO: wandb flag?
        self.generate_eval_report()
        log.info("Finished evaluation and saved the report in %s", self.training)


    def evaluate_model(self, use_wandb: bool = False) -> dict:
        """
        Post-training evaluation of trained model predictions.

        Parameters
        ----------
        use_wandb : bool
            Whether to activate Weights & Biases logging.

        Returns
        -------
        dict
            Evaluation metrics, with keys ``"distribution_prediction"`` and ``"count_prediction"``.
        """
        log.info("Starting model evaluation...")

        pred_path = os.path.join(self.training, "model_prediction.parquet")

        if not os.path.isfile(pred_path):
            raise FileNotFoundError(f"Prediction file not found: {pred_path}")

        model_pred = pd.read_parquet(pred_path)

        if not model_pred.index.equals(self.x_test.index):
           raise ValueError("model_prediction index does not match test set index")

        self.conditions =  list(self.y_c["cds"].columns)
        newcols = [(c, "count_true") for c in self.y_c["cds"].columns]

        model_pred[newcols] = (
            self.y_c["cds"].loc[model_pred.index].map(lambda x: [np.nansum(s) for s in x])
        )
        tot_reads_df = pd.Series([list(row[~np.isnan(row)]) for row in self.tot_reads],index=self.y_c["cds"].columns,dtype=object,)
        rpms = self.y_c["cds"].loc[model_pred.index].apply(lambda c: c.apply(lambda r: np.nansum(np.stack(r), axis=1)*1e6/tot_reads_df.loc[c.name]), axis=0)

        rpms = rpms.map(lambda x: [np.nan if s<1 else np.log(s) for s in x])
        newcols = [(c, "rpm_true") for c in self.y_c["cds"].columns] 
        model_pred[newcols] = rpms.values

        model_pred[pd.MultiIndex.from_product([self.y_c["cds"].columns, ["profile_true"]], names=["cond", "info"])] = [[[rep for rep in cond.T] for cond in self.y_test[model_pred.index.get_loc(s),:,:,:]] for s in model_pred.index]

        self.eval_results = {"count_prediction": {}, "distribution_prediction": {}}

        if use_wandb:
            run = wandb.init(project="ribonet", group="benchmarking", id="debug")
        
        self.model_pred = model_pred

        cds = [self.x_c.loc[gene, "cds"][:int(self.x_c.loc[gene, "cds_len"])] for gene in model_pred.index]
        codons = {
            g: [s[i:i+3] for i in range(0, len(s), 3)]
            for g, s in zip(model_pred.index, cds)
        }
        counts_parts = []
        for cond in self.conditions:
            maxrep = len(model_pred.xs("profile_true", level=1, axis=1)[cond].iloc[0])
            countstemp = pd.DataFrame(columns=["gene","pos","counts_pred","counts_true","Psite","Asite","Esite","cond","rep"])
            countstemp["gene"] = np.repeat(model_pred.index, self.model.seq_len*maxrep)
            countstemp["pos"] = np.tile(np.arange(self.model.seq_len), maxrep*len(model_pred.index))
            countstemp["rep"] = np.tile(np.repeat(np.arange(maxrep), self.model.seq_len), len(model_pred.index))
            countstemp["cond"] = cond
            profile_true = model_pred.xs("profile_true", level=1, axis=1)[cond].map(np.array).explode()
            if self.seq_resolution == "codon":
                profile_true = np.stack(profile_true)[:,self.utr_pool_size[0]:self.utr_pool_size[0]+self.model.seq_len].reshape(-1)
            else:
                # set utrs to nan for each sequence
                for idx, row in profile_true.items():
                    utr5len = self.x_c.loc[idx, "utr5_len"].astype(int)
                    cdslen = self.x_c.loc[idx, "cds_len"].astype(int)
                    row[:utr5len] = np.nan
                    row[utr5len+cdslen:] = np.nan
                profile_true = np.stack(profile_true).reshape(-1)
            countstemp["counts_true"] = profile_true
            abscounts = np.nansum(np.stack(model_pred.xs("profile_true", level=1, axis=1)[cond].map(np.array).explode()),axis=1).reshape(-1,1)
            profile_pred = model_pred.xs("profile_pred", level=1, axis=1)[cond].map(np.array)
            if self.seq_resolution == "codon":
                profile_pred = (np.repeat(np.stack(profile_pred),maxrep,axis=0)*abscounts)[:,self.utr_pool_size[0]:self.utr_pool_size[0]+self.model.seq_len].reshape(-1)
            else:
                for idx, row in profile_pred.items():
                    utr5len = self.x_c.loc[idx, "utr5_len"].astype(int)
                    cdslen = self.x_c.loc[idx, "cds_len"].astype(int)
                    row[:utr5len] = np.nan
                    row[utr5len+cdslen:] = np.nan
                profile_pred = (np.repeat(np.stack(profile_pred),maxrep,axis=0)*abscounts).reshape(-1)
            countstemp["counts_pred"] = profile_pred
            nas = np.isnan(countstemp["counts_true"])
            countstemp = countstemp.loc[~nas]
            if self.seq_resolution == "codon":
                countstemp["Psite"] = [codons[g][p] for g,p in zip(countstemp["gene"],countstemp["pos"])]
                countstemp["Asite"] = [codons[g][p+1] if p+1<len(codons[g]) else np.nan for g,p in zip(countstemp["gene"],countstemp["pos"])]
                countstemp["Esite"] = [codons[g][p-1] if p-1>=0 else np.nan for g,p in zip(countstemp["gene"],countstemp["pos"])]
            else:
                # For nucleotide resolution, map nucleotide positions to codons
                def get_codon_at_site(row, site_offset=0):
                    gene = row['gene']
                    pos = row['pos']
                    utr5_len = int(self.x_c.loc[gene, 'utr5_len'])
                    cds_pos = pos - utr5_len  # Convert absolute to CDS-relative
                    codon_idx = (cds_pos + site_offset) // 3
                    if 0 <= codon_idx < len(codons[gene]):
                        return codons[gene][int(codon_idx)]
                    else:
                        return np.nan
                
                countstemp["Psite"] = countstemp.apply(lambda row: get_codon_at_site(row, 0), axis=1)
                countstemp["Asite"] = countstemp.apply(lambda row: get_codon_at_site(row, 3), axis=1)
                countstemp["Esite"] = countstemp.apply(lambda row: get_codon_at_site(row, -3), axis=1)
            counts_parts.append(countstemp)
        self.counts_per_pos = pd.concat(counts_parts, axis=0, ignore_index=True)

        self.eval_results["distribution_prediction"] = eval_prof_corr(self.counts_per_pos, self.conditions, use_wandb)
        self.eval_results["count_prediction"]["correlation"] = eval_count_pred(self.x_c, model_pred, self.x_c["cds_len"], use_wandb)

        # Train RPMs needed to build the baseline model for the explained variance calculation
        train_rpms = self.y_c["cds"].loc[self.x_train.index].apply(lambda c: c.apply(lambda r: np.nansum(np.stack(r), axis=1)*1e6/tot_reads_df.loc[c.name]), axis=0)
        train_rpms = train_rpms.map(lambda x: [np.nan if s<1 else np.log(s) for s in x])
        train_col_index = pd.MultiIndex.from_product([self.y_c["cds"].columns, ["rpm_true"]], names=["cond", "info"])
        traindf = pd.DataFrame(train_rpms.values, index=self.x_train.index, columns=train_col_index)

        self.eval_results["count_prediction"]["explained_variance"] = eval_explvar(self.x_c, model_pred, self.x_c.loc[self.x_train.index], traindf, self.x_c["cds_len"], use_wandb)
        
        log.info("Model evaluation completed.")

        if use_wandb:
            run.finish()

        return self.eval_results


    def load_for_plotting(self):
        """
        Load saved predictions and configuration for interactive plotting.

        Reads only from disk (config_ribonet.json and model_prediction.parquet);
        never writes any files. Sets the attributes needed to call
        plot_prediction_per_gene() without running initialize_training() or
        evaluate_model().

        Returns
        -------
        None

        Raises
        ------
        FileNotFoundError
            If config_ribonet.json or model_prediction.parquet are missing.
        NotImplementedError
            If seq_resolution is not 'codon'.
        """
        config_path = os.path.join(self.project_dir, "config_ribonet.json")
        if not os.path.isfile(config_path):
            raise FileNotFoundError(
                f"Config file not found: {config_path}. Make sure the model has been trained."
            )
        with open(config_path) as f:
            config = json.load(f)

        self.seq_len = config["seq_len"]
        self.utr_pool_size = config.get("utr_pool_size")
        self.utr_pool = config.get("utr_pool", 0.3)
        self.utr_len = config.get("utr_len")
        self.conditions = list(self.y_c["cds"].columns)

        pred_path = os.path.join(self.training, "model_prediction.parquet")
        if not os.path.isfile(pred_path):
            raise FileNotFoundError(
                f"Prediction file not found: {pred_path}. Run training first."
            )
        model_pred = pd.read_parquet(pred_path)
        test_y = self.y_c.loc[model_pred.index]

        if self.seq_resolution == "codon":
            maxcodonlen = config["seq_len"]
            maxutr5len = config["utr_len"][0]
            maxutr3len = config["utr_len"][1]
            utr5poolsize, utr3poolsize = config["utr_pool_size"]
            padded_seq_len = maxcodonlen + utr5poolsize + utr3poolsize

            utr5pool = torch.nn.AdaptiveAvgPool1d(utr5poolsize)
            utr3pool = torch.nn.AdaptiveAvgPool1d(utr3poolsize)

            max_reps = max(s.shape[0] for s in test_y["cds"].iloc[0].values)
            n_genes = len(model_pred)
            ymat = np.full([n_genes, len(self.conditions), padded_seq_len, max_reps], np.nan)

            for n, cond in enumerate(self.conditions):
                nrep = test_y["cds"].iloc[0, n].shape[0]
                conddf = test_y.xs(cond, level=1, axis=1)
                cdstemp = np.stack([
                    [pad_array(row.loc["cds"][s].astype(np.float32), maxcodonlen * 3)
                     for s in range(row.iloc[0].shape[0])]
                    for _, row in conddf.iterrows()
                ])
                cdstemp = np.add.reduceat(cdstemp, np.arange(0, cdstemp.shape[2], 3), axis=2)
                utr5temp = utr5pool(
                    torch.tensor(np.stack([
                        [row.loc["utr5"][s][-maxutr5len:].astype(np.float32) for s in range(nrep)]
                        for _, row in conddf.iterrows()
                    ]), dtype=torch.float32)
                ).detach().numpy() * (maxutr5len / utr5poolsize)
                utr3temp = utr3pool(
                    torch.tensor(np.stack([
                        [row.loc["utr3"][s][:maxutr3len].astype(np.float32) for s in range(nrep)]
                        for _, row in conddf.iterrows()
                    ]), dtype=torch.float32)
                ).detach().numpy() * (maxutr3len / utr3poolsize)
                ymat[:, n, :, :nrep] = np.transpose(
                    np.concatenate([utr5temp, cdstemp, utr3temp], axis=2), (0, 2, 1)
                )
        else:
            raise NotImplementedError(
                "load_for_plotting() currently only supports seq_resolution='codon'."
            )

        model_pred[pd.MultiIndex.from_product(
            [self.conditions, ["profile_true"]], names=["cond", "info"]
        )] = [
            [[rep for rep in cond.T] for cond in ymat[i, :, :, :]]
            for i in range(len(model_pred))
        ]

        self.model_pred = model_pred
        log.info(
            "Loaded predictions for %d test genes across %d conditions.",
            len(model_pred), len(self.conditions)
        )


    def plot_count_correlation(self):
        '''
        Generate scatter plots showing correlation between predicted and true ribosome counts.
        Creates a grid of subplots, one for each experimental condition, displaying
        predicted vs true counts with Pearson correlation coefficients.

        Parameters
        ----------
        None

        Returns
        -------
        matplotlib.figure.Figure
            Figure containing count correlation plots for all conditions.
        '''
        testdf = self.model_pred
        conditions = testdf.columns.unique("cond")
        n_cond = len(conditions)
        n_cols = math.ceil(math.sqrt(n_cond))
        n_rows = math.ceil(n_cond / n_cols)

        fig, axs = plt.subplots(
            n_rows,
            n_cols,
            figsize=(3*n_cols, 3*n_rows),
            squeeze=False,
            sharex=True,
            sharey=True,
        )

        for n, cond in enumerate(conditions):
            r, c = divmod(n, n_cols)
            ax = axs[r, c]

            tempdf = pd.melt(
                pd.concat([testdf[cond]["count_pred"],
                        pd.DataFrame(testdf[cond]["rpm_true"].to_list(),
                            index=testdf.index,
                        ),], axis=1,),
                id_vars="count_pred",
                value_name="rpm_true",
                ignore_index=False,
            ).dropna()

            ax.scatter(
                tempdf["count_pred"],
                tempdf["rpm_true"],
                alpha=0.1,
                color="#4b7b6a",
                s=10,
            )
            ax.axline((0, 0), slope=1, color="grey", linestyle="--", linewidth=0.8, zorder=0)
            mask = ~(np.isnan(tempdf["rpm_true"].values) | np.isnan(tempdf["count_pred"].values))
            r_pearson = np.corrcoef(tempdf["rpm_true"].values[mask], tempdf["count_pred"].values[mask])[0, 1] if mask.sum() >= 2 else float("nan")
            ax.set_title(f"{cond} - R={r_pearson:.3f}")

        for k in range(n_cond, n_rows * n_cols):
            r, c = divmod(k, n_cols)
            fig.delaxes(axs[r, c])

        for r in range(n_rows):
            axs[r, 0].set_ylabel("Predicted log(RPM)")
        for c in range(n_cols):
            axs[n_rows - 1, c].set_xlabel("Observed log(RPM)")

        fig.tight_layout()
        return fig

    def plot_explained_variance(self):
        """
        Bar plot comparing explained variance of RiboNet count prediction
        vs RNA abundance as a baseline.

        Returns
        -------
        matplotlib.figure.Figure
        """
        explvar = self.eval_results["count_prediction"].get("explained_variance")
        if explvar is None:
            raise ValueError("explained_variance not found in eval_results. Run evaluate_model() first.")
        
        ribonet_ev, abundance_ev = explvar["explained_variance_ribonet"], explvar["explained_variance_rna_baseline"]

        fig, ax = plt.subplots(figsize=(3, 5))
        bars = ax.bar(
            ["RiboNet", "RNA Abundance"],
            [ribonet_ev, abundance_ev],
            color=["#4b7b6a", "#CDCDCD"]
        )

        # Annotate each bar with its value
        for bar, val in zip(bars, [ribonet_ev, abundance_ev]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.1f}%",
                ha="center",
                va="bottom",
                fontsize=10
            )

        ax.set_ylim(0, 100)
        ax.set_ylabel("Explained Variance (%)")
        fig.tight_layout()
        return fig

    def plot_distribution_correlation(self, max_cols=6):
        '''
        Generate scatter plots showing correlation between predicted and true distribution signals.
        Creates a grid of subplots showing nucleotide-level predictions vs true values
        for each experimental condition.

        Parameters
        ----------
        max_cols : int, default=6
            Maximum number of columns in the subplot grid.

        Returns
        -------
        matplotlib.figure.Figure
            Figure containing distribution correlation plots for all conditions.
        '''
        conditions = self.model_pred.columns.unique("cond")
        n_cond = len(conditions)

        n_cols = min(max_cols, math.ceil(math.sqrt(n_cond)))
        n_rows = math.ceil(n_cond / n_cols)

        fig, axs = plt.subplots(
            n_rows,
            n_cols,
            figsize=(3*n_cols, 3*n_rows),
            squeeze=False,
            sharex=True,
            sharey=True,
        )
        for n,cond in enumerate(conditions):
            trues = np.log2(self.counts_per_pos.loc[self.counts_per_pos.cond==cond,"counts_true"]*10000 + 2**10) - 10
            preds = np.log2(self.counts_per_pos.loc[self.counts_per_pos.cond==cond,"counts_pred"]*10000 + 2**10) - 10

            r, c = divmod(n, n_cols)
            norm_cor = str("%s - R=%1.3f" % (cond, stats.spearmanr(trues, preds, nan_policy="omit")[0]))

            pv, tv = preds.values, trues.values
            counts_h, xedges, yedges = np.histogram2d(pv, tv, bins=200)
            xi_bin = np.clip(np.digitize(pv, xedges) - 1, 0, counts_h.shape[0] - 1)
            yi_bin = np.clip(np.digitize(tv, yedges) - 1, 0, counts_h.shape[1] - 1)
            density = counts_h[xi_bin, yi_bin]
            density = np.clip(density, 0, len(pv)/2000)  # Cap density for better visualization
            sort_idx = density.argsort()
            axs[r,c].scatter(x=pv[sort_idx], y=tv[sort_idx], s=5, edgecolors='none',
                             c=density[sort_idx], cmap="cividis", rasterized=True)
            axs[r,c].axline((0, 0), slope=1, color="grey", linestyle="--", linewidth=0.8, zorder=0)
            axs[r,c].set_title(norm_cor)

        for k in range(n_cond, n_rows * n_cols):
            r, c = divmod(k, n_cols)
            fig.delaxes(axs[r, c])

        ax_max = axs.flat[0].get_xlim()[1]
        max_raw = (2 ** (ax_max + 10) - 2**10) / 10000
        max_pow = int(math.floor(math.log10(max(max_raw, 1))))
        raw_ticks = [0] + [10**i for i in range(max_pow + 1)]
        tick_pos = [np.log2(v * 10000 + 2**10) - 10 for v in raw_ticks]
        for r in range(n_rows):
            axs[r, 0].set_ylabel("Observed")
        for c in range(n_cols):
            axs[n_rows - 1, c].set_xlabel("Predicted")
        axs.flat[0].set_xticks(tick_pos)
        axs.flat[0].set_xticklabels([str(v) for v in raw_ticks])
        axs.flat[0].set_yticks(tick_pos)
        axs.flat[0].set_yticklabels([str(v) for v in raw_ticks])

        fig.tight_layout()
        return fig


    def plot_counts_bar(self):
        """
        Create a grouped bar plot comparing prediction vs replicate correlations
        for profiling and count metrics using already computed eval_results.
        
        Returns:
            Matplotlib figure
        """
        count_metrics = self.eval_results["count_prediction"].get("correlation", {})
        count_conditions = [c for c in count_metrics]
        count_pred = [count_metrics[c]["prediction"] for c in count_conditions]
        count_rna = [count_metrics[c]["rna_abundance"] for c in count_conditions]

        fig, ax = plt.subplots(1, 1, figsize=(max(0.8*len(count_conditions), 4), 5))
        
        # Count prediction bar plot
        x = np.arange(len(count_conditions))
        width = 0.35
        ax.bar(x - width/2, count_pred, width, label="Predicted vs Observed", color="#4b7b6a")
        ax.bar(x + width/2, count_rna, width, label="RNA-seq baseline", color="#CDCDCD")
        ax.set_xticks(x)
        ax.set_xticklabels(count_conditions, rotation=45, ha="right")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Pearson Correlation")
        ax.legend()

        fig.tight_layout()
        return fig
    
    def plot_dist_bar(self):
        """
        Create a grouped bar plot comparing prediction vs replicate correlations
        for profiling and count metrics using already computed eval_results.
        
        Returns:
            Matplotlib figure
        """

        prof_metrics = self.eval_results.get("distribution_prediction", {})
        prof_conditions = list(prof_metrics.keys())
        prof_pred = [prof_metrics[c]["prediction_correlation"] for c in prof_conditions]
        prof_rep = [prof_metrics[c]["replicate_correlation"] for c in prof_conditions]

        fig, axs = plt.subplots(1, 1, figsize=(max(0.8*len(prof_conditions), 4), 5))

        # Profiling prediction bar plot
        x = np.arange(len(prof_conditions))
        width = 0.35
        axs.bar(x - width/2, prof_pred, width, label="Predicted vs Observed", color="#4b7b6a")
        axs.bar(x + width/2, prof_rep, width, label="Replicate correlations' average", color="#CDCDCD")
        axs.set_xticks(x)
        axs.set_xticklabels(prof_conditions, rotation=45, ha="right")
        axs.set_ylim(0, 1)
        axs.set_ylabel("Spearman Correlation")
        axs.legend()

        fig.tight_layout()
        return fig
    
    def plot_codon_density(self):
        """
        Plot heatmaps of log2-normalized codon pause scores for true and predicted counts.

        Pause scores are computed as counts divided by the mean counts per
        (gene, condition, replicate), then averaged by A-site codon and condition.
        Stop codons are excluded.

        Returns
        -------
        matplotlib.figure.Figure
            Figure containing heatmaps for true and predicted pause scores.
        """
        counts_per_pos = self.counts_per_pos.copy()

        # Compute mean per (gene, cond, rep) and join back — avoids dtype mismatch on merge keys
        genemean_true = counts_per_pos.groupby(["gene", "cond", "rep"])["counts_true"].mean().rename("mean_true")
        genemean_pred = counts_per_pos.groupby(["gene", "cond", "rep"])["counts_pred"].mean().rename("mean_pred")

        counts_per_pos = counts_per_pos.join(genemean_true, on=["gene", "cond", "rep"])
        counts_per_pos = counts_per_pos.join(genemean_pred, on=["gene", "cond", "rep"])

        # Validate join worked before proceeding
        if "mean_true" not in counts_per_pos.columns or counts_per_pos["mean_true"].isna().all():
            raise RuntimeError(
                "Join on ['gene', 'cond', 'rep'] produced no matches. "
                f"counts_per_pos dtypes: {counts_per_pos[['gene','cond','rep']].dtypes.to_dict()}"
            )

        counts_per_pos["pause_true"] = counts_per_pos["counts_true"] / counts_per_pos["mean_true"]
        counts_per_pos["pause_pred"] = counts_per_pos["counts_pred"] / counts_per_pos["mean_pred"]

        A_pause = {
            "True": counts_per_pos.groupby(["Asite", "cond"])["pause_true"].mean().unstack(level=1).T,
            "Pred": counts_per_pos.groupby(["Asite", "cond"])["pause_pred"].mean().unstack(level=1).T,
        }

        stopcod = [c for c in STOP_CODONS if c in A_pause["True"].columns]
        A_pause["True"].drop(stopcod, axis=1, inplace=True)
        A_pause["Pred"].drop(stopcod, axis=1, inplace=True)

        # Sort conditions by read length, then by condition name
        def _cond_sort_key(cond):
            parts = str(cond).rsplit("_", 1)
            readlen = parts[1]
            return (readlen, str(cond))

        sorted_idx = sorted(A_pause["True"].index, key=_cond_sort_key)
        A_pause["True"] = A_pause["True"].loc[sorted_idx]
        A_pause["Pred"] = A_pause["Pred"].loc[sorted_idx]

        fig, axs = plt.subplots(2, 1, figsize=(15, max(0.6*A_pause["True"].shape[0], 4)), sharex=True)
        v = 2
        sns.heatmap(np.log2(A_pause["True"]), cmap=color_map, center=0.0, vmin=-v, vmax=v, ax=axs[0])
        axs[0].set(title="Observed")
        sns.heatmap(np.log2(A_pause["Pred"]), cmap=color_map, center=0.0, vmin=-v, vmax=v, ax=axs[1])
        axs[1].set(title="Predicted")

        fig.tight_layout()
        return fig

    def plot_prediction_per_gene(self, gene, seq_range=None, y_max="fixed", print_sequence=True):
        """
        Plot predicted vs true ribosome profiles for a single gene across conditions.

        Parameters
        ----------
        gene : str
            Gene identifier
        seq_range : list[int] | None
            Optional [start, end] sequence window. If None, full transcript is used.
        y_max : str or float, default="fixed"
            Y-axis upper limit. ``"fixed"`` uses the maximum across all subplots and
            applies it uniformly. ``"flexible"`` lets each subplot auto-scale
            independently. A float sets an explicit shared upper limit.
        print_sequence : bool, default=True
            Whether to print the annotated gene sequence to stdout.

        Returns
        -------
            Matplotlib figure
        """
        testdf = self.model_pred
        ngene = testdf.index.get_loc(gene)
        genename = testdf.index[ngene]

        if print_sequence:
            regsizes = [self.x_c.loc[genename, "utr5_len"], self.x_c.loc[genename, "cds_len"], self.x_c.loc[genename, "utr3_len"]]
            print_seq(self.x_c.loc[genename], regsizes)

        if self.seq_resolution == "codon":
            # Positions are expressed in true nucleotide coordinates (relative to this
            # gene's transcript start).
            utr5_nt = int(self.x_c.loc[genename, "utr5_len"])
            cds_nt = int(self.x_c.loc[genename, "cds_len"])
            utr3_nt = int(self.x_c.loc[genename, "utr3_len"])
            n_codons = cds_nt // 3
            maxutr5_nt = self.utr_len[0]
            total_len = self.seq_len + sum(self.utr_pool_size)  # full padded length per replicate

            utr5_idx = np.arange(self.utr_pool_size[0])
            cds_idx = np.arange(self.utr_pool_size[0], self.utr_pool_size[0] + self.seq_len)
            utr3_idx = np.arange(self.utr_pool_size[0] + self.seq_len, total_len)

            utr5_pos = (utr5_idx / self.utr_pool).astype(int) - (maxutr5_nt - utr5_nt)
            cds_pos = utr5_nt + (cds_idx - self.utr_pool_size[0]) * 3
            utr3_pos = utr5_nt + cds_nt + (utr3_idx - self.utr_pool_size[0] - self.seq_len) / self.utr_pool

            # Mask out padding positions that don't correspond to real sequence for this gene
            utr5_pos = np.where(utr5_pos >= 0, utr5_pos, np.nan)
            cds_pos = np.where((cds_idx - self.utr_pool_size[0]) < n_codons, cds_pos, np.nan)
            utr3_pos = np.where(utr3_pos < (utr5_nt + cds_nt + utr3_nt), utr3_pos, np.nan)

            posarray = np.concatenate([utr5_pos, cds_pos, utr3_pos])
            cdsrange = [utr5_nt, utr5_nt + cds_nt]
        else:
            seqlen = int(self.x_c.loc[genename, "cds_len"] + self.x_c.loc[genename, "utr5_len"] + self.x_c.loc[genename, "utr3_len"])
            total_len = self.seq_len
            gap = np.full(total_len - seqlen, np.nan)
            posarray = np.concatenate([np.arange(seqlen), gap])
            cdsrange = [self.x_c.loc[genename, "utr5_len"], self.x_c.loc[genename, "utr5_len"] + self.x_c.loc[genename, "cds_len"]]

        # Clip the CDS shading to the plotted window so it never extends past the visible data
        if seq_range is not None:
            plot_cdsrange = [max(cdsrange[0], seq_range[0]), min(cdsrange[1], seq_range[1])]
        else:
            plot_cdsrange = cdsrange

        # Scale figure width down for narrow windows (e.g. small genes or a tight seq_range).
        full_width = 15
        min_width = 3
        width_ref_span = 100
        valid_pos = posarray[~np.isnan(posarray)]
        if seq_range is not None:
            n_points = np.sum((valid_pos >= seq_range[0]) & (valid_pos < seq_range[1]))
        else:
            n_points = valid_pos.size
        fig_width = full_width if n_points >= width_ref_span else max(min_width, full_width * n_points / width_ref_span)

        n_cond = len(self.conditions)
        fig, axs = plt.subplots(n_cond, 1, figsize=(fig_width, 1.6 * n_cond), sharex=True, sharey=(y_max != "flexible"))
        if n_cond == 1:
            axs = axs[np.newaxis]
        fig.suptitle(genename)

        for n, cond in enumerate(self.conditions):
            tempdf = pd.melt(
                pd.concat([
                    pd.DataFrame(
                        testdf[cond][["profile_pred"]].iloc[ngene].values.tolist(),
                        index=["profile_pred"]
                    ).T,
                    pd.DataFrame(testdf[cond][["profile_true"]].iloc[ngene].to_list()[0]).T
                ], axis=1),
                id_vars="profile_pred",
                value_name="profile_true",
                ignore_index=True
            )

            n_reps = tempdf["variable"].nunique()

            # Validate length before remapping
            expected_len = total_len * n_reps
            if len(tempdf) != expected_len:
                log.warning(
                    "Gene %s cond %s: tempdf length %d != expected %d (total_len=%d, n_reps=%d). Skipping.",
                    gene, cond, len(tempdf), expected_len, total_len, n_reps
                )
                continue

            tempdf["pos"] = np.tile(posarray, n_reps)

            # Normalize per replicate
            totreads = tempdf[["variable", "profile_true"]].groupby("variable").sum()
            tempdf["profile_true"] = tempdf.apply(
                lambda x: x["profile_true"] / totreads.loc[x["variable"]].values[0], axis=1
            )

            if seq_range is not None:
                tempdf = tempdf.loc[
                    (tempdf["pos"] >= seq_range[0]) & (tempdf["pos"] < seq_range[1])
                ]

            tempdf = tempdf.dropna(subset=["pos"])

            ax = axs[n]
            if plot_cdsrange[0] < plot_cdsrange[1]:
                ax.axvspan(plot_cdsrange[0], plot_cdsrange[1], facecolor="#eeeeee")

            sns.lineplot(tempdf, x="pos", y="profile_true", color="#91baac", errorbar=None, ax=ax)
            sns.scatterplot(tempdf, x="pos", y="profile_true", color="#91baac", ax=ax, marker=".", s=40, alpha=0.7)
            sns.lineplot(tempdf.loc[tempdf["variable"] == 0], x="pos", y="profile_pred", color="#3d6758", ax=ax)

            ax.set(title=cond, ylabel="Reads")
            ax.margins(x=0)

        axs[n_cond - 1].set_xlabel("Sequence position")

        if y_max != "flexible":
            global_ymax = max(ax.get_ylim()[1] for ax in axs.flat) if y_max == "fixed" else float(y_max)
            for ax in axs.flat:
                ax.set_ylim(bottom=0, top=global_ymax)

        fig.tight_layout(rect=[0, 0, 1, 0.98])
        return fig

    def generate_eval_report(self, filename: str = 'training.pdf', use_wandb: bool = False):
        """
        Generate a PDF report for model evaluation metrics and correlations,
        including count prediction, profiling metrics, and bar plots comparing
        prediction vs replicate/RNA abundance per condition.

        Args:
            filename: Name of output PDF report file
            use_wandb: Whether to log metrics to wandb (optional)
        """
        # Run evaluation if not already done
        if not hasattr(self, "eval_results"):
            self.evaluate_model(use_wandb=use_wandb)

        # Create directory for plots
        plots_path = os.path.join(self.training, "plots")
        if not os.path.exists(plots_path):
            os.makedirs(plots_path, exist_ok=True)
        
        log.info("Generating PDF report with evaluation results...")

        # Start PDF report
        create_report(os.path.join(self.training, filename), title="RiboTraining Report")
        add_text("This report presents the evaluation of RiboNet model predictions after training. "
                 "The model predicts both the total ribosome footprint counts per transcript and the distribution of footprints along the transcript. " \
                 "Evaluation metrics include:")
        add_text("1. Counts prediction: percentage of explained variance across genes and conditions, and Pearson correlation between predicted and observed counts per condition.")
        add_text("2. Distribution prediction: Spearman correlation between predicted and observed footprint distributions per condition.")
        add_text("3. Normalized codon occupancy of predicted and observed distributions.")
        add_page_break()
        add_toc()

        # Count prediction metrics
        add_heading("Evaluation of Counts Prediction")
        add_text("To assess the accuracy of our new model in predicting RPF counts per transcript, we compare its predictions not only against the ground truth but also against a baseline derived from RNA abundance. " \
                "Since RNA-seq signal is an input to the model, it serves as a useful lower bound for expected performance.")
        
        add_subheading("Explained Variance (%)")
        add_text("The explained variance metric quantifies how much of the variance in observed RPF counts across genes and conditions can be explained by the model's predictions, compared to a baseline model that uses RNA abundance alone.")
        ev = self.eval_results["count_prediction"]["explained_variance"]
        add_table(pd.DataFrame({
            "Model": ["RiboNet", "RNA Baseline"],
            "Explained Variance (%)": [f"{ev['explained_variance_ribonet']:.1f}", f"{ev['explained_variance_rna_baseline']:.1f}"]
        }))

        count_fig = self.plot_explained_variance()
        _, h = count_fig.get_size_inches()
        count_fig.savefig(os.path.join(plots_path, "explained_variance.pdf"))
        add_plot(count_fig, height=h/1.8)
        plt.close(count_fig)
        
        add_subheading("Counts Correlation")
        add_text("The Pearson correlation between predicted and observed RPF counts per transcript is computed for each condition to evaluate the model's performance. We also compute the Pearson correlation between raw log RNA abundance and observed RPF counts as a baseline.")

        countstab = pd.DataFrame({k: v for k, v in self.eval_results["count_prediction"]["correlation"].items() if isinstance(v, dict)}).T
        countstab.columns = ["Pred vs Obs", "Obs vs RNA"]
        countstab.to_csv(os.path.join(plots_path, "count_correlation.tsv"), sep="\t")
        add_table(countstab, include_index=True)

        barplot = self.plot_counts_bar()
        _, h = barplot.get_size_inches()
        barplot.savefig(os.path.join(plots_path, "count_correlation_bar.pdf"))
        add_plot(barplot, height=h/1.8)

        count_fig = self.plot_count_correlation()
        count_fig.savefig(os.path.join(plots_path, "count_correlation.pdf"))
        add_plot(count_fig)
        plt.close(count_fig)

        add_page_break()

        # Distribution metrics
        add_heading("Evaluation of Distribution Prediction")

        add_subheading("Distribution Correlation")
        add_text("The accuracy of predicted ribosome footprint distributions along transcripts is evaluated by computing the Spearman correlation between the predicted and observed footprint counts per position for each condition. To contextualize these correlations, we also compute the mean Spearman correlation between biological replicates for each condition, which serves as an upper bound for expected performance given experimental noise.")
        
        disttab = pd.DataFrame.from_dict(self.eval_results["distribution_prediction"]).T
        disttab.columns = ["Pred vs Obs", "Replicates"]
        disttab.to_csv(os.path.join(plots_path, "distribution_correlation.tsv"), sep="\t")
        add_table(disttab, include_index=True)

        barplot = self.plot_dist_bar()
        _, h = barplot.get_size_inches()
        barplot.savefig(os.path.join(plots_path, "distribution_correlation_bar.pdf"))
        add_plot(barplot, height=h/1.8)

        prof_fig = self.plot_distribution_correlation()
        prof_fig.savefig(os.path.join(plots_path, "distribution_correlation.pdf"))
        add_plot(prof_fig)
        plt.close(prof_fig)

        add_subheading("Codon Occupancy")
        add_text("The ribosome occupancies on each of the 61 amino-acid-encoding codons is computed as the ratio between the ribosome density within each A-site codon and the overall density along each gene CDS, and taking the average across all genes. This normalized occupancy reflects the relative decoding time of translation elongation at each codon")
        prof_fig = self.plot_codon_density()
        prof_fig.savefig(os.path.join(plots_path, "codon_occupancy.pdf"))
        add_plot(prof_fig)
        plt.close(prof_fig)

        add_page_break()

        add_subheading("Distribution Prediction per gene")
        add_text("To further illustrate the model's performance in predicting ribosome footprint distributions, we show examples of predicted vs observed profiles for a selection of highly expressed genes across conditions. The dark line shows the model prediction, the light points show the individual replicates of observed footprints, the light line is the average across replicates.")
        
        total_expr = sum(self.model_pred[(cond, "profile_true")].apply(np.nansum) for cond in self.conditions)
        most_expressed_genes = total_expr.sort_values(ascending=False).head(5).index.tolist()
        for gene in most_expressed_genes:
            prof_fig = self.plot_prediction_per_gene(gene, print_sequence=False)
            add_plot(prof_fig)
            plt.close(prof_fig)

        finalize_report()
