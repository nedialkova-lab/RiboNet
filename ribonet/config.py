from .utils.model_utils import extract_lm_name

class Nuc_Config(object):

    def __init__(self,
                 seq_len: int = 1000, 
                 utr_len: list = [1000,300],
                 args=None):
        super().__init__()

        self.seq_len = seq_len           # sequence length of input vector
        self.utr_len = utr_len           # sequence length of UTR5 and UTR3
        self.emb_lowered = 128           # dimension of LM embeddings after linear layer
        self.n_encoder_layers = 1        # Number of encoder layers (MHA + MLP)
        self.num_heads = 8               # number of attention heads
        self.ff_dim = 128                # hidden layer size in the mlp
        self.dropout_rate = 0.1          # dropout rate
        self.final_kernel_size = 41      # final kernel size of profile output

        if args is not None:
            self.epoch = args.get("epoch", None)
            self.batch = args.get("batch", None)
            self.split = args.get("split", None)
            self.learning = args.get("learning", None)
            self.weight_decay = args.get("weight_decay", None)
            self.warmup = args.get("warmup", None)
            self.alpha = args.get("alpha", None)
            self.beta = args.get("beta", None)
            self.seq_resolution = args.get("seq_resolution", None)
            self.lm = extract_lm_name(args.get("lm", None))
            self.model_dir = args.get("model_dir", None)
            self.cache_dir = args.get("cache_dir", None)
            self.species_proxy = args.get("species_proxy", None)
            self.window_size = args.get("window_size", None)
            self.overlap = args.get("overlap", None)
            self.id = args.get("id", None)
            self.decoder = args.get("decoder", None)
            self.left_special_tokens = args.get("left_special_tokens", None)
            self.right_special_tokens = args.get("right_special_tokens", None)
            self.fixed_length = args.get("fixed_length", None)
            self.embedding_dim = args.get("embedding_dim", None)
            self.embeddings = args.get("embeddings", None)
            self.use_annotation_emb = args.get("use_annotation_emb", True)

        
    def __repr__(self):
        return '\n'.join(['%s: %s' % (key, str(value)) for key, value in self.__dict__.items()])

class CDS_Config(object):

    def __init__(self,
                 seq_len: int = 1000, 
                 utr_len: list = [1000,300],
                 utr_pool_size: list = [300,90], 
                 args=None):
        super().__init__()

        self.seq_len = seq_len                  # sequence length of input vector
        self.utr_len = utr_len                  # sequence length of UTR5 and UTR3
        self.utr_pool_size = utr_pool_size      # dimension of UTR5 and UTR3 after average pooling
        self.vocab_size = 65                    # vocab length of tokens: 64 codons + pad
        self.n_filters = [72, 80, 88, 96, 128]  # number of filters in each conv layer
        self.kernel_size = 5                    # kernel size of CNNs
        self.n_encoder_layers = 1               # Number of encoder layers (MHA + MLP)
        self.num_heads = 8                      # number of attention heads
        self.ff_dim = 128                       # hidden layer size in the mlp
        self.dropout_rate = 0.1                 # dropout rate
        self.final_kernel_size = 15             # final kernel size of profile output

        if args is not None:
            self.epoch = args.get("epoch", None)
            self.batch = args.get("batch", None)
            self.split = args.get("split", None)
            self.learning = args.get("learning", None)
            self.weight_decay = args.get("weight_decay", None)
            self.warmup = args.get("warmup", None)
            self.alpha = args.get("alpha", None)
            self.beta = args.get("beta", None)
            self.seq_resolution = args.get("seq_resolution", None)
            self.lm = extract_lm_name(args.get("lm", None))
            self.model_dir = args.get("model_dir", None)
            self.cache_dir = args.get("cache_dir", None)
            self.species_proxy = args.get("species_proxy", None)
            self.id = args.get("id", None)
            self.utr_pool = args.get("utr_pool", None)
            self.decoder = args.get("decoder", None)
            self.left_special_tokens = args.get("left_special_tokens", None)
            self.right_special_tokens = args.get("right_special_tokens", None)
            self.fixed_length = args.get("fixed_length", None)
            self.embedding_dim = args.get("embedding_dim", None)
            self.embeddings = args.get("embeddings", None)
            self.ablate_utr5 = args.get("ablate_utr5", False)
            self.ablate_utr3 = args.get("ablate_utr3", False)

    def __repr__(self):
        return '\n'.join(['%s: %s' % (key, str(value)) for key, value in self.__dict__.items()])
