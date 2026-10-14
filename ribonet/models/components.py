import numpy as np
import torch
from torch import nn
from sklearn.decomposition import PCA

class CellStateDecoder(nn.Module):
    def __init__(self, 
                n_tracks,
                n_conditions,
                embedding_dim,
                dropout_rate,
                kernel_shapes,
                count_fc_shape,
                pca_embeddings):
        super().__init__()
            
        self.kernel_shapes = kernel_shapes
        self.count_fc_shape = count_fc_shape
        self.embedding_dim = embedding_dim
        self.readlength_embedding_dim = 1
        self.n_tracks = n_tracks
        self.n_conditions = n_conditions
        self.n_readlengths = n_tracks // n_conditions

        if pca_embeddings is not None:
            if isinstance(pca_embeddings, (list, np.ndarray)):
                # Pre-computed embeddings loaded from config JSON — register directly
                t = torch.tensor(pca_embeddings, dtype=torch.float16)
                self.register_buffer('pca_embeddings', t)
            else:
                # Raw count DataFrame — run PCA to derive embeddings.
                self.generate_pca_embeddings(pca_embeddings)
        else:
            self.condition_embedding = nn.Embedding(num_embeddings=n_conditions, embedding_dim=embedding_dim - self.readlength_embedding_dim)
        self.readlength_embedding = nn.Embedding(num_embeddings=self.n_readlengths, embedding_dim=self.readlength_embedding_dim)
        
        self.dropout_rate = dropout_rate

        self.cell_state_to_conv = nn.Sequential(
            nn.Linear(embedding_dim, 32),
            nn.GELU(),
            nn.Linear(32, 64), # TODO: dropout missing???
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
        )
        #heads for kernel weights; the prod is the total number of params for the tensor
        self.kernel_heads = nn.ModuleList([
            nn.Linear(128, torch.prod(torch.tensor(kernel_shape)))
            for kernel_shape in kernel_shapes
        ])
        
        # heads for the count prediction
        self.count_fc_head = nn.Linear(128, torch.prod(torch.tensor(count_fc_shape)))
        self.count_bias_head = nn.Linear(128, count_fc_shape[0])

        self.bias_heads = nn.ModuleList([nn.Linear(128, kernel_shape[0]) for kernel_shape in kernel_shapes])

    def generate_pca_embeddings(self, pca_embeddings):
        # use the counts to generate PCA embeddings of dimension self.embedding_dim - 1
        # the last dimension is used for encoding the readlengths
        n_components = min(self.embedding_dim - self.readlength_embedding_dim, *pca_embeddings.shape)
        pca = PCA(n_components=n_components)
        embeddings = pca.fit_transform(pca_embeddings)

        # z-score normalization
        embeddings = (embeddings - embeddings.mean(axis=1, keepdims=True)) / embeddings.std(axis=1, keepdims=True)

        # pad missing dimensions with zero if requested embedding dimension is larger than
        # number of PCs 
        missing_dims = (self.embedding_dim - 1) - n_components
        if missing_dims > 0:
            embeddings = np.pad(embeddings, pad_width=((0, 0), (0, missing_dims)), mode='constant', constant_values=0)

        embeddings = torch.tensor(embeddings, dtype=torch.float16)
        self.register_buffer('pca_embeddings', embeddings)

        return embeddings

    def forward(self, track_ids):
        condition_ids = track_ids // self.n_readlengths
        readlength_ids = track_ids % self.n_readlengths
        if hasattr(self, 'pca_embeddings'):
            embeddings = self.pca_embeddings[condition_ids]
        else:
            embeddings = self.condition_embedding(condition_ids)
        
        readlength_embeddings = self.readlength_embedding(readlength_ids)
        embeddings = torch.cat([embeddings, readlength_embeddings], dim=-1)
        
        processed_embeddings = self.cell_state_to_conv(embeddings)
        # generate weights
        kernels = [kernel_head(processed_embeddings).view(-1, *kernel_shape)  #  all three are (n_track, out, in, kernel)
                   for kernel_head, kernel_shape in zip(self.kernel_heads, self.kernel_shapes)]
        count_weight = self.count_fc_head(processed_embeddings).view(-1, *self.count_fc_shape) # should be (n_track, out, in)
        count_bias = self.count_bias_head(processed_embeddings) # (n_track, 1)
        
        biases = [bias_head(processed_embeddings) for bias_head in self.bias_heads] # shapes (n_track, 1)
        return (count_weight, count_bias, *kernels, *biases)


class ResNet(nn.Module):
    def __init__(self, input_dim, n_filters=64, n_layers=8, first_layer_kernel_size=25, kernel_size=3, dilation=True, padding="same"):
        super().__init__()
        self.input_dim = input_dim
        self.n_layers = n_layers
        self.kernel_size = kernel_size

        self.iconv = nn.Conv1d(self.input_dim, n_filters, first_layer_kernel_size, padding=padding)
        self.inorm = nn.BatchNorm1d(n_filters)

        self.rconvs = nn.ModuleList([
            nn.Conv1d(n_filters, n_filters,
                      kernel_size=kernel_size,
                      dilation=2**i if dilation else 1,
                      padding="same")
            for i in range(1, self.n_layers)
        ])
        self.batchnorms = nn.ModuleList([
            nn.BatchNorm1d(n_filters) for i in range(1, self.n_layers)
        ])
        
        self.relu = nn.ReLU()

    def forward(self, x):
        # convolutional block
        x = torch.permute(x, (0, 2, 1))  # B, embed, seq_len
        x = self.inorm(self.relu(self.iconv(x)))
        for i in range(self.n_layers - 1):
            x_conv = self.batchnorms[i](self.relu(self.rconvs[i](x)))
            x = x + x_conv
        x = torch.permute(x, (0, 2, 1))  # B, seq_len, mid_dim
        return x
