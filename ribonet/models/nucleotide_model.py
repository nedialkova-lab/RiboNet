import torch
import torch.nn as nn
import torch.nn.functional as F
from .components import CellStateDecoder, ResNet

class InputEmbedding(nn.Module):
    def __init__(self, seq_len, lm, emb_lowered, left_special_tokens, right_special_tokens, window_size, overlap, fixed_length=False):
        super(InputEmbedding, self).__init__()

        self.seq_len = seq_len
        self.lm = lm
        self.emb_lowered = emb_lowered
        self.left_special_tokens = left_special_tokens
        self.right_special_tokens = right_special_tokens
        self.window_size = window_size
        self.overlap = overlap
        self.fixed_length = fixed_length
        self.model_embeddings = {}
        self.hook_list = []

        def getModelEmbeddings(name):
            def hook(model, input, output):
                self.model_embeddings[name] = output[0].detach()
            return hook
        if self.lm is not None:
            encoder_layers = self.lm.bert.encoder.layer if hasattr(self.lm, "bert") else self.lm.encoder._modules["layer"]
            for i,block in enumerate(encoder_layers):
                self.hook_list.append(block.register_forward_hook(getModelEmbeddings(i)))
            self.lm_weights = nn.Parameter(torch.zeros(len(self.hook_list))) # N_layers
        else:
            self.conv = ResNet(6, 64)
        
        if self.lm is not None and self.emb_lowered != self.lm.config.hidden_size:
            self.lmfc = torch.nn.Linear(self.lm.config.hidden_size, self.emb_lowered)
            self.dropout = nn.Dropout(0.7)
        elif self.lm is None and self.emb_lowered != 64:
            self.lmfc = torch.nn.Linear(64, self.emb_lowered)
            self.dropout = nn.Dropout(0.4)
    
    def get_embedding(self, x):
        self.lm.eval()
        # fixed_length models (e.g. SpeciesLM) were trained exclusively on full-length sequences
        # with no padding. Applying an attention mask would shrink the softmax denominator away
        # from the training distribution and produce noisy representations for shorter sequences.
        with torch.no_grad(), torch.amp.autocast(next(self.parameters()).device.type):
            if self.fixed_length:
                outputs = self.lm(x)
            else:
                attention_mask = (x != self.lm.config.pad_token_id).long()
                outputs = self.lm(x, attention_mask=attention_mask)

        embedding = [self.model_embeddings[i].float() for i in range(len(self.model_embeddings))]
        embedding = torch.stack(embedding,axis=-1) # B, L_in, hidden_dim, N_layers
        return embedding

    def embed_windows(self, x):
        batch_size, seq_len = x.shape
        window_size = self.window_size
        overlap = self.overlap

        stride = window_size - overlap

        # Split the input sequences into overlapping windows
        windows = []
        for start in range(0, seq_len - window_size + 1, stride):
            end = start + window_size
            window = x[:, start:end]
            windows.append(window)

        if seq_len % stride != 0:
            start = seq_len - window_size
            end = seq_len
            window = x[:, start:end]
            windows.append(window)

        # Embed each window
        window_embeddings = []
        for window in windows:
            # add eos token to empty sequences
            rows_to_modify = window[:, 0] == 0
            window[rows_to_modify, 0] = 2
            embedding = self.get_embedding(window)
            window_embeddings.append(embedding)

        # Concatenate the embeddings while averaging overlapping regions
        final_embedding = torch.zeros(
            batch_size, seq_len, window_embeddings[0].shape[2], window_embeddings[0].shape[3],
            device=x.device
        )
        count = torch.ones(batch_size, seq_len, device=x.device) #TODO 

        for i, embedding in enumerate(window_embeddings):
            start = i * stride
            end = start + window_size

            if end > final_embedding.size(1):
                end = final_embedding.size(1)
                start = end - window_size  # Adjust start to ensure the window size is maintained
            
            final_embedding[:, start:end, :, :] += embedding
            count[:, start:end] += 1
        count = count[:,:-2]

        # Average overlapping regions
        final_embedding = final_embedding[:,self.left_special_tokens:-self.right_special_tokens,:,:]

        final_embedding /= count.unsqueeze(-1).unsqueeze(-1)

        return final_embedding
    
    def forward(self, seq):
        
        if self.lm is not None:
            # Get hidden states from LMs
            embed = self.embed_windows(seq) # B, L_kmer, hidden, N_layers
            # do attention pooling over layers
            lm_weights = torch.softmax(self.lm_weights, dim=0)
            # TODO: remove these? 
            device = embed.device
            lm_weights = lm_weights.to(device=device)
            
            seq = (embed*lm_weights).sum(axis=-1)
        else:
            # one hot
            seq = F.one_hot(seq, num_classes=6).float() # B, seq_len, vocab_size(6)
            seq = self.conv(seq)
        
        if self.emb_lowered != seq.size(dim=-1):
            # Dense layer to match CDS conv dimensions
            seq = self.lmfc(seq) # B, seq_len, emb_lowered
            seq = self.dropout(seq)
        return seq
    
    
class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, num_heads, ff_dim, dropout_rate):
        super(TransformerEncoder, self).__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate

        # Multi-head attention module
        self.mha = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, dropout=dropout_rate,
                                         batch_first=True)

        # Position-wise feedforward layers
        self.fc1 = nn.Linear(input_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, input_dim)

        # Dropout and LayerNorm layers
        self.dropout = nn.Dropout(dropout_rate)
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.relu = nn.ReLU()

    def forward(self, src): # B, seq_length, input_dim
        # Multi-head attention
        mha_output = self.mha(src, src, src, need_weights=False)[0]
        src = src + self.dropout(mha_output)  # Apply residual connection
        src = self.norm1(src)  # Apply layer normalization

        # Position-wise feedforward
        ff_output = self.relu(self.fc1(src))
        ff_output = self.dropout(ff_output)
        ff_output = self.fc2(ff_output)
        src = src + self.dropout(ff_output)  # Apply residual connection
        src = self.norm2(src)  # Apply layer normalization

        return src # B, seq_length, input_dim
        
class RiboNet(nn.Module):
    def __init__(self, lm, configs, n_tracks = 1, freeze_lm = True):
        super().__init__()
        self.lm = lm
        self.seq_len = configs.seq_len
        self.n_encoder_layers = configs.n_encoder_layers
        self.ff_dim = configs.ff_dim
        self.num_heads = configs.num_heads
        self.dropout_rate = configs.dropout_rate
        self.final_kernel_size = configs.final_kernel_size
        self.n_tracks = n_tracks
        self.freeze_lm = freeze_lm
        self.left_special_tokens = configs.left_special_tokens
        self.right_special_tokens = configs.right_special_tokens
        self.fixed_length = configs.fixed_length
        self.emb_lowered = configs.emb_lowered
        self.decoder = configs.decoder
        self.use_annotation_emb = configs.use_annotation_emb
        self.window_size = configs.window_size
        self.overlap = configs.overlap
        # Freeze the LMs if necessary
        if self.freeze_lm:
            if self.lm is not None:
                for p in self.lm.parameters():
                    if p.requires_grad:
                        p.requires_grad = False
        
        # Trainable layers
        self.input_module = InputEmbedding(self.seq_len, self.lm, self.emb_lowered, self.left_special_tokens, self.right_special_tokens, self.window_size, self.overlap, self.fixed_length)
        # Positional embeddings
        self.pos_embed = nn.Embedding(self.seq_len, self.emb_lowered)
        self.annot_embed = nn.Embedding(6, self.emb_lowered)

        self.transformer_encoder = nn.ModuleList([TransformerEncoder(self.emb_lowered, self.num_heads, self.ff_dim, dropout_rate=self.dropout_rate) 
                                                  for i in range(self.n_encoder_layers)])
        
        self.relu = nn.ReLU()        
        if self.decoder is None:
            self.out_conv = torch.nn.Conv1d(self.emb_lowered, self.n_tracks, kernel_size=self.final_kernel_size, padding="same")
            self.count_fc = torch.nn.Linear(self.emb_lowered, self.n_tracks)
        else:
            kernel_shape = (1, self.emb_lowered, self.final_kernel_size)

            self.cell_state_decoder = CellStateDecoder(
                n_tracks=self.n_tracks,
                n_conditions=self.decoder["n_conditions"],
                embedding_dim=self.decoder["embedding_dim"],
                dropout_rate=0.2,
                kernel_shapes=(kernel_shape,),
                count_fc_shape=(1, self.emb_lowered),
                pca_embeddings=self.decoder["pca_embeddings"]
            )
        
        self.global_avgpool = torch.nn.AdaptiveAvgPool1d(1)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_lm:
            # ensure the lm is always in eval
            if self.lm is not None:
                self.lm.eval()
            
    def forward(self, x):
        # input embeddings
        seq = self.input_module(x["seq"])
        positions = torch.arange(seq.size(1), device=seq.device)
        if self.use_annotation_emb:
            seq = seq + self.pos_embed(positions) + self.annot_embed(x["annotation"])
        else:
            seq = seq + self.pos_embed(positions)

        # self-attention block
        for attn_layer in self.transformer_encoder:
            seq = attn_layer(seq)

        bottleneck = torch.permute(seq, (0, 2, 1)) # B, n_filters, seq_len

        # profile and count head: either trainable conv kernels or from cell state decoder
        if not hasattr(self, 'cell_state_decoder'):
            # Output heads
            profile = self.out_conv(bottleneck)
            # count head
            count = self.global_avgpool(bottleneck).squeeze() # B, n_filters
            count = self.count_fc(count) # B, n_tracks
        else:
            all_state_ids = torch.arange(self.n_tracks, device = bottleneck.device)
            count_w, count_b, seq_k, seq_b = self.cell_state_decoder(all_state_ids)

            all_profiles = []

            for i in range(self.n_tracks):
                # profile head
                profile = F.conv1d(bottleneck, seq_k[i], bias=seq_b[i], padding="same")
                all_profiles.append(profile)
            profile = torch.cat(all_profiles, dim=1) # B, n_tracks, seq_len

            # count head
            count = self.global_avgpool(bottleneck).squeeze() # B, n_filters
            count = F.linear(count, count_w.squeeze(1), count_b.squeeze(1)) # killed one dim in the count weight, bias matrices --> count_w (8,128), count_b (8), count (B, 8)
        
        rna_norm_factor = torch.log(x["rna"].float()) 
        count = count + rna_norm_factor # logRibo = logRNA + logTE
        
        return profile.float(), count.float()
