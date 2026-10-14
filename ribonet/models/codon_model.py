import torch
import torch.nn as nn
import torch.nn.functional as F
from .components import CellStateDecoder, ResNet

class InputEmbedding(nn.Module):
    def __init__(self, seq_len, vocab_size, lm5, lm3, utr_dim,
                 utr_pool_size, left_special_tokens, right_special_tokens, fixed_length=[False, False]):
        super(InputEmbedding, self).__init__()

        self.seq_len = seq_len
        self.utr_pool_size = utr_pool_size
        self.vocab_size = vocab_size
        self.lm5 = lm5; self.lm3 = lm3
        self.utr_dim = utr_dim
        self.left_special_tokens = left_special_tokens
        self.right_special_tokens = right_special_tokens
        self.fixed_length = fixed_length
        self.model_embeddings = {"lm5":{}, "lm3":{}}
        self.hook_list = {"lm5":[], "lm3":[]}
        # Hook the hooks
        def getModelEmbeddings(name, lm):
            def hook(model, input, output):
                self.model_embeddings[lm][name] = output
            return hook
        if self.lm5 is not None:
            encoder_layers = self.lm5.bert.encoder.layer if hasattr(self.lm5, "bert") else self.lm5.encoder._modules["layer"]
            for i,block in enumerate(encoder_layers):
                self.hook_list["lm5"].append(block.register_forward_hook(getModelEmbeddings(i,"lm5")))
            self.lm5_weights = nn.Parameter(torch.zeros(len(self.hook_list["lm5"]))) # N_layers
        else:
            self.utr5_conv = ResNet(6, 64)

        if self.lm3 is not None:
            encoder_layers = self.lm3.bert.encoder.layer if hasattr(self.lm3, "bert") else self.lm3.encoder._modules["layer"]
            for i,block in enumerate(encoder_layers):
                self.hook_list["lm3"].append(block.register_forward_hook(getModelEmbeddings(i,"lm3")))
            self.lm3_weights = nn.Parameter(torch.zeros(len(self.hook_list["lm3"]))) # N_layers
        else:
            self.utr3_conv = ResNet(6, 64)
        
        self.avgpool5 = torch.nn.AdaptiveAvgPool1d(self.utr_pool_size[0])
        self.avgpool3 = torch.nn.AdaptiveAvgPool1d(self.utr_pool_size[1])

        if self.lm5 is not None and self.utr_dim != self.lm5.config.hidden_size:
            self.lm5fc = torch.nn.Linear(self.lm5.config.hidden_size, self.utr_dim)
            self.dropout = nn.Dropout(0.7)
        elif self.lm5 is None and self.utr_dim != 64:
            self.lm5fc = torch.nn.Linear(64, self.utr_dim)
            self.dropout = nn.Dropout(0.4)
        if self.lm3 is not None and self.utr_dim != self.lm3.config.hidden_size:
            self.lm3fc = torch.nn.Linear(self.lm3.config.hidden_size, self.utr_dim)
            self.dropout = nn.Dropout(0.7)
        elif self.lm3 is None and self.utr_dim != 64:
            self.lm3fc = torch.nn.Linear(64, self.utr_dim)
            self.dropout = nn.Dropout(0.4)
    
    def get_embedding(self, x, lm):
        active_lm = self.lm5 if lm == "lm5" else self.lm3
        # fixed_length models (e.g. SpeciesLM) were trained exclusively on full-length sequences
        # with no padding. Applying an attention mask would shrink the softmax denominator away
        # from the training distribution and produce noisy representations for shorter sequences.
        is_fixed_length = self.fixed_length[0] if lm == "lm5" else self.fixed_length[1]
        if is_fixed_length:
            active_lm(x)
        else:
            attention_mask = (x != active_lm.config.pad_token_id).long()
            active_lm(x, attention_mask=attention_mask)
        embedding = [self.model_embeddings[lm][i][0].float() for i in range(len(self.model_embeddings[lm]))]
        embedding = torch.stack(embedding,axis=-1) # B, L_in, hidden_dim, N_layers
        # extract non-special tokens
        lst = self.left_special_tokens[0] if lm=="lm5" else self.left_special_tokens[1]
        rst = self.right_special_tokens[0] if lm=="lm5" else self.right_special_tokens[1]
        embedding = embedding[:,lst:-rst,:,:]
        return embedding
    
    def forward(self, cds, utr5, utr3):
        # CDS one hot
        cds = F.one_hot(cds, num_classes=self.vocab_size).float() # B, seq_len, vocab_size(65)
        
        if self.lm5 is not None:
            # Get hidden states from UTR LMs
            embed5 = self.get_embedding(utr5, "lm5") # B, L_kmer, hidden, N_layers
            # do attention pooling over layers
            lm5_weights = torch.softmax(self.lm5_weights, dim=0)
            embed5 = (embed5*lm5_weights).sum(axis=-1)
        else:
            # UTR one hot
            utr5 = F.one_hot(utr5, num_classes=6).float() # B, seq_len, vocab_size(6)
            embed5 = self.utr5_conv(utr5)
        
        if self.lm3 is not None:
            # Get hidden states from UTR LMs
            embed3 = self.get_embedding(utr3, "lm3") # B, L_kmer, hidden, N_layers
            # do attention pooling over layers
            lm3_weights = torch.softmax(self.lm3_weights, dim=0)
            embed3 = (embed3*lm3_weights).sum(axis=-1)
        else:
            # UTR one hot
            utr3 = F.one_hot(utr3, num_classes=6).float() # B, seq_len, vocab_size(6)
            embed3 = self.utr3_conv(utr3)

        # Reduce length of UTRs by pooling
        embed5 = embed5[:,:-3,:] # remove start codon
        embed3 = embed3[:,3:,:] # remove stop codon
        embed5 = torch.permute(self.avgpool5(torch.permute(embed5,(0,2,1))),(0,2,1))
        embed3 = torch.permute(self.avgpool3(torch.permute(embed3,(0,2,1))),(0,2,1))
        if self.utr_dim != embed5.size(dim=-1):
            # Dense layer to match CDS conv dimensions
            embed5 = self.lm5fc(embed5) # B, seq_len, utr_embed
            embed5 = self.dropout(embed5)
        
        if self.utr_dim != embed3.size(dim=-1):
            embed3 = self.lm3fc(embed3) # B, seq_len, utr_embed​
            embed3 = self.dropout(embed3)

        return cds, embed5, embed3
    
    
class ConvTower(nn.Module):
    def __init__(self, input_dim, n_filters, kernel_size, padding = "same"):
        super(ConvTower, self).__init__()
        self.input_dim = input_dim
        self.n_filters = n_filters
        self.kernel_size = kernel_size

        self.conv1 = nn.Conv1d(self.input_dim, self.n_filters[0], kernel_size=self.kernel_size, padding=padding)
        self.norm1 = nn.BatchNorm1d(n_filters[0])
        self.conv2 = nn.Conv1d(self.n_filters[0], self.n_filters[1], kernel_size=self.kernel_size, padding=padding)
        self.norm2 = nn.BatchNorm1d(self.n_filters[1])
        self.conv3 = nn.Conv1d(self.n_filters[1], self.n_filters[2], kernel_size=self.kernel_size, padding=padding)
        self.norm3 = nn.BatchNorm1d(self.n_filters[2])
        self.conv4 = nn.Conv1d(self.n_filters[2], self.n_filters[3], kernel_size=self.kernel_size, padding=padding)
        self.norm4 = nn.BatchNorm1d(self.n_filters[3])
        self.conv5 = nn.Conv1d(self.n_filters[3], self.n_filters[4], kernel_size=self.kernel_size, padding=padding)
        self.norm5 = nn.BatchNorm1d(self.n_filters[4])
        self.relu = nn.ReLU()

    def forward(self, x):
        # convolutional block
        x = torch.permute(x, (0, 2, 1)) # B, embed​, seq_len
        x = self.norm1(self.relu(self.conv1(x)))
        x = self.norm2(self.relu(self.conv2(x)))
        x = self.norm3(self.relu(self.conv3(x)))
        x = self.norm4(self.relu(self.conv4(x)))
        x = self.norm5(self.relu(self.conv5(x)))
        x = torch.permute(x,(0, 2, 1)) # B, seq_len, n_filters

        return x
    
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
    def __init__(self, lm5, lm3, configs, n_tracks = 1,
                 freeze_lm = True):
        super().__init__()
        self.lm5 = lm5; self.lm3 = lm3
        self.seq_len = configs.seq_len
        self.utr_len = configs.utr_len
        self.utr_pool_size = configs.utr_pool_size
        self.vocab_size = configs.vocab_size
        self.n_filters = configs.n_filters
        self.kernel_size = configs.kernel_size
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
        self.decoder = configs.decoder
        # Freeze the LMs if necessary
        if self.freeze_lm:
            if self.lm5 is not None:
                for p in self.lm5.parameters():
                    if p.requires_grad:
                        p.requires_grad = False
            if self.lm3 is not None:
                for p in self.lm3.parameters():
                    if p.requires_grad:
                        p.requires_grad = False
        
        # Trainable layers
        self.input_module = InputEmbedding(self.seq_len, self.vocab_size, self.lm5, self.lm3, self.n_filters[4],
                                           self.utr_pool_size, self.left_special_tokens, self.right_special_tokens,
                                           self.fixed_length)
        self.conv_tower = ConvTower(self.vocab_size, self.n_filters, self.kernel_size)
        self.pos_embed = nn.Embedding(self.utr_pool_size[0]+self.seq_len+self.utr_pool_size[1] + 1, self.n_filters[4]) # +1 because we have max_length positions and an addditional positional embedding token for "no" sequence
        self.transformer_encoder = nn.ModuleList([TransformerEncoder(self.n_filters[4], self.num_heads, self.ff_dim, dropout_rate=self.dropout_rate) 
                                                  for i in range(self.n_encoder_layers)])
        utr5size = int((self.final_kernel_size*3)*(self.utr_pool_size[0]/self.utr_len[0]))
        utr3size = int((self.final_kernel_size*3)*(self.utr_pool_size[1]/self.utr_len[1]))

        # decoder: Either trainable kernel parameters or predicted kernels from cell state decoder
        if self.decoder is None: 
            self.out_conv_cds = torch.nn.Conv1d(self.n_filters[4], self.n_tracks, kernel_size=self.final_kernel_size, padding="same")
            self.out_conv_utr5 = torch.nn.Conv1d(self.n_filters[4], self.n_tracks, kernel_size=utr5size, padding="same")
            self.out_conv_utr3 = torch.nn.Conv1d(self.n_filters[4], self.n_tracks, kernel_size=utr3size, padding="same")
            self.count_fc = torch.nn.Linear(self.n_filters[4], self.n_tracks)
        else:
            utr5_kernel_shape=(1,self.n_filters[4], utr5size)
            cds_kernel_shape=(1, self.n_filters[4], self.final_kernel_size)
            utr3_kernel_shape=(1, self.n_filters[4], utr3size)

            self.cell_state_decoder = CellStateDecoder(
                n_tracks=self.n_tracks,
                n_conditions=self.decoder["n_conditions"],
                embedding_dim=self.decoder["embedding_dim"],
                dropout_rate=0.2,
                kernel_shapes=(utr5_kernel_shape, cds_kernel_shape, utr3_kernel_shape),
                count_fc_shape=(1, self.n_filters[4]),
                pca_embeddings=self.decoder["pca_embeddings"]
            )
        
        self.global_avgpool = torch.nn.AdaptiveAvgPool1d(1)
        self.ablate_utr5 = getattr(configs, "ablate_utr5", False)
        self.ablate_utr3 = getattr(configs, "ablate_utr3", False)
    
    def train(self, mode=True):
        super().train(mode)
        if self.freeze_lm:
            # ensure the lm is always in eval
            if self.lm5 is not None:
                self.lm5.eval()
            if self.lm3 is not None:
                self.lm3.eval()

    def forward(self, x):
        # input embeddings
        cds, embed5, embed3 = self.input_module(x["cds"], x["utr5"], x["utr3"])
        # UTR ablations: zero out embedding before concatenation so no sequence info leaks
        if self.ablate_utr5:
            embed5 = torch.zeros_like(embed5)
        if self.ablate_utr3:
            embed3 = torch.zeros_like(embed3)
        # conv layers
        cds = self.conv_tower(cds)
        # concatenate and position encode
        seq = torch.cat([embed5, cds, embed3], axis=1)
        #positions = torch.arange(seq.size(1), device=seq.device)
        positions = x["pos"]
        seq = seq + self.pos_embed(positions) # B, utrdim5+seq_len+utrdim3, embed​
        # self-attention block
        for attn_layer in self.transformer_encoder:
            seq = attn_layer(seq)
        bottleneck = torch.permute(seq, (0, 2, 1)) # B, n_filters, utrdim5+seq_len+utrdim3

        # profile and count head: either trainable conv kernels or from cell state decoder
        if not hasattr(self, "cell_state_decoder"):
            profile = torch.cat((self.out_conv_utr5(bottleneck)[:,:,:embed5.size(dim=1)],
                                self.out_conv_cds(bottleneck)[:,:,embed5.size(dim=1):-embed3.size(dim=1)],
                                self.out_conv_utr3(bottleneck)[:,:,-embed3.size(dim=1):]), dim=2) # B, n_tracks, seq_len
            # count head
            count = self.global_avgpool(bottleneck).squeeze() # B, n_filters
            count = self.count_fc(count) # B, n_tracks
        else:
            # now the cell state decoder weights for profile prediction
            all_state_ids = torch.arange(self.n_tracks, device = bottleneck.device)
            
            # get params
            count_w, count_b, utr5_k, cds_k, utr3_k, utr5_b, cds_b, utr3_b = self.cell_state_decoder(all_state_ids)
            
            all_profiles = []
            # iterate over tracks
            for i in range(self.n_tracks):
                # perform the convs
                utr5_p = F.conv1d(bottleneck, utr5_k[i], bias=utr5_b[i], padding="same")
                cds_p = F.conv1d(bottleneck, cds_k[i], bias=cds_b[i], padding="same")
                utr3_p = F.conv1d(bottleneck, utr3_k[i], bias=utr3_b[i], padding="same")

                # stitch the profiles back together
                stitched_profile = torch.cat((utr5_p[:,:,:embed5.size(1)],
                                            cds_p[:,:,embed5.size(1):-embed3.size(1)],
                                            utr3_p[:,:,-embed3.size(1):]), dim=2)
                # add profile of this track to list
                all_profiles.append(stitched_profile)
            # concat the list
            profile = torch.cat(all_profiles, dim=1)
            
            # count head
            count = self.global_avgpool(bottleneck).squeeze(2) # B, n_filters
            count = F.linear(count, count_w.squeeze(1), count_b.squeeze(1)) # killed one dim in the count weight, bias matrices --> count_w (8,128), count_b (8), count (B, 8)
        
        rna_norm_factor = torch.log(x["rna"].float()) #+ self.rna_weight.float()
        count = count + rna_norm_factor # logRibo = logRNA + logTE

        return profile.float(), count.float()
