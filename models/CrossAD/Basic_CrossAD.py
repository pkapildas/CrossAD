import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import json

from .Attention_Blocks import *
from .EncDec import *
from .Context_Blocks import *
from .Graph_Blocks import *

class Configs:
    def __init__(self, json_path):
        with open(json_path) as f:
            configs = json.load(f)
            self.__dict__.update(configs)


class Basic_CrossAD(nn.Module):
    def __init__(self, configs):
        super(Basic_CrossAD, self).__init__()
        seq_len=configs.seq_len
        patch_len=configs.patch_len
        d_model=configs.d_model
        ms_kernerls=configs.ms_kernels
        ms_method=configs.ms_method

        self.n_scales = len(ms_kernerls)
        self.ms_utils = MS_Utils(ms_kernerls, ms_method)
        learnable_pe = getattr(configs, 'learnable_pe', False)
        self.pos_embedding = PositionalEmbedding(d_model, learnable=learnable_pe)
        self.patch_embedding = PatchEmbedding(d_model, patch_len=patch_len, stride=patch_len, padding=(patch_len-1), dropout=0.)
        
        # Cross Variable Attention
        self.use_cross_var = getattr(configs, 'use_cross_var', False)
        if self.use_cross_var:
            self.cross_var_attn = CrossVariableAttention(d_model=d_model, n_heads=configs.n_heads, dropout=configs.attn_dropout)
        
        # Dynamic Graph Neural Network
        self.use_gnn = getattr(configs, 'use_gnn', True)
        if self.use_gnn:
            self.dynamic_gnn = DynamicGraphModule(d_model=d_model, dropout=configs.attn_dropout)
        
        # Scale Attention
        self.scale_attention = ScaleAttention(n_scales=self.n_scales, d_model=configs.d_model)
        self.ms_t_lens = self.ms_utils._dummy_forward(seq_len)
        self.ms_p_lens = self.patch_embedding._dummy_forward(self.ms_t_lens)
        self.ms_t_lens_ = [PN * patch_len for PN in self.ms_p_lens]

        self.register_buffer('scale_ind_mask', self.ms_utils.scale_ind_mask(self.ms_p_lens))
        self.register_buffer('next_scale_mask', self.ms_utils.next_scale_mask(self.ms_p_lens))

        if "batch" in configs.norm.lower():
            encoder_norm = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
            decoder_norm = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            encoder_norm = nn.LayerNorm(d_model)
            decoder_norm = nn.LayerNorm(d_model)

        self.encoder=Encoder(
            layers=[
                EncoderLayer(
                    attention=AttentionLayer(ScaledDotProductAttention(attn_dropout=configs.attn_dropout), d_model=configs.d_model, n_heads=configs.n_heads, proj_dropout=configs.proj_dropout),
                    d_model=configs.d_model,
                    d_ff=configs.d_ff,
                    norm=configs.norm,
                    dropout=configs.ff_dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=encoder_norm
        )
        self.decoder=Decoder(
            layers=[
                DecoderLayer(
                    self_attention=AttentionLayer(ScaledDotProductAttention(attn_dropout=configs.attn_dropout), d_model=configs.d_model, n_heads=configs.n_heads, proj_dropout=configs.proj_dropout),
                    cross_attention=AttentionLayer(ScaledDotProductAttention(attn_dropout=configs.attn_dropout), d_model=configs.d_model, n_heads=configs.n_heads, proj_dropout=configs.proj_dropout),
                    d_model=configs.d_model,
                    d_ff=configs.d_ff,
                    norm=configs.norm,
                    dropout=configs.ff_dropout,
                    activation=configs.activation
                ) for _ in range(configs.d_layers)
            ],
            norm_layer=decoder_norm,
            projection=nn.Sequential(nn.Linear(configs.d_model, configs.patch_len), nn.Flatten(-2))
        )
        self.context_net=ContextNet(
            router = Router(seq_len=self.ms_t_lens[-1], n_vars=1, n_query=configs.n_query, topk=configs.topk),
            querys = nn.Parameter(torch.randn(configs.n_query, configs.query_len, configs.d_model)),
            extractor = Extractor(
                layers=[
                    ExtractorLayer(
                        cross_attention=AttentionLayer(ScaledDotProductAttention(attn_dropout=configs.attn_dropout), d_model=configs.d_model, n_heads=configs.n_heads, proj_dropout=configs.proj_dropout),
                        d_model=configs.d_model,
                        d_ff=configs.d_ff,
                        norm=configs.norm,
                        dropout=configs.ff_dropout,
                        activation=configs.activation
                    ) for _ in range(configs.m_layers)
                ],
                context_size=configs.bank_size,
                query_len=configs.query_len,
                d_model=configs.d_model,
                decay=configs.decay,
                epsilon=configs.epsilon
            )
        )

    def _forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        bs, t, c = x_enc.shape

        # CI
        x_enc = x_enc.permute(0, 2, 1)                                                          
        x_enc = x_enc.reshape(bs*c, t, 1)                                                       # x_enc: [bs*c x t x 1]
        router_input = x_enc

        # generate multi-scale x_enc
        ms_x_enc_list = self.ms_utils(x_enc)
        ms_gt = self.ms_utils.concat_sampling_list(ms_x_enc_list[1:] + [x_enc])                 # ms_gt: [bs*c x ms_t x 1]
        ms_gt = ms_gt.reshape(bs, c, -1)                                                        # ms_gt: [bs x c x ms_t]
        ms_gt = ms_gt.permute(0, 2, 1)                                                          # ms_gt: [bs x ms_t x c]

        # patch_embedding + pos_embedding
        x_enc = x_enc.permute(0, 2, 1)                                                          # x_enc: [bs*c x 1 x t]
        _, x_enc = self.patch_embedding(x_enc)                                                  # x_enc: [bs*c x pn x pl]
        for i in range(self.n_scales):
            x_enc_i = ms_x_enc_list[i]
            x_enc_i = x_enc_i.permute(0, 2, 1)                                                  # x_enc_i: [bs*c x 1 x t_i]
            x_enc_emb_i, _ = self.patch_embedding(x_enc_i)                                      # x_enc_emb_i: [bs*c x pn_i x d_model]
            ms_x_enc_list[i] = x_enc_emb_i
        ms_x_enc = self.ms_utils.concat_sampling_list(ms_x_enc_list)                            # ms_x_enc: [bs*c x ms_pn x d_model]

        pos_emb = self.pos_embedding(self.ms_p_lens[-1])                                        # pos_emb: [1 x pn x d_model]
        ms_pos_emb_list = self.ms_utils(pos_emb)
        ms_pos_emb = self.ms_utils.concat_sampling_list(ms_pos_emb_list)                        # ms_pos_emb: [1 x ms_pn x d_model]

        ms_x_enc = ms_x_enc + ms_pos_emb                                                        # ms_x_enc: [bs*c x ms_pn x d_model]

        # scale-independence encoder
        ms_x_enc, attn_weights = self.encoder(ms_x_enc, self.scale_ind_mask)                    # ms_x_enc_repr: [bs*c x ms_pn x d_model]

        # Inter-Variable Dependency Learning
        if self.use_cross_var:
            ms_x_enc = self.cross_var_attn.forward_with_shape(ms_x_enc, bs, c)

        # Dynamic Graph Neural Network
        if self.use_gnn:
            ms_x_enc = self.dynamic_gnn.forward_with_shape(ms_x_enc, bs, c)

        # period context
        if self.training:
            query_latent_distances, context = self.context_net(router_input, ms_x_enc)              # context: [N*query_len x d_model]
            context = context.unsqueeze(0).expand(bs*c, -1, -1)                                     # context: [bs*c x N*query_len x d_model]
            query_latent_distances = query_latent_distances.reshape(bs, c, 1)                       # query_latent_distances: [bs x c x 1]
            query_latent_distances = query_latent_distances.permute(0, 2, 1)                        # query_latent_distances: [bs x 1 x c]
        else:
            query_latent_distances = torch.zeros(bs, 1, c)
            context = self.context_net.extractor.concat_context(self.context_net.extractor.context)
            context = context.unsqueeze(0).expand(bs*c, -1, -1)

        # next-scale decoder
        ms_x_enc_list = self.ms_utils.split_2_list(ms_x_enc, ms_t_lens=self.ms_p_lens, mode="encoder")
        ms_x_dec_list = self.ms_utils.up(ms_x_enc_list, ms_t_lens=self.ms_p_lens)
        ms_x_dec = self.ms_utils.concat_sampling_list(ms_x_dec_list)                                                # ms_x_dec_repr: [bs*c x up(ms_pn) x d_model]

        ms_x_dec, self_attn_weights, cross_attn_weights = self.decoder(ms_x_dec, context, self.next_scale_mask)     # ms_x_dec: [bs*c x up(ms_pn)*pl]
        ms_x_dec = ms_x_dec.reshape(bs*c, -1, 1)                                                                    # ms_x_dec: [bs*c x up(ms_pn)*pl x 1]

        ms_x_dec = ms_x_dec.reshape(bs, c, -1)                                                                      # ms_x_dec: [bs x c x up(ms_pn)*pl]
        ms_x_dec = ms_x_dec.permute(0, 2, 1)                                                                        # ms_x_dec: [bs x up(ms_pn)*pl x c]
        ms_x_dec_list = self.ms_utils.split_2_list(ms_x_dec, ms_t_lens=self.ms_t_lens_, mode="decoder")
        for i in range(len(ms_x_dec_list)):
            ms_x_dec_list[i] = ms_x_dec_list[i][:, :self.ms_t_lens[i+1]]
        ms_x_dec = self.ms_utils.concat_sampling_list(ms_x_dec_list)                                                # ms_x_dec: [bs x ms_t x c]
        
        return ms_gt, ms_x_dec, query_latent_distances
    
    def _ms_anomaly_score(self, ms_x_dec, ms_gt):
        ms_score = F.mse_loss(ms_x_dec, ms_gt, reduction="none")                                                    # score: [bs x ms_t x c]
        ms_score_list = self.ms_utils.split_2_list(ms_score, ms_t_lens=self.ms_t_lens, mode="decoder")              # score_list: [[bs x t1 x c] ... [bs x ti x c]]

        # Dynamic Fusion
        fused_score = self.scale_attention(ms_score_list)
        return fused_score

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        ms_gt, ms_x_dec, query_latent_distances = self._forward(x_enc, x_mark_enc, x_dec, x_mark_dec)               # [bs x ms_t x c]
        return F.mse_loss(ms_x_dec, ms_gt), torch.mean(query_latent_distances), ms_x_dec, ms_gt

    def infer(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        ms_gt, ms_x_dec, query_latent_distances = self._forward(x_enc, x_mark_enc, x_dec, x_mark_dec)               # [bs x ms_t x c]                              
        return self._ms_anomaly_score(ms_x_dec, ms_gt), query_latent_distances                                      # [bs, t, c], [bs, 1, c]


class MS_Utils(nn.Module):
    def __init__(self, kernels, method="interval_sampling"):
        super().__init__()
        self.kernels = kernels
        self.method = method

    def concat_sampling_list(self, x_enc_sampling_list):
        return torch.concat(x_enc_sampling_list, dim=1)                                                             # [b x ms_t x -1]
    
    def split_2_list(self, ms_x_enc, ms_t_lens, mode="encoder"):
        if mode == "encoder":
            return list(torch.split(ms_x_enc, split_size_or_sections=ms_t_lens[:-1], dim=1))
        elif mode == "decoder":
            return list(torch.split(ms_x_enc, split_size_or_sections=ms_t_lens[1:], dim=1))
    
    def scale_ind_mask(self, ms_t_lens):
        L = sum(t_len for t_len in ms_t_lens[:-1])
        d = torch.cat([torch.full((t_len,), i) for i, t_len in enumerate(ms_t_lens[:-1])]).view(1, L, 1)
        dT = d.transpose(1, 2)
        return torch.where(d == dT, 0., -torch.inf).reshape(1, 1, L, L).contiguous().bool()
    
    def next_scale_mask(self, ms_t_lens):
        L = sum(t_len for t_len in ms_t_lens[1:])
        d = torch.cat([torch.full((t_len,), i) for i, t_len in enumerate(ms_t_lens[1:])]).view(1, L, 1)
        dT = d.transpose(1, 2)
        return torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, L, L).contiguous().bool()
    
    def down(self, x_enc):
        x_enc = x_enc.permute(0, 2, 1)                                                                              # [b x c x t]
        x_enc_sampling_list = []
        for kernel in self.kernels:
            pad_x_enc = F.pad(x_enc, pad=(0,kernel-1), mode="replicate")
            x_enc_i = pad_x_enc.unfold(dimension=-1, size=kernel, step=kernel)                                      # [b x c x t_i x kernel]
            if self.method == "average_pooling":
                x_enc_i = torch.mean(x_enc_i, dim=-1)
            elif self.method == "interval_sampling":
                x_enc_i = x_enc_i[:, :, :, 0]
            x_enc_sampling_list.append(x_enc_i.permute(0, 2, 1))                                                    # [b x t_i x c]

        return x_enc_sampling_list

    def up(self, x_enc_sampling_list, ms_t_lens):
        for i in range(len(ms_t_lens)-1):
            x_enc = x_enc_sampling_list[i].permute(0, 2, 1)                                                         # [b x c x t]
            up_x_enc = F.interpolate(x_enc, size=ms_t_lens[i+1], mode='nearest').permute(0, 2, 1)                   # [b x t x c]
            x_enc_sampling_list[i] = up_x_enc
        return x_enc_sampling_list
    
    @torch.no_grad()
    def _dummy_forward(self, input_len):
        dummy_x = torch.ones((1, input_len, 1))
        dummy_sampling_list = self.down(dummy_x)
        ms_t_lens = []
        for i in range(len(dummy_sampling_list)):
            ms_t_lens.append(dummy_sampling_list[i].shape[1])
        ms_t_lens.append(input_len)
        return ms_t_lens
    
    def forward(self, x_enc):
        return self.down(x_enc)


class CrossVariableAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        
        # Self-Attention across Variables
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # x: [BS*C, T, D]
        # We need to reshape to [BS, C, T, D]
        # But we don't know C here directly, we only know BS*C.
        # We can assume x came from the encoder [BS*C, T, D]
        
        # Wait, inside Basic_CrossAD we know BS and C. 
        # So we should pass BS and C to this forward method.
        pass

    def forward_with_shape(self, x, bs, c):
        # x: [BS*C, T, D]
        _, T, D = x.shape
        
        # 1. Reshape to [BS, C, T, D]
        x_reshaped = x.view(bs, c, T, D)
        
        # 2. Pool over Time -> [BS, C, D]
        # This represents the "summary" of each variable
        x_summary = torch.mean(x_reshaped, dim=2)
        
        # 3. Cross-Variable Attention -> [BS, C, D]
        # Query, Key, Value all from x_summary
        attn_out, _ = self.attn(x_summary, x_summary, x_summary)
        
        # 4. Residual Connection + Norm
        x_summary = self.norm(x_summary + self.dropout(attn_out))
        
        # 5. Broadcast back to Time -> [BS, C, T, D]
        # flexible broadcasting: [BS, C, 1, D] expand to T
        inter_var_context = x_summary.unsqueeze(2).expand(-1, -1, T, -1)
        
        # 6. Add to original features (Residual)
        x_enhanced = x_reshaped + inter_var_context
        
        # 7. Flatten back to [BS*C, T, D]
        return x_enhanced.reshape(bs*c, T, D)


class ScaleAttention(nn.Module):
    def __init__(self, n_scales, d_model):
        super().__init__()
        self.n_scales = n_scales
        # Simple attention implementation to determine weight for each scale
        # Input: list of scores [B, Ti, C]
        # We process them to generate weights [B, n_scales, 1, 1]
        
        self.attn_mlp = nn.Sequential(
            nn.Linear(n_scales, 64),
            nn.Tanh(),
            nn.Linear(64, n_scales),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, ms_score_list):
        # 1. Upsample all scores to the finest resolution
        # finest scale is the last one
        T_final = ms_score_list[-1].shape[1]
        B, _, C = ms_score_list[-1].shape
        
        upsampled_scores = []
        for i, score in enumerate(ms_score_list):
            if score.shape[1] != T_final:
                # [B, Ti, C] -> [B, C, Ti] -> Interpolate -> [B, C, T_final] -> [B, T_final, C]
                s = score.permute(0, 2, 1)
                s = F.interpolate(s, size=T_final, mode='linear')
                s = s.permute(0, 2, 1)
                upsampled_scores.append(s)
            else:
                upsampled_scores.append(score)
        
        # 2. Compute global context for attention
        # We pool each scale's score to get a summary statistic [B, 1] per scale
        # [B, n_scales]
        scale_contexts = []
        for s in upsampled_scores:
            # Mean score per scale
            mean_s = torch.mean(s, dim=(1, 2)) # [B]
            scale_contexts.append(mean_s)
        
        scale_contexts = torch.stack(scale_contexts, dim=1) # [B, n_scales]
        
        # 3. Compute weights
        weights = self.attn_mlp(scale_contexts) # [B, n_scales]
        weights = weights.unsqueeze(-1).unsqueeze(-1) # [B, n_scales, 1, 1]
        
        # 4. Multiply and Sum
        # Stack scores: [B, n_scales, T_final, C]
        stacked_scores = torch.stack(upsampled_scores, dim=1)
        
        weighted_scores = stacked_scores * weights
        fused_score = torch.sum(weighted_scores, dim=1) # [B, T_final, C]
        
        return fused_score
    

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000, learnable=False):
        super(PositionalEmbedding, self).__init__()
        
        self.learnable = learnable
        if self.learnable:
            self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
            nn.init.uniform_(self.pe, -0.02, 0.02)
        else:
            # Compute the positional encodings once in log space.
            pe = torch.zeros(max_len, d_model).float()
            pe.require_grad = False

            position = torch.arange(0, max_len).float().unsqueeze(1)
            div_term = (torch.arange(0, d_model, 2).float()
                        * -(math.log(10000.0) / d_model)).exp()

            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)

            pe = pe.unsqueeze(0)
            self.register_buffer('pe', pe)

    def forward(self, t_len):
        return self.pe[:, :t_len]
    

class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    @torch.no_grad()
    def _dummy_forward(self, input_lens):
        ms_p_lens = []
        for i, input_len in enumerate(input_lens):
            dummy_x = torch.ones((1, 1, input_len))
            dummy_x = self.padding_patch_layer(dummy_x)
            dummy_x = dummy_x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
            ms_p_lens.append(dummy_x.shape[2])
        return ms_p_lens

    def forward(self, x):
        # do patching
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x_patch = x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        x = self.value_embedding(x)
        return self.dropout(x), x_patch