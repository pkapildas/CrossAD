import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        self.x_proj = nn.Linear(self.d_inner, (2 * self.d_state + 1), bias=False)  # B, C, delta
        
        # A and D are parameters
        self.A_log = nn.Parameter(torch.log(torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Delta parameter (fixed / learned)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU() # SiLU (Swish) activation

    def ssm(self, x):
        """
        x: (B, L, D_inner)
        Returns: y (B, L, D_inner), h_last (B, D_inner, d_state)
        """
        B, L, D_inner = x.shape
        
        # Project x to get dynamic parameters
        # x_proj: (B, L, dt_rank + 2*d_state)
        x_proj = self.x_proj(x)
        
        # Split projections
        # dt_proj (1 -> D_inner) handles the broadcast from dt_rank to D_inner
        # But here x_proj is (B,L, ...). 
        # In Mamba: dt = Softplus(Linear(dt_rank)(x))
        # Here we simplified: x_proj[:,:,0] is passed to dt_proj? 
        # Let's align with the init: self.dt_proj = nn.Linear(1, self.d_inner)
        
        delta = F.softplus(self.dt_proj(x_proj[:, :, 0:1])) # (B, L, D_inner)
        B_ssm = x_proj[:, :, 1:1+self.d_state]             # (B, L, d_state)
        C_ssm = x_proj[:, :, 1+self.d_state:]              # (B, L, d_state)
        
        # Discretize A
        # A_log is (D_inner, d_state)
        A = -torch.exp(self.A_log)                         # (D_inner, d_state)
        
        # Pre-compute discretized terms for the loop to save time
        # dA = exp(delta * A)
        # delta: (B, L, D, 1), A: (1, 1, D, N) -> (B, L, D, N)
        dA = torch.exp(delta.unsqueeze(-1) * A.view(1, 1, self.d_inner, self.d_state))
        
        # dB = delta * B
        # delta: (B, L, D, 1), B_ssm: (B, L, 1, N) -> (B, L, D, N)
        dB = delta.unsqueeze(-1) * B_ssm.unsqueeze(2)
        
        # Scan
        # h_t = dA * h_{t-1} + dB * x_t
        # y_t = C * h_t
        
        x_us = x.unsqueeze(-1) # (B, L, D, 1)
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device)
        ys = []
        
        # Simple sequential scan (efficient enough for L~100)
        for t in range(L):
            h = dA[:, t] * h + dB[:, t] * x_us[:, t]
            
            # y_t = sum(h * C)
            # h: (B, D, N), C_ssm[:, t]: (B, N)
            # Need to broadcast C to D
            y_t = torch.sum(h * C_ssm[:, t].unsqueeze(1), dim=-1) # (B, D)
            ys.append(y_t)
            
        y = torch.stack(ys, dim=1) # (B, L, D)
        
        # Return y and the final hidden state h for the Router
        return y + x * self.D, h

    def forward(self, x):
        B, L, D = x.shape
        
        xz = self.in_proj(x) # (B, L, 2*D_inner)
        x, z = xz.chunk(2, dim=-1)
        
        # Conv
        x = x.transpose(1, 2) # (B, D, L)
        x = self.conv1d(x)[:, :, :L]
        x = x.transpose(1, 2)
        
        x = self.act(x)
        y, h_last = self.ssm(x)
        
        # Gate
        y = y * self.act(z)
        
        out = self.out_proj(y)
        return self.dropout(out), h_last

class SSMEncoderLayer(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1, activation="gelu"):
        super().__init__()
        self.mamba = MambaBlock(d_model, d_state, d_conv, expand, dropout)
        self.norm = nn.LayerNorm(d_model) # Pre-norm usually
        
    def forward(self, x, attn_mask=None):
        # Mamba layer (replaces Attention + FFN in one block, or just Attention?)
        # Mamba usually replaces Transformer Block entirely.
        
        mamba_out, h_last = self.mamba(x)
        x = self.norm(x + mamba_out)
        return x, h_last 

class SSMEncoder(nn.Module):
    def __init__(self, layers, norm_layer=None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        
    def forward(self, x, attn_mask=None):
        states = []
        for layer in self.layers:
            x, h = layer(x, attn_mask=attn_mask)
            states.append(h)
        
        if self.norm is not None:
            x = self.norm(x)
            
        return x, states

class SSMDecoderLayer(nn.Module):
    def __init__(self, d_model, cross_attention, d_state=16, d_conv=4, expand=2, d_ff=None, dropout=0.1, activation="gelu"):
        super().__init__()
        
        # Self-Mixing (Mamba)
        self.self_mixing = MambaBlock(d_model, d_state, d_conv, expand, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        # Cross-Attention (Standard)
        self.cross_attention = cross_attention
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # FFN
        d_ff = d_ff or 4 * d_model
        self.norm3 = nn.LayerNorm(d_model)
        self.activation_type = activation
        
        if self.activation_type == "swiglu":
            self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
            self.conv2 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
            self.conv3 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        else:
            self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
            self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1) # standard FFN
            self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        # 1. Self-Mixing (Mamba)
        # Unpack result
        mamba_out, _ = self.self_mixing(x)
        x = self.norm1(x + mamba_out)
        
        # 2. Cross-Attention
        x_, cross_attn_weights = self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask,
        )
        x = self.norm2(x + self.dropout(x_))
        
        # 3. FFN
        y = x
        if self.activation_type == "swiglu":
            y_t = y.transpose(-1, 1)
            gate = F.silu(self.conv1(y_t))
            val = self.conv2(y_t)
            y = self.dropout(self.conv3(gate * val).transpose(-1, 1))
        else:
            y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
            y = self.dropout(self.conv2(y).transpose(-1, 1))
        
        return self.norm3(x + y), None, cross_attn_weights

class SSMDecoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection
        
    def forward(self, x, cross, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x, _, cross_attn_weights = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
            
        if self.norm is not None:
            x = self.norm(x)
            
        if self.projection is not None:
            x = self.projection(x)
            
        return x, None, cross_attn_weights
