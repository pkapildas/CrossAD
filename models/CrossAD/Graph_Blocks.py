import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicGraphModule(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # Learnable weights for dynamic adjacency matrix
        # A = Softmax(Relu(Q @ K.T) / scale)
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        
        # GCN weight
        self.gcn_linear = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        self.activation = nn.ReLU()

    def forward(self, x):
        """
        x: [BS, c, t, d]
        """
        bs, c, t, d = x.shape
        
        # Pool time to get node features -> [BS, c, d]
        nodes = torch.mean(x, dim=2)
        
        # Compute Dynamic Adjacency Matrix
        # Q, K: [BS, c, d]
        Q = self.W_q(nodes)
        K = self.W_k(nodes)
        
        # A_logits: [BS, c, c]
        A_logits = torch.matmul(Q, K.transpose(-1, -2)) / (d ** 0.5)
        # Apply Softmax to normalize
        A = F.softmax(A_logits, dim=-1)
        
        # GCN Operation
        # H' = A * (H * W)
        # H_transformed: [BS, c, d]
        H_transformed = self.gcn_linear(nodes)
        
        # Message Passing
        # [BS, c, c] @ [BS, c, d] -> [BS, c, d]
        nodes_out = torch.matmul(A, H_transformed)
        
        # Activation + Dropout
        nodes_out = self.activation(nodes_out)
        nodes_out = self.dropout(nodes_out)
        
        # Residual + Norm
        nodes_out = self.norm(nodes + nodes_out)
        
        # Broadcast back to time dimension
        # [BS, c, d] -> [BS, c, 1, d] -> [BS, c, t, d]
        out = nodes_out.unsqueeze(2).expand(-1, -1, t, -1)
        
        # Add to original input (another residual connection from the input of the block)
        return x + out

    def forward_with_shape(self, x, bs, c):
        # Wrapper to match Basic_CrossAD signature if needed
        # x: [BS*C, T, D]
        # We need to reshape to [BS, C, T, D]
        _, T, D = x.shape
        x_reshaped = x.view(bs, c, T, D)
        
        out = self.forward(x_reshaped)
        
        # Reshape back to [BS*C, T, D]
        return out.reshape(bs*c, T, D)
