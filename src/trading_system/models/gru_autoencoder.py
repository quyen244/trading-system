import torch
import torch.nn as nn
import torch.nn.functional as F
from trading_system.utils.logger import setup_logger

logger = setup_logger('HybridModel')

class OptimizedGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.2):
        super(OptimizedGRU, self).__init__()
        self.gru1 = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.gru2 = nn.GRU(hidden_dim, hidden_dim // 2, batch_first=True)
        self.ln2 = nn.LayerNorm(hidden_dim // 2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim // 2, 1) # Reconstruction head for target_return

    def forward(self, x):
        # x shape: (Batch, Window, Features)
        out, _ = self.gru1(x)
        out = self.ln1(out)
        out = torch.relu(out)
        
        # Second GRU layer
        out, hn = self.gru2(out)
        
        # Feature Extraction: Get final hidden state
        # hn shape: (num_layers, batch, hidden_size)
        last_hidden = hn[-1] # (Batch, Hidden//2)
        features = self.dropout(self.ln2(last_hidden))
        
        # Reconstruction (prediction of target_return)
        pred = self.fc(features)
        return pred, features


# 1. Custom GRU Block (Đã tinh chỉnh)
class MyGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.2, is_batch_first=True):
        """
        Custom GRU block with LayerNorm and Dropout
        Args:
            input_dim (int): Number of features in the input
            hidden_dim (int): Number of features in the hidden state
            dropout (float, optional): Dropout rate. Defaults to 0.2.
            is_batch_first (bool, optional): If True, the input and output tensors are provided as (batch, seq, feature). Defaults to True.
        """
        super(MyGRU, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=is_batch_first)
        self.ln = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out, _ = self.gru(x) 
        
        out = self.ln(out)
        out = F.relu(out)
        return self.dropout(out)

# 2. Lớp Attention (Quan trọng)
class AttentionBlock(nn.Module):
    def __init__(self, hidden_dim):
        """
        Attention block to compute attention weights for each time-step
        Args:
            hidden_dim (int): Number of features in the hidden state
        returns:
            context_vector (torch.Tensor): Context vector of shape (Batch, Hidden_Dim)
            weights (torch.Tensor): Attention weights of shape (Batch, Seq_Len, 1)
        """
        super(AttentionBlock, self).__init__()
        # Mạng nơ-ron nhỏ để tính trọng số alpha (score) cho từng time-step
        self.attention_score = nn.Linear(hidden_dim, 1)

    def forward(self, gru_output):
        # Tính điểm số cho từng time-step
        scores = self.attention_score(gru_output)  # (Batch, Seq_Len, 1)
        
        # Chuyển điểm số thành xác suất (weights) bằng Softmax
        weights = F.softmax(scores, dim=1)  # (Batch, Seq_Len, 1)
        
        # Nhân trọng số với output của GRU để tạo Context Vector
        # (Batch, Seq_Len, Hidden) * (Batch, Seq_Len, 1) -> Sum theo chiều Seq -> (Batch, Hidden)
        context_vector = torch.sum(weights * gru_output, dim=1)
        
        return context_vector, weights

# 3. Encoder: Input -> GRU Layers -> Attention -> Latent Vector
class GRUEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64, 32], dropout=0.2):
        """
        Encoder: Input -> GRU Layers -> Attention -> Latent Vector
        Args:
            input_dim (int): Number of features in the input
            hidden_dims (list, optional): List of hidden dimensions for GRU layers. Defaults to [64, 32].
            dropout (float, optional): Dropout rate. Defaults to 0.2.
        returns:
            context_vector (torch.Tensor): Context vector of shape (Batch, Hidden_Dim)
            weights (torch.Tensor): Attention weights of shape (Batch, Seq_Len, 1)
        """
        super(GRUEncoder, self).__init__()
        
        self.layers = nn.ModuleList()
        curr_input_dim = input_dim

        # Tạo stack các lớp GRU
        for h_dim in hidden_dims:
            self.layers.append(
                MyGRU(curr_input_dim, h_dim, dropout=dropout)
            )
            curr_input_dim = h_dim
            
        # Lớp Attention đặt ở cuối Encoder
        self.attention = AttentionBlock(curr_input_dim)
        self.final_dim = curr_input_dim

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        
        # out lúc này là (Batch, Seq_Len, Final_Hidden_Dim)
        # Đi qua Attention để thu về 1 vector duy nhất
        context_vector, weights = self.attention(out)
        
        return context_vector, weights

# 4. Decoder: Latent Vector -> Repeat -> GRU -> Reconstruction
class GRUDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim, hidden_dim=64, seq_len=10, dropout=0.2):
        """
        Decoder: Latent Vector -> Repeat -> GRU -> Reconstruction
        Args:
            latent_dim (int): Number of features in the latent vector
            output_dim (int): Number of features in the output
            hidden_dim (int, optional): Number of features in the hidden state. Defaults to 64.
            seq_len (int, optional): Sequence length. Defaults to 10.
            dropout (float, optional): Dropout rate. Defaults to 0.2.
        returns:
            reconstruction (torch.Tensor): Reconstructed features of shape (Batch, Seq_Len, Output_Dim)
        """
        super(GRUDecoder, self).__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        
        # Decoder GRU: Input là latent_dim, Output là hidden_dim
        self.gru = nn.GRU(latent_dim, hidden_dim, batch_first=True)
        
        # Lớp Linear cuối cùng để tái tạo lại feature ban đầu (26 features)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, latent_vector):
        # latent_vector shape: (Batch, Latent_Dim)
        
        # Repeat Vector: Lặp lại latent vector 10 lần để tạo thành chuỗi
        # Shape thành: (Batch, 10, Latent_Dim)
        x = latent_vector.unsqueeze(1).repeat(1, self.seq_len, 1)
        
        # Đưa vào GRU
        out, _ = self.gru(x) # (Batch, 10, Hidden_Dim)
        out = self.dropout(out)
        
        # Reconstruct lại features
        reconstruction = self.fc(out) # (Batch, 10, Output_Dim=26)
        
        return reconstruction

# 5. Full Autoencoder Model
class GRUAutoEncoder(nn.Module):
    def __init__(self, input_dim=26, seq_len=10):
        """
        Autoencoder model using GRU for feature extraction and reconstruction
        Args:
            input_dim (int): Number of features in the input
            seq_len (int): Sequence length. Defaults to 10.
        returns:
            reconstruction (torch.Tensor): Reconstructed features of shape (Batch, Seq_Len, Output_Dim)
            latent_vector (torch.Tensor): Latent vector of shape (Batch, Latent_Dim)
        """
        super(GRUAutoEncoder, self).__init__()
        
        # Encoder: input_dim -> 64 -> 32 -> Attention -> Context(32)
        self.encoder = GRUEncoder(input_dim, hidden_dims=[64, 32])
        
        # Decoder: Context(32) -> GRU(64) -> input_dim
        self.decoder = GRUDecoder(latent_dim=32, output_dim=input_dim, hidden_dim=64, seq_len=seq_len)

    def forward(self, x):
        # 1. Encode
        latent_vec, weights = self.encoder(x)
        
        # 2. Decode
        reconstruction = self.decoder(latent_vec)
        
        return reconstruction, latent_vec, weights

# --- Test thử shape ---
if __name__ == "__main__":
    dummy_input = torch.randn(32, 10, 26)
    
    model = GRUAutoEncoder(input_dim=26, seq_len=10)
    
    recon, latent, attn = model(dummy_input)
    
    print("Input Shape:", dummy_input.shape)          # (32, 10, 26)
    print("Recon Shape:", recon.shape)                # (32, 10, 26) -> Phải khớp Input
    print("Latent Shape:", latent.shape)              # (32, 32) -> Dùng cái này để visualize t-SNE hoặc classify
    print("Attention Weights:", attn.shape)           # (32 , 10 , 1)