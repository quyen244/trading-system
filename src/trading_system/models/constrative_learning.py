import torch
from torch import nn
from torch.nn import functional as F
from trading_system.utils.logger import setup_logger
from trading_system.models.gru_autoencoder import GRUAutoEncoder

import torch
from torch import nn
from torch.nn import functional as F

# 1. Định nghĩa hàm Triplet Loss
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # Tính khoảng cách Euclidean
        dist_pos = F.pairwise_distance(anchor, positive, p=2)
        dist_neg = F.pairwise_distance(anchor, negative, p=2)
        
        # Công thức: max(d(a,p) - d(a,n) + margin, 0)
        loss = torch.mean(torch.clamp(dist_pos - dist_neg + self.margin, min=0.0))
        return loss

# 2. Module Contrastive Learning hoàn chỉnh
class ContrastiveModel(nn.Module):
    def __init__(self, model , latent_dim=32, projection_dim=16, device='cpu'):
        super(ContrastiveModel, self).__init__()
        
        self.device = device
        
        # Lấy phần Encoder từ Autoencoder có sẵn
        # Giả sử encoder_model là instance của TradingAutoencoder bạn đã viết
        self.encoder = model.encoder 
        
        # Projection Head: MLP nhỏ để ánh xạ latent space sang không gian tính Loss
        # Giúp Latent gốc giữ được nhiều thông tin ngữ nghĩa hơn
        self.projection_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, projection_dim)
        )
        
        self.to(device)

    def forward_one(self, x):
        # Chạy qua Encoder để lấy Latent Vector
        # Lưu ý: encoder trả về (context_vector, weights)
        latent, _ = self.encoder(x)
        
        # Chạy qua Projection Head
        proj = self.projection_head(latent)
        
        # Chuẩn hóa vector (L2 Normalization) giúp training ổn định hơn
        proj = F.normalize(proj, p=2, dim=1)
        return proj, latent

    def forward(self, anchor, positive, negative):
        # Chạy cả 3 nhánh
        proj_a, _ = self.forward_one(anchor)
        proj_p, _ = self.forward_one(positive)
        proj_n, _ = self.forward_one(negative)
        
        # Trả về các vector đã project để tính Loss
        return proj_a, proj_p, proj_n

    # Hàm tiện ích để lấy feature sau khi train xong (dùng cho t-SNE hoặc Classifier)
    def get_embedding(self, x):
        with torch.no_grad():
            self.eval()
            x = x.to(self.device)
            latent, _ = self.encoder(x)
            self.train()
        return latent.cpu().numpy()

# --- Ví dụ cách sử dụng (Training Loop) ---
if __name__ == "__main__":
    # Giả lập dữ liệu
    BATCH_SIZE = 32
    SEQ_LEN = 10
    FEAT_DIM = 26
    
    # 1. Khởi tạo
    # Import class TradingAutoencoder từ bài trước
    # full_ae = TradingAutoencoder(...) 
    # model = ContrastiveModel(full_ae, device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Ở đây mình giả lập instance encoder để code chạy được
    # (Bạn thay bằng model thật của bạn)
    from unittest.mock import MagicMock
    dummy_encoder = MagicMock()
    dummy_encoder.encoder = lambda x: (torch.randn(x.size(0), 32), None) # Giả lập output
    
    model = ContrastiveModel(dummy_encoder, device='cpu')
    criterion = TripletLoss(margin=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 2. Giả lập 1 batch dữ liệu Triplet
    # Trong thực tế, bạn cần viết DataLoader để lấy ngẫu nhiên:
    # - Anchor: 1 đoạn chart Buy
    # - Positive: 1 đoạn chart Buy khác (hoặc augment từ anchor)
    # - Negative: 1 đoạn chart Sell/Hold
    anchor_img = torch.randn(BATCH_SIZE, SEQ_LEN, FEAT_DIM)
    pos_img = torch.randn(BATCH_SIZE, SEQ_LEN, FEAT_DIM)
    neg_img = torch.randn(BATCH_SIZE, SEQ_LEN, FEAT_DIM)

    # 3. Training Step
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    p_a, p_p, p_n = model(anchor_img, pos_img, neg_img)
    
    # Tính Loss
    loss = criterion(p_a, p_p, p_n)
    
    # Backward
    loss.backward()
    optimizer.step()
    
    print(f"Loss: {loss.item()}")

    # 4. Sau khi train xong, lấy feature để visualize
    embedding = model.get_embedding(anchor_img[0:1]) # Lấy 1 mẫu
    print("Embedding Shape:", embedding.shape) # (1, 32)