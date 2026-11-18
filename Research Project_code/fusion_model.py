import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=150):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

class FusionThreatModel(nn.Module):
    def __init__(self, num_classes=32):
        super(FusionThreatModel, self).__init__()
        self.d_model = 128
        self.skel_embedding = nn.Linear(132, self.d_model)
        self.pos_encoder = PositionalEncoding(self.d_model, max_len=150)
        encoder_layers = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=4, dim_feedforward=256, batch_first=True, dropout=0.3)
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers=2)
        self.spatial_fc = nn.Sequential(nn.Linear(2048, 128), nn.ReLU(), nn.Dropout(0.3))
        self.fusion_fc = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3))
        self.watchdog_head = nn.Linear(128, 2)
        self.analyst_head = nn.Linear(128, num_classes)

    def forward(self, skel, spat):
        x = self.skel_embedding(skel)
        x = self.pos_encoder(x)
        transformer_out = self.transformer(x)
        kinematic_feat, _ = torch.max(transformer_out, dim=1)
        spatial_feat = self.spatial_fc(spat)
        fused_vector = torch.cat((kinematic_feat, spatial_feat), dim=1)
        shared_features = self.fusion_fc(fused_vector)
        out_binary = self.watchdog_head(shared_features)
        out_fine = self.analyst_head(shared_features)
        return out_binary, out_fine