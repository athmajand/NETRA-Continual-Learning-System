import torch
import torch.nn as nn

class FusionThreatModel(nn.Module):
    def __init__(self, num_classes=31):
        super(FusionThreatModel, self).__init__()
        
        # =================================================================
        # NETRA Mk-I: TEMPORAL LSTM (The Old Architecture)
        # Hidden size is exactly 128 based on checkpoint
        # =================================================================
        self.lstm = nn.LSTM(
            input_size=132, 
            hidden_size=128, 
            num_layers=2, 
            batch_first=True
        )
        
        # =================================================================
        # SPATIAL STREAM (ResNet50 Context)
        # Output is 128 to match LSTM
        # =================================================================
        self.spatial_fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        # =================================================================
        # TWO-HEADED BRAIN (Threat Detection + Fine Classification)
        # Fusion is concatenating 128 (LSTM) + 128 (Spatial) = 256
        # =================================================================
        self.binary_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
        
        self.fine_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, skel, spat):
        # --- 1. LSTM Kinematic Processing ---
        lstm_out, _ = self.lstm(skel)
        kinematic_feat = lstm_out[:, -1, :]  # Extract the final hidden state
        
        # --- 2. Spatial Context Processing ---
        spatial_feat = self.spatial_fc(spat)
        
        # --- 3. Feature Fusion (Direct Concatenation) ---
        fused_vector = torch.cat((kinematic_feat, spatial_feat), dim=1) 
        
        # --- 4. Dual Classification ---
        out_binary = self.binary_head(fused_vector)
        out_fine = self.fine_head(fused_vector)
        
        return out_binary, out_fine
    