import torch
import torch.nn as nn

from transformers import PatchTSTConfig, PatchTSTForPrediction



class PatchTSTConfig_crossAtt(nn.Module):
    def __init__(self, context_length=1600, prediction_length=400, patch_length=40, stride=20, d_model=128, 
                 num_heads=8, encoder_layers=3, revin=False, num_input_channels=12, use_cross_att=True):
        super().__init__()
        config = PatchTSTConfig(
            context_length=1600, prediction_length=400, patch_length=40, stride=20, d_model=128, 
                 num_heads=8, encoder_layers=3, revin=False, num_input_channels=12
        )

        # Load la backbone
        self.backbone = PatchTSTForPrediction(
            config
        )

        """
        # Si on est en cross 
        if use_cross_att:
            self.head = 
        else:
            self.head = 
        """

    def forward(self, x, obs_mask):
        x = self.backbone(past_values=x, past_observed_mask=obs_mask)
        return x



# Petit test
if __name__ == '__main__':
    device= "mps"
    model = PatchTSTConfig_crossAtt().eval().to(device)
    # Configuration
    batch_size = 1
    seq_len = 1600  # context_length
    num_vars = 12    # num_input_channels
    idx_reel = 1200 # Le signal s'arrête à 1200, le reste est du padding

    # 1. Créer le signal avec padding
    signal = torch.zeros(batch_size, seq_len, num_vars)
    signal[:, :idx_reel, :] = torch.randn(batch_size, idx_reel, num_vars) # Données réelles

    # 2. Créer le masque d'observation
    obs_mask = torch.zeros(batch_size, seq_len, num_vars)
    obs_mask[:, :idx_reel, :] = 1

    with torch.no_grad():
        signal = signal.to(device)
        obs_mask = obs_mask.to(device)

        outputs = model(
            signal,
            obs_mask
        )
        prediction = outputs.prediction_outputs.cpu()
        print(prediction.shape)
