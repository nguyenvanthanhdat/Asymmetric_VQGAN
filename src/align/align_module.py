from src.align.custom_attn import CustomSelfAttention
import torch.nn as nn

class AlignModel(nn.Module):
    def __init__(self, input_dim, output_dim, feature_dim):
        super(AlignModel, self).__init__()
        self.mae_output = input_dim
        self.prompt_output = output_dim
        
        self.dimmension_align = nn.Linear(input_dim, output_dim) # MAE's output -> prompt's shape
        self.attn1 = CustomSelfAttention(feature_dim)
        self.attn2 = CustomSelfAttention(feature_dim)
        self.attn3 = CustomSelfAttention(feature_dim)
        self.attn4 = CustomSelfAttention(feature_dim)
        
    def forward(self, mae_output):
        # change order of dimmension
        mae_output = mae_output.permute(0, 2, 1)
        dimm_aligned = self.dimmension_align(mae_output)
        dimm_aligned = dimm_aligned.permute(0, 2, 1)
        attn1 = self.attn1(dimm_aligned)
        attn2 = self.attn2(attn1)
        attn3 = self.attn3(attn2)
        attn4 = self.attn4(attn3)
        # attn4 = attn4.permute(0, 2, 1)
        print(attn4.shape)
        return attn4
        

