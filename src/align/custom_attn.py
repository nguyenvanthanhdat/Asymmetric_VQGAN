import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomSelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(CustomSelfAttention, self).__init__()
        self.input_dim = input_dim
        self.query = nn.Linear(input_dim, input_dim) # [Batch, Seq, Dim]
        self.key = nn.Linear(input_dim, input_dim) # [Batch, Seq, Dim]
        self.value = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=2)
        
    def forward(self, x): # x.shape = [Batch, Seq, Dim]
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        scores = torch.bmm(q, k.transpose(1, 2)) / (self.input_dim ** 0.5)
        attention = self.softmax(scores)
        weighted = torch.bmm(attention, v)
        return weighted