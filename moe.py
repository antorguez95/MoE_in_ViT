# MODIFIED from: https://gist.github.com/ruvnet/0928768dd1e4af8816e31dde0a0205d5

"""
This model integrates the MoE concept within a Transformer architecture. Each token's
representation is processed by a subset of experts, determined by the gating mechanism.
This architecture allows for efficient and specialized handling of different aspects of the
data, aiming for the adaptability and efficiency noted in the Mixtral 8x7B model's design
philosophy. The model activates only a fraction of the available experts for each token,
significantly reducing the computational resources needed compared to activating all experts
for all tokens.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the Expert class
class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Expert, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# Define the Gating Network class
class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts):
        super(GatingNetwork, self).__init__()
        self.gate = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        return F.softmax(self.gate(x), dim=2)

# Define the Mixture of Experts Layer class
class MoELayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_experts):
        super(MoELayer, self).__init__()
        self.experts = nn.ModuleList([Expert(input_dim, hidden_dim, output_dim) for _ in range(num_experts)])
        self.gate = GatingNetwork(input_dim, num_experts)

    def forward(self, x, num_experts_per_tok):
        gating_scores = self.gate(x)
        topk_gating_scores, topk_indices = gating_scores.topk(num_experts_per_tok, dim=2, sorted=False)
        # Create a mask to zero out the contributions of non-topk experts
        mask = torch.zeros_like(gating_scores).scatter_(2, topk_indices, 1)
        # Use the mask to retain only the topk gating scores
        gating_scores = gating_scores * mask
        # Normalize the gating scores to sum to 1 across the selected top experts
        gating_scores = F.normalize(gating_scores, p=1, dim=2)
        
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        expert_outputs = expert_outputs.transpose(1, 2)
        output = torch.einsum('bte,bteo->bto', gating_scores, expert_outputs)
        return output