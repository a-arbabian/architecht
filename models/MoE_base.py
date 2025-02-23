import torch
import torch.nn as nn
import torch.nn.functional as F

class MoE(nn.Module):
    def __init__(self, num_experts, input_dim, output_dim):
        super().__init__()
        self.gate = nn.Linear(in_features=input_dim, 
                              out_features=num_experts)
        
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features=input_dim, out_features=input_dim*2),
                nn.ReLU(),
                nn.Linear(in_features=input_dim*2, out_features=output_dim)
            ) for _ in range(num_experts)
        ])
        
        
    def forward(self, x):
        # Shape: [batch_size, num_experts]
        gate_weights = F.softmax(self.gate(x), dim=-1) 
        # Shape: [batch_size, 1, num_experts]
        gate_weights = gate_weights.unsqueeze(1)
        
        # Shape: [batch_size, output_dim, num_experts]
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=-1)

        # Weighted sum of expert outputs by gate weights
        # Shape: [batch_size, output_dim, num_experts
        output = expert_outputs * gate_weights
        # Shape: [batch_size, output_dim]
        output = torch.sum(output, dim=-1)
        return output

if __name__ == '__main__':
    batch_size = 16
    input_dize = 16
input_dim = 512
x = torch.randn(batch_size, input_dim)
model = MoE(num_experts=5,
           input_dim=input_dim,
           output_dim=2)
y = model.forward(x)
print(y)