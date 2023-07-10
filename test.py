import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads):
        super(SelfAttention, self).__init__()
        self.attention = nn.MultiheadAttention(input_dim, num_heads)
        self.fc = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        x, attn_output_weights = self.attention(x, x, x)
        x = self.fc(x)

        return x, attn_output_weights

input_dim = 512
hidden_dim = 64
num_heads = 8

self_attention = SelfAttention(input_dim, hidden_dim, num_heads)

x = torch.randn(10, 20, input_dim)
output, attn_output_weights = self_attention(x)

target = torch.randn(10, 20, hidden_dim)

loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(output, target)

optimizer = torch.optim.Adam(self_attention.parameters(), lr=0.001)

optimizer.zero_grad()
loss.backward()
optimizer.step()
