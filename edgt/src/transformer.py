import torch
import torch.nn as nn

class MoE(nn.Module):
    def __init__(self, input_size, num_experts, top_k):
        super(MoE, self).__init__()
        self.experts = nn.ModuleList([nn.Linear(input_size, input_size) for _ in range(num_experts)])
        self.gate = nn.Linear(input_size, num_experts)
        self.top_k = top_k

    def forward(self, x, router_mask):
        gate_logits = self.gate(x)
        # In a real implementation, we would use the router mask to select the experts.
        # For now, we'll just use the top-k experts based on the gate logits.
        top_k_logits, top_k_indices = gate_logits.topk(self.top_k, dim=-1)

        output = torch.zeros_like(x)
        for i in range(x.size(0)):
            for j in range(x.size(1)):
                for k in range(self.top_k):
                    expert_index = top_k_indices[i, j, k]
                    output[i, j] += self.experts[expert_index](x[i, j]) * top_k_logits[i, j, k]
        return output

class TransformerLayer(nn.Module):
    def __init__(self, input_size, num_experts, top_k):
        super(TransformerLayer, self).__init__()
        self.attention = nn.MultiheadAttention(input_size, 8)
        self.moe = MoE(input_size, num_experts, top_k)
        self.norm1 = nn.LayerNorm(input_size)
        self.norm2 = nn.LayerNorm(input_size)

    def forward(self, x, router_mask):
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_output)
        moe_output = self.moe(x, router_mask)
        x = self.norm2(x + moe_output)
        return x

class EDGT(nn.Module):
    def __init__(self, num_layers, input_size, num_experts, top_k):
        super(EDGT, self).__init__()
        self.layers = nn.ModuleList([TransformerLayer(input_size, num_experts, top_k) for _ in range(num_layers)])

    def forward(self, x, router_mask):
        for layer in self.layers:
            x = layer(x, router_mask)
        return x

def main():
    model = EDGT(num_layers=2, input_size=1024, num_experts=8, top_k=2)
    dummy_input = torch.randn(1, 10, 1024)
    dummy_mask = torch.randn(1, 10, 128)
    torch.onnx.export(model, (dummy_input, dummy_mask), "edgt.onnx",
                      input_names=["input", "router_mask"], output_names=["output"])
    print("Model exported to edgt.onnx")

if __name__ == "__main__":
    main()
