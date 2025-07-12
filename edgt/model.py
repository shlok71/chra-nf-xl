import torch
import math
from torch import nn
from transformers import DistilBertModel, DistilBertConfig

def get_alibi_biases(n_heads, seq_len):
    """
    Returns a tensor of shape (n_heads, seq_len, seq_len) with ALiBi biases.
    """
    def get_slopes(n):
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]

        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2**math.floor(math.log2(n))
            return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2]

    slopes = torch.Tensor(get_slopes(n_heads))
    alibi = slopes.unsqueeze(1).unsqueeze(1) * torch.arange(seq_len).unsqueeze(0).unsqueeze(0).expand(n_heads, -1, -1)
    alibi = alibi.view(n_heads, 1, seq_len)
    alibi = alibi - torch.arange(seq_len).unsqueeze(0).unsqueeze(0).expand(n_heads, -1, -1)
    alibi = alibi.abs().mul(-1)
    return alibi

from transformers.models.distilbert.modeling_distilbert import DistilBertAttention

class CustomDistilBertAttention(DistilBertAttention):
    def __init__(self, config):
        super().__init__(config)
        self.alibi_biases = None

    def forward(self, x, attn_mask=None, head_mask=None, output_attentions=False):
        """
        x: (bs, seq_len, dim)
        attn_mask: (bs, seq_len)
        """
        bs, q_len, dim = x.size()

        # q, k, v projections
        q = self.q_lin(x).view(bs, q_len, self.n_heads, self.dim // self.n_heads).permute(0, 2, 1, 3) # (bs, n_heads, q_len, head_dim)
        k = self.k_lin(x).view(bs, q_len, self.n_heads, self.dim // self.n_heads).permute(0, 2, 3, 1) # (bs, n_heads, head_dim, q_len)
        v = self.v_lin(x).view(bs, q_len, self.n_heads, self.dim // self.n_heads).permute(0, 2, 1, 3) # (bs, n_heads, q_len, head_dim)

        # Attention scores
        scores = torch.matmul(q, k) / math.sqrt(self.dim // self.n_heads) # (bs, n_heads, q_len, q_len)

        # Add ALiBi biases
        if self.alibi_biases is None or self.alibi_biases.shape[1] != q_len:
            self.alibi_biases = get_alibi_biases(self.n_heads, q_len).to(scores.device)

        scores += self.alibi_biases.unsqueeze(0)

        if attn_mask is not None:
            scores = scores + attn_mask

        attn = nn.Softmax(dim=-1)(scores) # (bs, n_heads, q_len, q_len)
        attn = self.dropout(attn) # (bs, n_heads, q_len, q_len)

        if head_mask is not None:
            attn = attn * head_mask

        context = torch.matmul(attn, v).permute(0, 2, 1, 3).contiguous().view(bs, q_len, -1) # (bs, q_len, dim)
        context = self.out_lin(context) # (bs, q_len, dim)

        if output_attentions:
            return (context, attn)
        else:
            return (context,)

from torch.quantization import QuantStub, DeQuantStub

class MoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_experts = 128
        self.k = 4
        self.rank = 128
        self.experts = nn.ModuleList([
            nn.Sequential(
                QuantStub(),
                nn.Linear(config.dim, self.rank),
                nn.ReLU(),
                nn.Linear(self.rank, config.hidden_dim),
                DeQuantStub()
            ) for _ in range(self.n_experts)
        ])
        self.expert_diagonals = nn.Parameter(torch.randn(self.n_experts, config.dim))
        self.diag_proj = nn.Linear(config.dim, config.hidden_dim)
        self.router = nn.Sequential(
            QuantStub(),
            nn.Linear(config.dim, self.n_experts),
            DeQuantStub()
        )
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x):
        # x: (bs, seq_len, dim)
        x = self.quant(x)
        bs, seq_len, dim = x.size()
        x = x.view(-1, dim) # (bs * seq_len, dim)

        # Get router logits
        router_logits = self.router(x) # (bs * seq_len, n_experts)

        # Enforce 80% activation sparsity
        threshold = torch.quantile(router_logits.abs(), 0.8)
        router_logits[router_logits.abs() < threshold] = 0

        # Get top-k experts
        top_k_logits, top_k_indices = router_logits.topk(self.k, dim=-1) # (bs * seq_len, k)

        # Softmax over top-k experts
        top_k_scores = nn.Softmax(dim=-1)(top_k_logits) # (bs * seq_len, k)

        # Flatten the top-k indices and scores
        flat_top_k_indices = top_k_indices.flatten() # (bs * seq_len * k)
        flat_top_k_scores = top_k_scores.flatten() # (bs * seq_len * k)

        # Create a combined representation of the tokens and their expert assignments
        # x_expanded: (bs * seq_len, k, dim)
        x_expanded = x.unsqueeze(1).expand(-1, self.k, -1)
        # flat_x: (bs * seq_len * k, dim)
        flat_x = x_expanded.reshape(-1, dim)

        # Create a batch index for each token
        batch_idx = torch.arange(bs * seq_len).unsqueeze(1).expand(-1, self.k).flatten().to(x.device) # (bs * seq_len * k)

        # Group tokens by expert
        # expert_inputs is a list of tensors, where each tensor contains the tokens for a specific expert
        expert_inputs = [flat_x[flat_top_k_indices == i] for i in range(self.n_experts)]
        expert_scores = [flat_top_k_scores[flat_top_k_indices == i] for i in range(self.n_experts)]

        # Process each expert in parallel
        expert_outputs = [torch.empty(0, device=x.device) for _ in range(self.n_experts)]
        for i in range(self.n_experts):
            if expert_inputs[i].shape[0] > 0:
                # Apply diagonal weights
                diag_output = expert_inputs[i] * self.expert_diagonals[i]
                diag_output = self.diag_proj(diag_output)
                # Apply low-rank factorization
                lora_output = self.experts[i](expert_inputs[i])
                # Combine outputs
                expert_output = diag_output + lora_output
                # Apply scores
                expert_output = expert_output * expert_scores[i].unsqueeze(-1)
                expert_outputs[i] = expert_output

        # Create the final output tensor
        output = torch.zeros(bs * seq_len, config.hidden_dim).to(x.device)

        # Scatter the expert outputs back to the original token positions
        for i in range(self.n_experts):
            if expert_outputs[i].shape[0] > 0:
                # Get the original indices of the tokens that were sent to this expert
                original_indices = batch_idx[flat_top_k_indices == i]
                output.index_add_(0, original_indices, expert_outputs[i])

        output = output.view(bs, seq_len, -1) # (bs, seq_len, hidden_dim)
        output = self.dequant(output)
        return output

class EDGTTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.distilbert = DistilBertModel(config)

        # Replace attention layers with custom attention
        for i in range(len(self.distilbert.transformer.layer)):
            self.distilbert.transformer.layer[i].attention = CustomDistilBertAttention(config)
            self.distilbert.transformer.layer[i].ffn = MoE(config)

    def forward(self, *args, **kwargs):
        return self.distilbert(*args, **kwargs)

    def fuse_model(self):
        for layer in self.distilbert.transformer.layer:
            for expert in layer.ffn.experts:
                torch.quantization.fuse_modules(expert, ['1', '2'], inplace=True)
            torch.quantization.fuse_modules(layer.ffn.router, ['1', '2'], inplace=True)

def get_model():
    config = DistilBertConfig.from_pretrained('distilbert-base-uncased')
    config.n_layers = 16
    model = EDGTTransformer(config)
    return model

if __name__ == '__main__':
    model = get_model()
    print(model)
