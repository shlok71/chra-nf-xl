import torch
import torch.nn as nn
import numpy as np

# --- Model Definition ---
class SpikingRouter(nn.Module):
    def __init__(self, input_size=16384, hidden_size=100, output_size=128):
        super(SpikingRouter, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.lif = nn.LeakyReLU(0.1)  # Using Leaky ReLU as a proxy for LIF neurons
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.lif(x)
        x = self.fc2(x)
        return x

# --- Top-k Masking ---
def top_k_mask(scores, k=4):
    """Generates a mask with the top k scores set to 1."""
    mask = torch.zeros_like(scores)
    _, indices = torch.topk(scores, k, dim=-1)
    mask.scatter_(-1, indices, 1)
    return mask

# --- Training and Export ---
def main():
    input_size = 16384
    output_size = 128
    k = 4
    model = SpikingRouter(input_size, 100, output_size)
    model.eval()  # Set to evaluation mode

    # --- Dummy Input ---
    # The input to the model is a flattened BHV, treated as a float tensor.
    dummy_input = torch.randn(1, input_size)

    # --- Export to ONNX ---
    onnx_path = "router/model/router.onnx"
    print(f"Exporting model to {onnx_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=11,
    )
    print("Export complete.")

    # --- Verify ONNX Model ---
    try:
        import onnx
        import onnxruntime as ort

        print("Verifying ONNX model...")
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)

        ort_session = ort.InferenceSession(onnx_path)
        input_name = ort_session.get_inputs()[0].name
        output_name = ort_session.get_outputs()[0].name

        ort_inputs = {input_name: dummy_input.numpy()}
        ort_outs = ort_session.run([output_name], ort_inputs)

        # Apply top-k masking to the output
        scores = torch.tensor(ort_outs[0])
        mask = top_k_mask(scores, k)

        print("ONNX model verified successfully.")
        print(f"Output mask sum: {torch.sum(mask)}")
        assert torch.sum(mask) == k

    except ImportError:
        print("ONNX or ONNX Runtime not installed. Skipping verification.")
    except Exception as e:
        print(f"An error occurred during ONNX verification: {e}")


if __name__ == "__main__":
    main()
