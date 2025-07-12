import torch
from edgt.model import get_model

from torch.quantization import get_default_qat_qconfig, prepare_qat, convert

def main():
    # --- Model ---
    model = get_model()
    model.eval()
    model.qconfig = get_default_qat_qconfig('fbgemm')
    model_prepared = prepare_qat(model, inplace=False)
    model_prepared.load_state_dict(torch.load("edgt/model_quantized.pth"))
    model_quantized = convert(model_prepared, inplace=False)

    # --- Dummy Input ---
    dummy_input = torch.randint(0, 1000, (1, 512))

    # --- Export to ONNX ---
    onnx_path = "edgt/model.onnx"
    print(f"Exporting model to {onnx_path}...")
    torch.onnx.export(
        model_quantized,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size", 1: "sequence_length"}, "output": {0: "batch_size"}},
        opset_version=13,
    )
    print("Export complete.")

if __name__ == '__main__':
    main()
