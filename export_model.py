import torch
from training.scripts.train_vision import SimpleCNN
import os
import onnx
from onnx2keras import onnx_to_keras
import tensorflow as tf
import numpy as np

def export_to_onnx():
    model = SimpleCNN()
    model_path = os.path.join(os.getcwd(), "neuroforge_vision.pt")
    model.load_state_dict(torch.load(model_path))
    model.eval()

    dummy_input = torch.randn(1, 1, 28, 28)
    torch.onnx.export(model, dummy_input, "neuroforge_vision.onnx", opset_version=11)

def export_to_tflite():
    onnx_model = onnx.load("neuroforge_vision.onnx")
    k_model = onnx_to_keras(onnx_model, ['input.1'])

    # Manually set weights
    for layer in k_model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.Dense):
            weights = []
            for initializer in onnx_model.graph.initializer:
                if initializer.name.startswith(layer.name):
                    weights.append(np.frombuffer(initializer.raw_data, dtype=np.float32).reshape(initializer.dims))
            layer.set_weights(weights)


    converter = tf.lite.TFLiteConverter.from_keras_model(k_model)
    tflite_model = converter.convert()
    with open("neuroforge_vision.tflite", "wb") as f:
        f.write(tflite_model)


if __name__ == '__main__':
    export_to_onnx()
    export_to_tflite()
