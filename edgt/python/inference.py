import onnxruntime
import numpy as np

class InferenceWrapper:
    def __init__(self, onnx_path):
        self.session = onnxruntime.InferenceSession(onnx_path)
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, input_data):
        input_data = np.array(input_data, dtype=np.int64)
        result = self.session.run(None, {self.input_name: input_data})
        return result[0]

if __name__ == '__main__':
    wrapper = InferenceWrapper("edgt/model.onnx")
    input_data = [[101, 7592, 1010, 2026, 3899, 1012, 102]]
    output = wrapper.predict(input_data)
    print(output)
