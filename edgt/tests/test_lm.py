import unittest
from edgt.python.inference import EDGTInference
import numpy as np

class TestLM(unittest.TestCase):

    def test_lm_quality(self):
        # This is a placeholder test. A real test would evaluate the
        # language modeling quality of the model on a test dataset.
        # For now, we'll just check that the output is not all zeros.
        inference = EDGTInference("../src/edgt.onnx")
        dummy_input = np.random.randn(1, 10, 1024).astype(np.float32)
        dummy_mask = np.random.randn(1, 10, 128).astype(np.float32)
        output = inference.generate(dummy_input, dummy_mask)
        self.assertFalse(np.all(output == 0))

if __name__ == "__main__":
    unittest.main()
