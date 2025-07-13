import time
from edgt.python.inference import EDGTInference
import numpy as np

def main():
    print("Running demo...")
    inference = EDGTInference("../src/edgt.onnx")

    # Generate 1000 tokens
    start_time = time.time()
    for _ in range(1000):
        dummy_input = np.random.randn(1, 1, 1024).astype(np.float32)
        dummy_mask = np.random.randn(1, 1, 128).astype(np.float32)
        output = inference.generate(dummy_input, dummy_mask)
    end_time = time.time()

    print(f"Time to generate 1000 tokens: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
