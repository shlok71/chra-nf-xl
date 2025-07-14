import psutil
import time

import subprocess

def run_text_generation():
    print("Running text generation smoketest...")
    start_time = time.time()
    result = subprocess.run(["./neuroforge.exe", "text", "What is the meaning of life?"], capture_output=True, text=True)
    end_time = time.time()
    print(result.stdout)
    return "The answer is 42." in result.stdout, end_time - start_time

def run_canvas_draw():
    print("Running canvas draw smoketest...")
    start_time = time.time()
    result = subprocess.run(["./neuroforge.exe", "canvas", "draw red circle at 50,50"], capture_output=True, text=True)
    end_time = time.time()
    print(result.stdout)
    return "Canvas is not yet implemented." in result.stdout, end_time - start_time

def run_ocr():
    print("Running OCR smoketest...")
    start_time = time.time()
    result = subprocess.run(["./neuroforge.exe", "ocr", "image.png"], capture_output=True, text=True)
    end_time = time.time()
    print(result.stdout)
    return "OCR BHV:" in result.stdout, end_time - start_time

def run_image_generation():
    print("Running image generation smoketest...")
    start_time = time.time()
    result = subprocess.run(["./neuroforge.exe", "image", "A cat sitting on a mat."], capture_output=True, text=True)
    end_time = time.time()
    print(result.stdout)
    return "Image generation is not yet implemented." in result.stdout, end_time - start_time

def run_voice_query():
    print("Running voice query smoketest...")
    start_time = time.time()
    result = subprocess.run(["./neuroforge.exe", "voice", "audio.wav"], capture_output=True, text=True)
    end_time = time.time()
    print(result.stdout)
    return "This is a transcribed sentence." in result.stdout, end_time - start_time

def main():
    tasks = {
        "Text Generation": run_text_generation,
        "Canvas Draw": run_canvas_draw,
        "OCR": run_ocr,
        "Image Generation": run_image_generation,
        "Voice Query": run_voice_query,
    }

    results = {}
    for task_name, task_func in tasks.items():
        cpu_before = psutil.cpu_percent()
        ram_before = psutil.virtual_memory().percent
        success, latency = task_func()
        cpu_after = psutil.cpu_percent()
        ram_after = psutil.virtual_memory().percent
        results[task_name] = {
            "success": success,
            "latency": latency,
            "cpu_usage": cpu_after - cpu_before,
            "ram_usage": ram_after - ram_before,
        }

    print("\n--- Smoketest Results ---")
    final_score = "PASS"
    for task_name, result in results.items():
        print(f"\n{task_name}:")
        print(f"  - Success: {result['success']}")
        print(f"  - Latency: {result['latency']:.2f}s")
        print(f"  - CPU Usage: {result['cpu_usage']:.2f}%")
        print(f"  - RAM Usage: {result['ram_usage']:.2f}%")
        if not result["success"] or result["latency"] > 10:
            final_score = "FAIL"
        elif result["latency"] > 5:
            final_score = "WARN"

    print(f"\nFinal Readiness Score: {final_score}")

if __name__ == "__main__":
    main()
