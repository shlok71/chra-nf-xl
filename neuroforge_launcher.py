import platform
import psutil
import ctypes

def get_system_specs():
    ram_gb = round(psutil.virtual_memory().total / (1024**3))
    cpu_cores = psutil.cpu_count(logical=False)
    avx2_support = False
    try:
        # This is a placeholder for checking AVX2 support.
        # A real implementation would use a more robust method.
        ctypes.cdll.LoadLibrary('libm.so.6')
        avx2_support = True
    except:
        pass
    return ram_gb, cpu_cores, avx2_support

import json

import argparse

def main():
    parser = argparse.ArgumentParser(description="NeuroForge Launcher")
    parser.add_argument("--offline", action="store_true", help="Run in offline mode.")
    args = parser.parse_args()

    try:
        with open('user_config.json', 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print("Configuration file not found. Please run neuroforge_configurator.py first.")
        return

    config['offline'] = args.offline

    print("Loading NeuroForge with the following configuration:")
    print(json.dumps(config, indent=4))

    # This is a placeholder for initializing NeuroForge with the loaded configuration.

if __name__ == "__main__":
    main()
