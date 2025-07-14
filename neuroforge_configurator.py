import json

def main():
    config = {}

    print("NeuroForge Configurator")
    print("---------------------")

    config['ram_limit'] = input("Enter RAM limit (GB): ")
    config['cpu_limit'] = input("Enter CPU core limit: ")

    print("\nSelect modules to preload (comma-separated):")
    print(" - text")
    print(" - ocr")
    print(" - canvas")
    config['preload_modules'] = input("> ").split(',')

    print("\nSelect UI mode:")
    print(" - cli")
    print(" - overlay")
    print(" - voice")
    config['ui_mode'] = input("> ")

    print("\nEnable/disable features (y/n):")
    config['web_search'] = input("Web search? ") == 'y'
    config['media_generation'] = input("Media generation? ") == 'y'
    config['agents'] = input("Agents? ") == 'y'

    with open('user_config.json', 'w') as f:
        json.dump(config, f, indent=4)

    print("\nConfiguration saved to user_config.json")

if __name__ == "__main__":
    main()
