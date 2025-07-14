import pyttsx3

def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def main():
    speak("Hello, world!")

if __name__ == "__main__":
    main()
