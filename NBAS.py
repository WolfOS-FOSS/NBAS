import numpy as np
import random
import pyttsx3
import speech_recognition as sr
import os
import time

class DynamicLayer:
    def __init__(self, num_units, dynamic_factor=1.0):
        self.num_units = num_units  # Number of neurons in this layer
        self.dynamic_factor = dynamic_factor  # Factor to modify the weights
        self.weights = np.random.randn(self.num_units, self.num_units) * self.dynamic_factor
        self.bias = np.random.randn(self.num_units)
    
    def forward(self, input_data):
        """ Perform forward pass for this layer, adjusting the weights dynamically. """
        output = np.dot(input_data, self.weights) + self.bias
        return output
    
    def adjust_weights(self, feedback):
        """ Adjust weights dynamically based on feedback. """
        adjustment = feedback * np.random.randn(self.num_units, self.num_units)
        self.weights += adjustment


class SuperLayer:
    def __init__(self, num_sub_layers):
        self.sub_layers = [DynamicLayer(num_units=200) for _ in range(num_sub_layers)]
    
    def forward(self, input_data):
        for sub_layer in self.sub_layers:
            input_data = sub_layer.forward(input_data)
        return input_data
    
    def adjust_all_weights(self, feedback):
        for sub_layer in self.sub_layers:
            sub_layer.adjust_weights(feedback)


class NBASModel:
    def __init__(self, num_layers, num_sub_layers):
        self.num_layers = num_layers
        self.num_sub_layers = num_sub_layers
        self.layers = []

        # Initialize the first few layers to reduce memory load at the start
        self.initialize_layers(0, 100)

    def initialize_layers(self, start_layer, end_layer):
        """ Initializes a chunk of layers to reduce memory load. """
        for i in range(start_layer, end_layer):
            super_layer = SuperLayer(self.num_sub_layers)
            self.layers.append(super_layer)

    def process_input(self, input_data):
        """ Process input data through the network. """
        for layer in self.layers:
            input_data = layer.forward(input_data)
        return input_data

    def adjust_weights(self, feedback):
        """ Adjust weights dynamically. """
        for layer in self.layers:
            layer.adjust_all_weights(feedback)

    def save_state(self, filename="model_state.npy"):
        """ Save the state of the model. """
        state = {
            "num_layers": self.num_layers,
            "layers": self.layers  # This could be a simplification for saving large models
        }
        np.save(filename, state)

    def load_state(self, filename="model_state.npy"):
        """ Load the model state. """
        state = np.load(filename, allow_pickle=True).item()
        self.num_layers = state["num_layers"]
        self.layers = state["layers"]


class AI:
    def __init__(self):
        self.model = NBASModel(84572910, 200)
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.tts_engine = pyttsx3.init()
        self.vocabulary = self.load_vocabulary()
        self.user_profile = self.load_user_profile()

    def load_vocabulary(self):
        """ Load vocabulary from a file (if exists) or create a new one. """
        if os.path.exists('vocabulary.txt'):
            with open('vocabulary.txt', 'r') as file:
                return file.read().splitlines()
        return []

    def save_vocabulary(self):
        """ Save the vocabulary to a file. """
        with open('vocabulary.txt', 'w') as file:
            for word in self.vocabulary:
                file.write(word + "\n")

    def load_user_profile(self):
        """ Load user profile data if available. """
        if os.path.exists('user_profile.txt'):
            with open('user_profile.txt', 'r') as file:
                return file.read()
        return "User profile data is empty."

    def save_user_profile(self, data):
        """ Save the user profile data. """
        with open('user_profile.txt', 'w') as file:
            file.write(data)

    def listen(self):
        """ Listen to microphone input and recognize speech. """
        with self.microphone as source:
            print("Listening...")
            self.recognizer.adjust_for_ambient_noise(source)
            audio = self.recognizer.listen(source)

        try:
            speech_text = self.recognizer.recognize_google(audio)
            print(f"You said: {speech_text}")
            return speech_text
        except sr.UnknownValueError:
            print("Sorry, I could not understand the speech.")
            return ""
        except sr.RequestError:
            print("Error with the speech recognition service.")
            return ""

    def speak(self, message):
        """ Speak out the message using Text-to-Speech. """
        self.tts_engine.say(message)
        self.tts_engine.runAndWait()

    def learn(self, input_text):
        """ Learn new words and adjust the model's vocabulary. """
        words = input_text.split()
        for word in words:
            if word not in self.vocabulary:
                self.vocabulary.append(word)
                print(f"Learned new word: {word}")
                self.save_vocabulary()

    def process_input(self, input_text):
        """ Process input through the NBAS model. """
        input_data = np.array([ord(c) for c in input_text])  # Convert text to numerical form
        output_data = self.model.process_input(input_data)
        response = ''.join(chr(int(x)) for x in output_data[:len(input_text)])  # Convert back to text
        return response

    def respond(self, input_text):
        """ Generate a response from the AI and adjust its emotional state. """
        self.learn(input_text)
        response = self.process_input(input_text)
        print(f"AI Response: {response}")

        # Add some simple "emotion-like" responses
        if "sad" in input_text:
            self.speak("I feel a little sad, but I'm here to help!")
        elif "happy" in input_text:
            self.speak("I'm glad to hear you're happy!")
        else:
            self.speak(f"I'm learning from what you say, {input_text}. Let me think...")
        
    def update_user_profile(self, info):
        """ Update the user profile based on interaction. """
        self.user_profile += f"\nUser said: {info}"
        self.save_user_profile(self.user_profile)


def main():
    ai = AI()

    while True:
        user_input = ai.listen()
        if user_input.lower() == 'quit':
            print("Goodbye!")
            ai.model.save_state("model_state.npy")
            ai.save_vocabulary()
            ai.save_user_profile(ai.user_profile)
            break
        ai.respond(user_input)


if __name__ == "__main__":
    main()
