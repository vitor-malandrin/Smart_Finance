import os
import pickle

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_model(model, file_name):
    with open(file_name, 'wb') as file:
        pickle.dump(model, file)
