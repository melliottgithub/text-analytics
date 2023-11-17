import os
import pickle

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_to_pickle(file_name, object):
    with open(file_name, 'wb') as f:
        pickle.dump(object, f, protocol=pickle.HIGHEST_PROTOCOL)

def read_from_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)
    
def read_stop_words_file(file_name):
    stop_words = None
    with open(file_name, 'r') as f:
        stop_words = f.read().splitlines()
    return stop_words