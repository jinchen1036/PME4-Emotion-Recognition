import os
import pickle
import numpy as np
from scipy.io.wavfile import write
from contextlib import redirect_stdout


def check_dirpath(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def save_file(filename, data, dir=None, sample_rate = None):
    if dir is not None:
        check_dirpath(dir)
    filetype = filename.split('.')[-1]
    save_path = os.path.join(dir, filename)

    if filetype == "wav":
        sample_rate = sample_rate if sample_rate is not None else 44100
        write(save_path,sample_rate, data.astype(np.int16))
        return save_path
    elif filetype == "npy":
        np.save(save_path, data)
    return None

def save_pickle_file(filename, data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def load_pickle_file(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def save_model_summary(main_path, ModelName,model):
    print("Save Model: " + ModelName + "_structure.txt")
    with open(os.path.join(main_path, ModelName + "_Structure.txt"), "w+") as f:
        with redirect_stdout(f):
            model.summary()


def save_cv_indexes(cv_i, train_index, test_index):
    cv_indexes_path = os.path.join(os.getenv("PME4_DATA_PATH"), "cv_indexes")
    check_dirpath(cv_indexes_path)
    np.save(os.path.join(cv_indexes_path, f"cv{cv_i}_train_indexes.npy"),train_index)
    np.save(os.path.join(cv_indexes_path, f"cv{cv_i}_test_indexes.npy"), test_index)