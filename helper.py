import settings
import os
from emoji import *
import numpy as np
from typing import Union
import logging
from cross_validation import CrossValidation

logger = logging.getLogger(__name__)
# Helper
def _load(emoji: Emoji, output_size: tuple) -> np.array:
    # Flatten
    sample, name = next(emoji)
    sample = sample.flatten()
    res = np.zeros(shape=(emoji.count(),sample.shape[0]))
    res[0] = sample

    index = 1
    for i, name in emoji:
        res[index] = i.flatten()
        index += 1
    return res

# Subprocess
def batch_load(data_dir: str, company_folder_name: dict, output_size: tuple[int, int], output_format=EmojiOutputFormat.RGB):
    res_X_size = 0
    res_X = np.zeros(shape=0)
    res_y = []

    for key, val in company_folder_name.items():
        # Iterate Emoji folder
        data_folder = os.path.join(data_dir, val)
        em = Emoji(data_folder, company_name=key, output_size=output_size, output_format=output_format)
        X = _load(em, output_size)
        y = [key] * X.shape[0]

        # Update X
        res_X_temp = np.zeros(shape=(res_X_size + X.shape[0], X.shape[1]))
        for i_ind, i_val in enumerate(res_X):
            if res_X_size == 0:
                break
            res_X_temp[i_ind] = i_val
        for i_ind, i_val in enumerate(X):
            res_X_temp[res_X_size+i_ind] = i_val
        res_X = res_X_temp
        res_X_size += X.shape[0]

        # Update y
        res_y.extend(y)

    return res_X, res_y

def map_y(y) -> np.array:
    y_map = {
        "Microsoft": 0,
        "Apple": 1,
        "Facebook": 2
    }
    y = [y_map[i] for i in y]
    return np.array(y)

def grayscale_and_resize_load():
    # Read the data
    DATA_DIR = os.environ["DATA_DIR"]
    X, y = batch_load(DATA_DIR, {"Facebook": "Meta", "Microsoft": "Microsoft", "Apple": "Apple"}, output_size=(36,36), output_format=EmojiOutputFormat.grayscale)
    # Transform y
    y = map_y(y)
    
    # Preprocessing the input data
    # Normalize input vector
    X = X/ 255.0
    return X,y

def grayscale_and_resize_load_small():
    # Read the data
    DATA_DIR = os.environ["DATA_DIR"]
    X, y = batch_load(DATA_DIR, {"Facebook": "Meta", "Microsoft": "Microsoft", "Apple": "Apple"}, output_size=(18,18), output_format=EmojiOutputFormat.grayscale)
    # Transform y
    y = map_y(y)
    
    # Preprocessing the input data
    # Normalize input vector
    X = X/ 255.0
    return X,y

def grayscale_enlarge_and_shrink():
    # Read the data
    DATA_DIR = os.environ["DATA_DIR"]
    X, y = batch_load(DATA_DIR, {"Facebook": "Meta", "Microsoft": "Microsoft", "Apple": "Apple"}, output_size=(288,288), output_format=EmojiOutputFormat.grayscale)
    # Transform y
    y = map_y(y)
    
    # Preprocessing the input data
    # Normalize input vector
    X = X/ 255.0

    pca = decomposition.PCA(36*36)
    X = pca.fit_transform(X)
    return X,y

def RGB_and_PCA_load():
    # Read the data
    DATA_DIR = os.environ["DATA_DIR"]
    X, y = batch_load(DATA_DIR, {"Facebook": "Meta", "Microsoft": "Microsoft", "Apple": "Apple"}, output_size=(72,72), output_format=EmojiOutputFormat.RGB)
    # Transform y
    y = map_y(y)
    
    # Preprocessing the input data
    # Normalize input vector
    X = X/ 255.0

    pca = decomposition.PCA(36*36)
    X = pca.fit_transform(X)
    return X,y

def RGBA_enlarge_and_shrink():
    # Read the data
    DATA_DIR = os.environ["DATA_DIR"]
    X, y = batch_load(DATA_DIR, {"Facebook": "Meta", "Microsoft": "Microsoft", "Apple": "Apple"}, output_size=(144,144), output_format=EmojiOutputFormat.RGBA)
    # Transform y
    y = map_y(y)
    
    # Preprocessing the input data
    # Normalize input vector
    X = X/ 255.0

    pca = decomposition.PCA(36*36)
    X = pca.fit_transform(X)
    return X,y

def RGBA_enlarge_and_shrink_large():
    # Read the data
    DATA_DIR = os.environ["DATA_DIR"]
    X, y = batch_load(DATA_DIR, {"Facebook": "Meta", "Microsoft": "Microsoft", "Apple": "Apple"}, output_size=(216,216), output_format=EmojiOutputFormat.RGBA)
    # Transform y
    y = map_y(y)
    
    # Preprocessing the input data
    # Normalize input vector
    X = X/ 255.0

    pca = decomposition.PCA(36*36)
    X = pca.fit_transform(X)
    return X,y

# Follow code only is an example of how to use those function
"""
# Main
def main():
    # Environ attributes
    DATA_DIR = os.environ["DATA_DIR"]
    
    # Read the data
    X, y = batch_load(DATA_DIR, {"Facebook": "Meta", "Microsoft": "Microsoft"}, output_size=(72,72))

    # Transform y
    y = map_y(y)
    
    # Cross validation
    cv_X = CrossValidation(X)
    cv_y = CrossValidation(y)

    # NN
    

if __name__ == "__main__":
    main()
"""