from sklearn import svm, metrics
import settings
from main import batch_load, map_y
from emoji import Emoji, EmojiOutputFormat
import numpy as np
import random
import os
import json
from tqdm import tqdm

# Read the data
DATA_DIR = os.environ["DATA_DIR"]
X, y = batch_load(DATA_DIR, {"Facebook": "Meta", "Microsoft": "Microsoft", "Apple": "Apple"}, output_size=(36,36), output_format=EmojiOutputFormat.grayscale)
# Transform y
y = map_y(y)

# Preprocessing the input data
# Normalize input vector
X = X/ 255.0

X_new = X.copy()
y_new = y.copy()
sequence = []
random.seed(114)
while len(sequence) != X.shape[0]:
    i = random.randint(0,X.shape[0]-1)
    if i not in sequence:
        sequence.append(i)
#print(sequence)
for i_ind, i_val in enumerate(X):
    X_new[sequence[i_ind]] = i_val
    y_new[sequence[i_ind]] = y[i_ind]
X = X_new.copy()
y = y_new.copy()

portion = 0.3
num_of_train = int(X.shape[0]*(1-portion))
X_train = X[:num_of_train]
y_train = y[:num_of_train]
X_test = X[num_of_train:]
y_test = y[num_of_train:]

test_para_res = []
C = [pow(10,i) for i in range(-10, 10, 1)]
Gamma = [pow(10,i) for i in range(-10, 10, 1)]

with tqdm(total=len(C)*len(Gamma)) as bar:
    for c in C:
        for gamma in Gamma:
            res = {}
            model = svm.SVC(cache_size=2000, C=c, gamma=gamma)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            res["C"] = c
            res["gamma"] = gamma
            res["accuracy"] = metrics.accuracy_score(y_test, y_pred)
            res["confusion matrix"] = metrics.confusion_matrix(y_test,y_pred)
            test_para_res.append(res)
            bar.update()

file = open(os.path.join(DATA_DIR, "SVM_parameter_and_result.json"), "w", encoding="utf-8")
json.dump(test_para_res, file, indent=4)
file.close()