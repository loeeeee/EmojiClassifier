from sklearn import svm, metrics
import logging
import settings
from main import batch_load, map_y
from emoji import Emoji, EmojiOutputFormat
import numpy as np
import random
import os
import json
from tqdm import tqdm
from sklearn import metrics, decomposition
import multiprocessing
from multiprocessing import Lock, Process, Queue, current_process
import time
import queue # imported for using queue.Empty exception

logger = logging.getLogger(__name__)

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

logger.info("Start loading data")
# Change how you want the data loo like
# X,y = RGB_and_PCA_load()
# X,y = grayscale_and_resize_load()
X,y = grayscale_enlarge_and_shrink()
logger.info("Finish loading data")

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

C = [pow(10,i) for i in range(-10, 10, 2)]
Gamma = [pow(10,i) for i in range(-10, 10, 2)]

def do_job(tasks_to_accomplish, tasks_that_are_done, X_train, y_train, X_test):
    while True:
        try:
            '''
                try to get task from the queue. get_nowait() function will 
                raise queue.Empty exception if the queue is empty. 
                queue(False) function would do the same task also.
            '''
            task = tasks_to_accomplish.get_nowait()
        except queue.Empty:

            break
        else:
            '''
                if no exception has been raised, add the task completion 
                message to task_that_are_done queue
            '''
            c = task[0]
            gamma = task[1]

            start_time = time.time()
            print(f"SVM with c: {c}, gamma: {gamma} start")
            logger.info("Process start fitting svm")
            res = {}
            model = svm.SVC(cache_size=2000, C=c, gamma=gamma)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            print(f"SVM with c: {c}, gamma: {gamma} finish")
            print(f"Using {time.time() - start_time}")
            res["C"] = c
            res["gamma"] = gamma
            res["accuracy"] = str(metrics.accuracy_score(y_test, y_pred))
            res["confusion matrix"] = str(metrics.confusion_matrix(y_test,y_pred))

            tasks_that_are_done.put(res)
    return True

number_of_processes = int(multiprocessing.cpu_count())
tasks_to_accomplish = Queue()
tasks_that_are_done = Queue()
processes = []
for c in C:
    for gamma in Gamma:
        tasks_to_accomplish.put([c,gamma])

# creating processes
print("Create process")
logger.info("Creating process!")
for w in range(number_of_processes):
    p = Process(target=do_job, args=(tasks_to_accomplish, tasks_that_are_done, X_train, y_train, X_test))
    processes.append(p)
    p.start()
# completing process
logger.info("Waiting process!")
for p in processes:
    p.join()
# print the output
test_para_res = []
logger.info("Saving result!")
while not tasks_that_are_done.empty():
    test_para_res.append(tasks_that_are_done.get())

DATA_DIR = os.environ["DATA_DIR"]
file = open(os.path.join(DATA_DIR, "SVM_parameter_and_result.json"), "w", encoding="utf-8")
json.dump(test_para_res, file, indent=4)
file.close()