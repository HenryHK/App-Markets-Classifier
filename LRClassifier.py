import csv
import math
import random
import time
from operator import itemgetter

import numpy as np

def concentrateData(train_data, labeled_data):
    training_list = list(csv.reader(open(train_data,'r'), delimiter=','))
    labeled_list = list(csv.reader(open(labeled_data,'r'), delimiter=','))
    for line in training_list:
        for l in labeled_list:
            if l[0] == line[0]:
                line.insert(1, l[1])
                break
    random.shuffle(training_list)
    return training_list[0:2000]

def labelCount(training_data):
    labelDict = {}
    for data in training_data:
        if data[1] in labelDict.keys():
            labelDict[data[1]] = labelDict[data[1]]+1
        else:
            labelDict[data[1]] = 1
    return labelDict 

def sigmoid(x):
    return 1.0/(1+np.exp(-x))

def loss():
    pass
    

def train(training_data, label_data, maxItr=100, alpha=0.5, method='gd'):
    start_time = time.time()

    # construct init weights = [1...]T for all label 13626 * 1
    weightsDict = {}
    for label in label_data.keys():
        weightsDict[label] = np.ones((13627,1))
    for label in label_data.keys():
        tfidf_mat = [] #tfidf matrix
        label_mat = [] #label matrix, for a specific label, 1/0 for yes/no
        for line in training_data:
            if line[1]==label:
                tfidf_mat.append([1.0] + [float(x) for x in line[2:]])
                label_mat.append(1.0)
            else:
                tfidf_mat.append([1.0] + [float(x) for x in line[2:]])
                label_mat.append(0.0)
        for i in range(maxItr):
            if method=='gd':
                tfidf_mat = np.mat(tfidf_mat) # 20103 * 13627
                label_mat = np.mat(label_mat).transpose() # 20103 * 1
                output = sigmoid(tfidf_mat * weightsDict[label]) # 20103 * 1
                error = label_mat - output # 20103 * 1
                weightsDict[label] = weightsDict[label] + alpha*tfidf_mat.transpose()*error
                print("iteration finished")

    end_time = time.time()
    print("The logistic regression takes " + str(end_time - start_time) + "!")
    return weightsDict

def test(weightsDict, test_data):
    # tfidf_mat = []
    correct_count = 0
    for line in test_data:
        result = {}
        tfidf = np.mat([1] + [float(x) for x in line[2:]])
        for label in weightsDict.keys():
            result[label] = sigmoid(tfidf*weightsDict[label])[0,0]
        label = max(result.items(), key=itemgetter(1))[0]
        if label == line[1]:
            correct_count += 1
    return correct_count



def main():
    labeled_training_list = concentrateData("assignment1_2017S1/training_data.csv", "assignment1_2017S1/training_labels.csv")
    
    # split training data
    for i in range(0,10):
        random.shuffle(labeled_training_list)
        test_data = labeled_training_list[0:200]
        training_data = labeled_training_list[200:]
        label_data = labelCount(training_data)

        weightsDict = train(training_data, label_data)
        result = test(weightsDict, test_data)
        print(result)

main()