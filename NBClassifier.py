#!/usr/bin/python3

import csv
from operator import itemgetter

import numpy as np
import math
import random


def concentrateData(train_data, labeled_data):
    training_list = list(csv.reader(open(train_data,'r'), delimiter=','))
    labeled_list = list(csv.reader(open(labeled_data,'r'), delimiter=','))
    for line in training_list:
        for l in labeled_list:
            if l[0] == line[0]:
                line.insert(1, l[1])
                break
    return training_list

# def labelCount(labeled_data):
#     labeled_list = list(csv.reader(open(labeled_data,'r'), delimiter=','))
#     labelDict = {}
#     for data in labeled_list:
#         # print(data[1])
#         if data[1] in labelDict.keys():
#             labelDict[data[1]] = labelDict[data[1]]+1
#         else:
#             labelDict[data[1]] = 1
#     return labelDict

def labelCount1(training_data):
    labelDict = {}
    for data in training_data:
        if data[1] in labelDict.keys():
            labelDict[data[1]] = labelDict[data[1]]+1
        else:
            labelDict[data[1]] = 1
    return labelDict 

# have some problem here
def getLabelProb(labeled_training_list, labelDict):
    labelProbDict = {}
    for label in labelDict.keys():
        labelProbDict[label]  = np.zeros(13626)
    for data in labeled_training_list:
        if data[1] in labelDict.keys():
            if(sum([float(x) for x in data[2:]])!=0):
                for index in range(13626):
                    if float(data[index+2]) > 0:
                        labelProbDict[data[1]][index] += float(data[index+2]) #change lable to data[1]
    for label in labelDict.keys():
        labelProbDict[label] = labelProbDict[label]/sum(labelProbDict[label])
    return labelProbDict

def readTestData(test_file):
    test_data = list(csv.reader(open(test_file,'r'), delimiter=','))
    return test_data

def run():
    pass

labeled_training_list = concentrateData("assignment1_2017S1/training_data.csv", "assignment1_2017S1/training_labels.csv")




# i = 0
# test_data = []
# for data in labeled_training_list:
#     test_data.append(data)
#     i += 1
#     if(i==2000):
#         break

# i = 0
# training_data = []
# for data in labeled_training_list:
#     if(i>3000):
#         training_data.append(data)
#     i += 1
i = 0

for i in range(0,10):
    random.shuffle(labeled_training_list)
    test_data = labeled_training_list[0:2001]
    training_data = labeled_training_list[2001:]
    labelDict = labelCount1(training_data)
    labelProbDict = getLabelProb(training_data, labelDict)

    result = 0
    for data in test_data:
        probDict = {}
        for label in labelProbDict.keys():
            prob = 1.0
            for index in range(13626):
                if float(data[index+2])>0:
                    # print("positive: "+labelProbDict[label][index])
                    if(labelProbDict[label][index]!=0):
                        prob += math.log(labelProbDict[label][index])
                else:
                    # print("negative: "+(1-labelProbDict[label][index]))
                    if(1-labelProbDict[label][index]!=0):
                        prob += math.log(1-labelProbDict[label][index])
            probDict[label] = prob*(labelDict[label]/18104)
        label = max(probDict.items(), key=itemgetter(1))[0]
        if(label == data[1]):
            result += 1
    print(result)


