#!/usr/bin/python3

import csv
import math
import random
import time
from operator import itemgetter

import numpy as np
from sklearn import metrics


#42%

# merge training and label -> name label tfidf ...
def concentrateData(train_data, labeled_data):
    training_list = list(csv.reader(open(train_data,'r'), delimiter=','))
    labeled_list = list(csv.reader(open(labeled_data,'r'), delimiter=','))
    for line in training_list:
        for l in labeled_list:
            if l[0] == line[0]:
                line.insert(1, l[1])
                break
    random.shuffle(training_list)
    return training_list

# count the ocurrences of each lable in training data
def labelCount1(training_data): 
    labelDict = {}
    for data in training_data:
        if data[1] in labelDict.keys():
            labelDict[data[1]] = labelDict[data[1]]+1
        else:
            labelDict[data[1]] = 1
    print(sum(labelDict.values()))
    return labelDict 

# calculte P(A|C) for each tf-idf value for each row
# return a dict with label as key, and list of probabilty as value
def getLabelProb(labeled_training_list, labelDict):
    labelProbDict = {}
    for label in labelDict.keys():
        labelProbDict[label]  = np.zeros(13626)
    for data in labeled_training_list:
        if data[1] in labelDict.keys():
            if(sum([float(x) for x in data[2:]])!=0):
                for index in range(13626):
                    # filter tf-idf values smaller than 0
                    if float(data[index+2]) > 0:
                        # use tf-idf directly
                        labelProbDict[data[1]][index] += float(data[index+2])
    for label in labelDict.keys():
        labelProbDict[label] = labelProbDict[label]/labelDict[label]
    return labelProbDict

def readTestData(test_file):
    test_data = list(csv.reader(open(test_file,'r'), delimiter=','))
    return test_data

def run():
    print("Begin to pre-process data")
    concentrate_start = time.time()
    concentrated_data = concentrateData("assignment1_2017S1/training_data.csv", "assignment1_2017S1/training_labels.csv")
    test_data = readTestData("assignment1_2017S1/test_data.csv")
    labelDict = labelCount1(concentrated_data)
    concentrate_end = time.time()
    print("The concentrate data and pre-processing process takes {} seconds\n".format(concentrate_end-concentrate_start))
    print("Begin training process")
    training_start = time.time()
    labelProbDict = getLabelProb(concentrated_data, labelDict)
    training_end = time.time()
    print("The training process takes {} seconds\n".format(training_end - training_start))
    print("Begin Predict Process")
    predict_start = time.time()
    predicted_result = []
    # calculate label probability of each label
    # and choose the label with the highest probability
    # as the final result
    for data in test_data:
        probDict = {}
        for label in labelProbDict:
            prob = 0
            for index in range(13626):
                if float(data[index+1])>0:
                    if labelProbDict[label][index]!=0:
                        prob += labelProbDict[label][index]*float(data[index+1])
            probDict[label] = prob + labelDict[label]/20103
        label = max(probDict.items(), key=itemgetter(1))[0]
        predicted_result.append((data[0],label))
    predict_end = time.time()
    print("The prediction process takes {} seconds\n".format(predict_end - predict_start))
    with open("assignment1_2017S1/predicted_labels.csv", 'w') as f:
        f_csv = csv.writer(f)
        f_csv.writerows(predicted_result)

run()


# labeled_training_list = concentrateData("assignment1_2017S1/training_data.csv", "assignment1_2017S1/training_labels.csv")

# i = 0

# recall_string = ""

# for i in range(0,10):
#     random.shuffle(labeled_training_list)
#     test_data = labeled_training_list[0:2000]
#     training_data = labeled_training_list[2000:]
#     labelDict = labelCount1(training_data)
#     labelProbDict = getLabelProb(training_data, labelDict)

#     result = 0
#     result_list = []
#     data_list = []
#     for data in test_data:
#         probDict = {}
#         for label in labelProbDict.keys():
#             prob = 0
#             for index in range(13626):
#                 if float(data[index+2])>0:
#                     # print("positive: "+labelProbDict[label][index])
#                     if(labelProbDict[label][index]!=0):
#                         prob += labelProbDict[label][index]*float(data[index+2])
#                 # else:
#                 #     # print("negative: "+(1-labelProbDict[label][index]))
#                 #     #if(1-labelProbDict[label][index]!=0):
#                 #     prob *= 1-labelProbDict[label][index]
#             probDict[label] = prob + labelDict[label]/18000
#         label = max(probDict.items(), key=itemgetter(1))[0]
#         result_list.append(label)
#         data_list.append(data[1])
#         if(label == data[1]):
#             result += 1
#     print(metrics.classification_report(result_list, data_list))
#     recall_string += metrics.classification_report(result_list, data_list)+"\n"
# print("The prediction result is \n", recall_string)
