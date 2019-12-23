# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 17:02:26 2019

@author: arisafrianto23@gmail.com
"""

import pandas as pd
import csv
from ast import literal_eval
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn import svm
from sklearn.metrics import confusion_matrix

#load data from csv with dataframe format
def loadData(filename):
    rawdata = pd.read_csv(filename)
    listObjek = []
    objek = pd.Series.tolist(rawdata['objek'])
    label = pd.Series.tolist(rawdata['label'])
    rawulasan = pd.Series.tolist(rawdata['ulasan'])
    ulasan = []
    for u in rawulasan:
        ulasan.append(literal_eval(u))
    for o in objek:
        if o not in listObjek:
            listObjek.append(o)
    return listObjek,objek,ulasan,label

#create term
def createTerm(ulasan):
    term = []
    for i in range(len(ulasan)):
        for j in range(len(ulasan[i])):
            if ulasan[i][j] not in term:
                term.append(ulasan[i][j])
    return term

##create bigram
#def createBigram(ulasan):
#    bigram = []
#    for i in range(len(ulasan)):
#        for j in range(len(ulasan[i])-1):
#            tempBigram = ulasan[i][j] + ' ' + ulasan[i][j+1]
#            if tempBigram not in bigram:
#                bigram.append(tempBigram)
#    return bigram
#
##create trigram
#def createTrigram(ulasan):
#    trigram = []
#    for i in range(len(ulasan)):
#        for j in range(len(ulasan[i])-2):
#            tempTrigram = ulasan[i][j] + ' ' + ulasan[i][j+1] + ' ' + ulasan[i][j+2]
#            if tempTrigram not in trigram:
#                trigram.append(tempTrigram)
#    return trigram

#create data to bigram model
def modelBigram(data):
    bigramData = []
    count = 0
    for i in range(len(data)):
        bigramPerData = []
        for j in range(len(data[i])-1):
            temp = data[i][j] + " " + data[i][j+1]
            bigramPerData.append(temp)
            count += 1
            print(count,"Bigram")
        bigramData.append(bigramPerData)
    return bigramData

#create data to trigram model
def modelTrigram(data):
    trigramData = []
    count = 0
    for i in range(len(data)):
        bigramPerData = []
        for j in range(len(data[i])-2):
            temp = data[i][j] + " " + data[i][j+1] + " " + data[i][j+2]
            bigramPerData.append(temp)
            count += 1
            print(count,"Trigram")
        trigramData.append(bigramPerData)

    return trigramData

#create document frequency
def createDf(term,ulasan):
    df = {}
    deletedDf = []
    for t in term:
        for i in range(len(ulasan)):
            if t in ulasan[i]:
                if t in df:
                    df[t] += 1
                else:
                    df[t] = 1
    countTreshold = 0
    for i in term:
        if df[i] <= 0:#ubah nilai ini buat threshold
            deletedDf.append(i)
            del df[i]
            countTreshold += 1
            print(countTreshold, "treshold applied")
    return df,deletedDf

#create tf idf
def createTfidf(ulasan,term,df,deletedDf):
    dataTFIDF = []
    for i in range(len(ulasan)):
        tempTFIDF = []
        for j in range(len(term)):
            if term[j] in ulasan[i] and term[j] not in deletedDf:
                tf = 0
                for k in range(len(ulasan[i])):
                    if term[j] == ulasan[i][k]:
                        tf += 1
                idf = math.log10(len(ulasan)/df[term[j]])                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
                tempTFIDF.append(idf*tf)
            else:
                tempTFIDF.append(0)
        dataTFIDF.append(tempTFIDF)
    return dataTFIDF

#accuracy
def accuracy(label,predict):
    acc = 0
    for i in range(len(label)):
        if label[i] == predict[i]:
            acc += 1
    return round(acc/len(label),4)

#comparison between positive and negative
def comparison(listObjek,objek,label):
    compare = []
    for i in range(len(listObjek)):
        positive = 0
        negative = 0
        allLabel = []
        count = 0
        for j in range(len(objek)):
            if listObjek[i] == objek[j]:
                count += 1
                if label[j] == 1:
                    positive += 1
                elif label[j] == 0:
                    negative += 1
                allLabel.append(label[j])
        compare.append([listObjek[i],round((positive/count),4),round((negative/count),4),allLabel])
    return compare

def confusionMatrics(label,predict):#dibikin itung manual biar keliatan, aslinya bisa langsung ya
    tn, fp, fn, tp = confusion_matrix(label,predict).ravel()
    precision = tp / ( tp + fp )
    recall = tp / ( tp + fn )
    fMeasure = 2 * precision * recall / ( precision + recall )
    return precision,recall,fMeasure,tn,fp,fn,tp
    

listObjek,objek,ulasan,label = loadData('clean_data.csv')
ulasanB = modelBigram(ulasan)
ulasanT = modelTrigram(ulasan)

unigram = createTerm(ulasan)
bigram = createTerm(ulasanB)
trigram = createTerm(ulasanT)

dfU,deletedDfU = createDf(unigram,ulasan)
dfB,deletedDfB = createDf(bigram,ulasanB)
dfT,deletedDfT = createDf(trigram,ulasanT)

tfidfU = createTfidf(ulasan,unigram,dfU,deletedDfU)
tfidfB = createTfidf(ulasanB,bigram,dfB,deletedDfB)
tfidfT = createTfidf(ulasanT,trigram,dfT,deletedDfT)

clf = RandomForestClassifier(n_estimators=100, max_depth = 64, min_samples_split = 3, criterion = 'entropy',  random_state=0)
#clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
#clf = svm.SVC(kernel='linear')

clf.fit(tfidfU,label)

from sklearn.tree import export_graphviz
import pydot

tree = clf.estimators_[1]
export_graphviz(tree, out_file = 'tree.dot', feature_names = unigram, rounded = True, precision = 1)
(graph, ) = pydot.graph_from_dot_file('tree.dot')
graph.write_png('tree.png')































