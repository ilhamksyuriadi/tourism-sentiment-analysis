# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 17:02:26 2019

@author: ilhamksyuriadi
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
    for i in range(len(data)):
        bigramPerData = []
        for j in range(len(data[i])-1):
            temp = data[i][j] + " " + data[i][j+1]
            bigramPerData.append(temp)
        bigramData.append(bigramPerData)
    return bigramData

#create data to trigram model
def modelTrigram(data):
    trigramData = []
    for i in range(len(data)):
        bigramPerData = []
        for j in range(len(data[i])-2):
            temp = data[i][j] + " " + data[i][j+1] + " " + data[i][j+2]
            bigramPerData.append(temp)
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
    

listObjek,objek,ulasan,label = loadData('clean_dataframe.csv')
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

#clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
clf = svm.SVC(kernel='linear')

predictU = cross_val_predict(clf,tfidfU,label,cv=5)
predictU = predictU.tolist()
accU = accuracy(label,predictU)
precisionU,recallU,fMeasureU,tnU,fpU,fnU,tpU = confusionMatrics(label,predictU)

predictB = cross_val_predict(clf,tfidfB,label,cv=5)
predictB = predictB.tolist()
accB = accuracy(label,predictB)
precisionB,recallB,fMeasureB,tnB,fpB,fnB,tpB = confusionMatrics(label,predictB)

predictT = cross_val_predict(clf,tfidfT,label,cv=5)
predictT = predictT.tolist()
accT = accuracy(label,predictT)
precisionT,recallT,fMeasureT,tnT,fpT,fnT,tpT = confusionMatrics(label,predictT)

actualCompare = comparison(listObjek,objek,label)
predictUCompare = comparison(listObjek,objek,predictU)
predictBCompare = comparison(listObjek,objek,predictB)
predictTCompare = comparison(listObjek,objek,predictT)

print("Actual compare:")
for c in actualCompare:
    print("Objek wisata:",c[0],"positive:",c[1],"negative:",c[2])
    csvName = "Actual "+c[0]+".csv"
    with open(csvName, 'w') as csvFile:
        writer = csv.writer(csvFile)
        for i in range(len(c[3])):
            writer.writerow([c[3][i]])
    csvFile.close()
print('')    
print("Unigram acc:", accU)
for c in predictUCompare:
    print("Objek wisata:",c[0],"positive:",c[1],"negative:",c[2])
    csvName = "Unigram "+c[0]+".csv"
    with open(csvName, 'w') as csvFile:
        writer = csv.writer(csvFile)
        for i in range(len(c[3])):
            writer.writerow([c[3][i]])
    csvFile.close()
print('') 
print("Bigram acc:", accB)
for c in predictBCompare:
    print("Objek wisata:",c[0],"positive:",c[1],"negative:",c[2])
    csvName = "Bigram "+c[0]+".csv"
    with open(csvName, 'w') as csvFile:
        writer = csv.writer(csvFile)
        for i in range(len(c[3])):
            writer.writerow([c[3][i]])
    csvFile.close()
print('')  
print("Trigram acc:", accT)
for c in predictTCompare:
    print("Objek wisata:",c[0],"positive:",c[1],"negative:",c[2])
    csvName = "Trigram "+c[0]+".csv"
    with open(csvName, 'w') as csvFile:
        writer = csv.writer(csvFile)
        for i in range(len(c[3])):
            writer.writerow([c[3][i]])
    csvFile.close()
    

#resultB = cross_val_score(clf,tfidfB,label,cv=5)
#predictB = cross_val_predict(clf,tfidfB,label,cv=5)
#resultT = cross_val_score(clf,tfidfT,label,cv=5)
#predictT = cross_val_predict(clf,tfidfT,label,cv=5)

    
#print(type(resultU))
#resultU = resultU.tolist()
#print(type(resultU))

#unigram = createUnigram(ulasan)
#bigram = createBigram(ulasan)
#trigram = createTrigram(ulasan)
#
#dfUni = createDf(unigram,ulasan)
#dataTfidf = createTfidf(ulasan,unigram,dfUni)
#
#clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
#result = cross_val_score(clf,dataTfidf,label,cv=10)
#print(sum(result)/len(result))




































