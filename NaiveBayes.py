import sys
import os
import numpy as numpy
from sklearn.naive_bayes import MultinomialNB
import random
import numpy as np
import pandas as pd
import math
import nltk
from nltk.corpus import stopwords
import string

###############################################################################
def vocabularyGen(PathPos,PathNeg, numberOfWords):
    wordCount={}
    exclude = set(string.punctuation)
    for filename in os.listdir(PathPos):
        #print("hi")
        AllFiles=[]
        currentfile = open(PathPos+filename)
        #words=currentfile.read().split()
        tokens = nltk.word_tokenize(currentfile.read())
        stop = set(stopwords.words('english'))
        text_no_stop_words_punct = [t for t in tokens if t not in stop and t not in string.punctuation]
        #print(text_no_stop_words_punct)
        #print(stripped)
        AllFiles.append(text_no_stop_words_punct)
        for x in AllFiles:
            #print("LOL")
            for y in x:
                word=""
                if(len(y)>1):
                    for char in y:
                        if char not in exclude:
                            word = word + char
                    y=word
                    y=y.lower()
                    #print("X ",x)
                    if y in wordCount:
                        wordCount[y]+=1
                    else: 
                        wordCount[y]=1
   
    
    keys=[]
    values=[]
    for x in wordCount.keys():
        keys.append(x)
    for x in wordCount.values():
        values.append(x)
    c=list(zip(keys,values))
    c.sort(key = lambda t: t[1])
    keys, values=zip(*c)
    keys=list(keys)
    values=list(values)
   
            
    #print(keys[len(keys)-numberOfWords:len(keys)], values[len(keys)-numberOfWords:len(keys)])
    return keys[len(keys)-numberOfWords:len(keys)], values[len(keys)-numberOfWords:len(keys)]
            
        

def transfer(fileDj, vocabulary, givenVocab):
    AllFiles=[]
    total_words=0
    if(len(givenVocab)==0):
        wordbank= {"love":0, "wonderful":0, "best":0, "great":0, "superb":0, "still":0, "beautiful":0, "bad":0, "worst":0, "stupid":0,
"waste":0, "boring":0, "?":0, "!":0, "UNK":0} 
    else: 
        wordbank={}
        #print("VOCAB",givenVocab, type(givenVocab))
        for x in givenVocab:
            #print(x[0])
            wordbank[x[0]]=0
        wordbank["UNK"]
    currentfile = open(fileDj)
    words=currentfile.read().split()
    #print(goodwords)
    AllFiles.append(words)
    for x in AllFiles:
        for y in x:
            y=y.lower()
            if(y=="loves" or y=="loved" or y=="loving"):
                y="love"
            if(y in wordbank):
                y=y
            else:
                y="UNK"
            wordbank[y]+=1
            total_words+=1
            #print(y)
    stuff= []
    for key, value in wordbank.items():
        #print(key, " ",value)
        stuff.append(value/total_words)
    return stuff


def loadData(Path):
    Xtrain=[]
    Ytrain=[]
    Ytest=[]
    Xtest=[]

    vocab= vocabularyGen(Path+"/training_set/pos/",Path+"/training_set/neg/",20)
    for filename in os.listdir(Path+"/training_set/pos"):
        stuff=transfer(Path+"/training_set/pos/"+filename,"","")
        Xtrain.append(stuff)
        Ytrain.append(1)
    for filename in os.listdir(Path+"/training_set/neg"):
        stuff=transfer(Path+"/training_set/neg/"+filename,"","")
        Xtrain.append(stuff)
        Ytrain.append(-1)
    for filename in os.listdir(Path+"/test_set/pos"):
        stuff=transfer(Path+"/test_set/pos/"+filename,"","")
        Xtest.append(stuff)
        Ytest.append(1)
    for filename in os.listdir(Path+"/test_set/neg"):
        stuff=transfer(Path+"/test_set/neg/"+filename,"","")
        Xtest.append(stuff)
        Ytest.append(-1)
    #c=list(zip(Xpart,Ypart))
    #random.shuffle(c)
    #Xpart, Ypart=zip(*c)
    #Xpart=list(Xpart)
    #Ypart=list(Ypart)
    #Xtest=[]
    #Ytest=[]

    #for x in range(0,140):
    #    Xtest.append(Xpart.pop(x))
    #    Ytest.append(Ypart.pop(x))
        #print(Xtest)
    #print(len(Xtest), len(Xtrain))
    #print(len(Ytest), len(Ytrain))
    
    return Xtrain,Xtest, Ytrain,Ytest


def naiveBayesMulFeature_train(Xtrain, ytrain):
    positivelist=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    negativelist=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    positivesum=0
    negativesum=0
    #print(len(Xtrain))
    counter=0
    for vec in (Xtrain):
        if(ytrain[counter]==1):
            for i in range(0,15):
                positivelist[i]+=vec[i]
                positivesum+=vec[i]
        elif(ytrain[counter]==-1):
            for i in range(0,15):
                negativelist[i]+=vec[i]
                negativesum+=vec[i]
        counter+=1
    for x in positivelist:
        x=((x+1)/(positivesum+15))#take  log instead?
    for x in negativelist:
        x=((x+1)/(negativesum+15)) #laplace smoothing of 1, 2 classes
    #print("positivelist: ",positivelist)
    #print("negativelist: ",negativelist)
    return positivelist, negativelist

#test set ?
def naiveBayesMulFeature_test(Xtest, ytest,thetaPos, thetaNeg):
    #document

    yPredict = []
    #print("ytest", ytest)
    testFrame= pd.DataFrame(ytest)
    testFrame.append(Xtest)
    for file in Xtest:
        #list of elements
        currentProbPos=0
        currentProbNeg=0
        #print("file length:",len(file))
        for word in range(0,14):#need the index instead
            #multiply probability times the thetaPos value
            currentProbPos=currentProbPos+math.log(thetaPos[word]**file[word])#need index
            currentProbNeg=currentProbNeg+math.log(thetaNeg[word]**file[word])
        if(currentProbPos>currentProbNeg):
            yPredict.append(1)
        else:
            yPredict.append(-1)
    number_correct=0
    for x in range(0,len(yPredict)-1):
        if(yPredict[x]==ytest[x]):
            number_correct+=1
    accuracy= float(number_correct) / len(ytest)
    #print("NaiveBayes Manual accuracy: ", accuracy)
            
    return yPredict, accuracy


def naiveBayesMulFeature_sk_MNBC(Xtrain, ytrain, Xtest, ytest):
    
    from sklearn.naive_bayes import MultinomialNB
    clf = MultinomialNB()
    clf.fit(Xtrain, ytrain)
    MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
    test_vals=(clf.predict(Xtest))
    num_correct=0
    #print("ytest THING:",test_vals)
    for x in range(0, len(test_vals)):
       
        if(ytest[x]==test_vals[x]):
            num_correct+=1
    accuracy = float(num_correct)/len(ytest)
    #print("SKLearn naivebayesMULFEATURE accuracy: ",accuracy)
    return Accuracy



#def naiveBayesMulFeature_testDirectOne(path,thetaPos, thetaNeg):
#   return yPredict


def naiveBayesMulFeature_testDirect(path,thetaPos, thetaNeg):
    yPredict = []

    return yPredict, Accuracy



def naiveBayesBernFeature_train(Xtrain, ytrain):
    positivelist=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    negativelist=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    positivesum=0
    negativesum=0
    #print(len(Xtrain))
    counter=0
    for vec in (Xtrain):
        if(ytrain[counter]==1):
            for i in range(0,14):
                if(vec[i]>0):
                    positivelist[i]+=1
            positivesum+=1
        elif(ytrain[counter]==-1):
            for i in range(0,14):
                if(vec[i]>0):
                    negativelist[i]+=1
            negativesum+=1
        counter+=1
    for x in positivelist:
        x=((x+1)/(positivesum+2))#take  log instead?
    for x in negativelist:
        x=((x+1)/(negativesum+2)) #laplace smoothing of 1, 2 classes
    #print("positivelist: ",positivelist)
    #print("negativelist: ",negativelist)
    return positivelist, negativelist

    
def naiveBayesBernFeature_test(Xtest, ytest, thetaPosTrue, thetaNegTrue):
    yPredict = []
    #print("ytest", ytest)
    testFrame= pd.DataFrame(ytest)
    testFrame.append(Xtest)
    for file in Xtest:
        #list of elements
        currentProbPos=0
        currentProbNeg=0
        #print("file length:",len(file))
        for word in range(0,14):#need the index instead
            #multiply probability times the thetaPos value
            if(file[word]>0):
                currentProbPos=currentProbPos+thetaPos[word]#need index
                currentProbNeg=currentProbNeg+thetaNeg[word]
            else:
                currentProbPos=currentProbPos+1-thetaPos[word]#need index
                currentProbNeg=currentProbNeg+1-thetaNeg[word]
                
        if(currentProbPos>currentProbNeg):
            yPredict.append(1)
        else:
            yPredict.append(-1)
    number_correct=0
    for x in range(0,len(yPredict)-1):
        if(yPredict[x]==ytest[x]):
            number_correct+=1
    accuracy= float(number_correct) / len(yPredict)
    #print("accuracy: bernoulli: ", accuracy)
            
    return yPredict, accuracy


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print ("Usage: python naiveBayes.py dataSetPath testSetPath")
        sys.exit()

    print ("--------------------")
    textDataSetsDirectoryFullPath = sys.argv[1]
    testFileDirectoryFullPath = sys.argv[2]


    Xtrain, Xtest,ytrain, ytest = loadData('textDataSetsDirectoryFullPath')


    thetaPos, thetaNeg = naiveBayesMulFeature_train(Xtrain, ytrain)
    print("naiveBayesMulFeature_train output: thetapos:", thetaPos," thetaneg ", thetaNeg)
   # print ("thetaPos =", thetaPos)
   # print ("thetaNeg =", thetaNeg)
   # print ("--------------------")

    yPredict, Accuracy = naiveBayesMulFeature_test(Xtest, ytest, thetaPos, thetaNeg)
   # print ("MNBC classification accuracy =", Accuracy)

    Accuracy_sk = naiveBayesMulFeature_sk_MNBC(Xtrain, ytrain, Xtest, ytest)
   #print ("Sklearn MultinomialNB accuracy =", Accuracy_sk)

 #   yPredict, Accuracy = naiveBayesMulFeature_testDirect(testFileDirectoryFullPath, thetaPos, thetaNeg)
 #   print "Directly MNBC tesing accuracy =", Accuracy
 #   print "--------------------"

    thetaPosTrue, thetaNegTrue = naiveBayesBernFeature_train(Xtrain, ytrain)
  #  print("thetaPosTrue =", thetaPosTrue)
 #   print("thetaNegTrue =", thetaNegTrue)
 #   print("--------------------")

    yPredict_bayes, Accuracy_bayes = naiveBayesBernFeature_test(Xtest, ytest, thetaPosTrue, thetaNegTrue)
  #  print ("--------------------")
    print("naiveBayesMulFeature_train output: thetapos:", thetaPos," thetaneg ", thetaNeg)
    print("naiveBayesMulFeature_test output: ypredict:", yPredict," accuracy ", Accuracy)
    print("naiveBayesBernFeature_train output: thetapos:", thetaPosTrue," thetaNegTrue ", thetaNegTrue)
    print("naiveBayesBernFeature_test output: ypredict:",yPredict_bayes," accuracy: ", Accuracy_sk)
    print("naiveBayesMulFeature_sk_MNBC output: accuracy:", Accuracy_sk)
