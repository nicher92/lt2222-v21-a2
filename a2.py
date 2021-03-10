import sys
import os
import numpy as np
import numpy.random as npr
import pandas as pd
import random
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('wordnet')
from collections import deque 
from itertools import chain
import matplotlib.pyplot as pyplot


# Module file with functions that you fill in so that they can be
# called by the notebook.  This file should be in the same
# directory/folder as the notebook when you test with the notebook.

# You can modify anything here, add "helper" functions, more imports,
# and so on, as long as the notebook runs and produces a meaningful
# result (though not necessarily a good classifier).  Document your
# code with reasonable comments.

# Function for Part 1
def preprocess(inputfile):
    text = inputfile.readlines()
    lemmatizer = WordNetLemmatizer()

    listoflines = []
    for line in text:
        row = nltk.word_tokenize(line)
        listoflines.append(row)
    
    
      #punctuation, not including dots 
    punctuation = ("`", '?', '``', "？", '﹖', "'", '"', ",", "!", ";", "(", ")", "-", ";", ":", ">", "<", "@", "?", ":", '*', "-", "´", "`")
    
    
    listofwords = []
    for item in listoflines:
        looplist = []
        for word in item:
            word = word.lower()
            word = lemmatizer.lemmatize(word)
            looplist.append(word)
        
        if item[2] not in punctuation:
            listofwords.append(looplist)
    
    return listofwords

# Code for part 2
class Instance:
    def __init__(self, neclass, features):
        self.neclass = neclass
        self.features = features

    def __str__(self):
        return "Class: {} Features: {}".format(self.neclass, self.features)

    def __repr__(self):
        return str(self)

def create_instances(data):
    instances = []
    wordlist = [x[2] for x in data]
    before = []
    after = []
    
    
    before_words = [[a,b,c,d,e] for a,b,c,d,e in zip(wordlist,wordlist[1:],wordlist[2:],wordlist[3:],wordlist[4:])]
    list_from_five = [[a,b,c] for a,b,c in zip(data[5:], data[6:], data[7:])]
    after_words = [[a,b,c,d,e,f,g] for a,b,c,d,e,f,g in zip(wordlist[6:],wordlist[7:],wordlist[8:],wordlist[9:],wordlist[10:],wordlist[11:],wordlist[12:])]
    
    zipper = zip(before_words, list_from_five, after_words)
    
    startsymbols = [ "<S1>", "<S2>", "<S3>", "<S4>", "<S5>"]
    endsymbols = ["</S1>", "</S2>", "</S3>", "</S4>", "</S5>"]
    
    
    for before, middle, after in zipper:
 
        if middle[0][4].startswith("b"):
            if middle[1][4].startswith("i"):
                after = after[1:]
            elif not middle[1][4].startswith("i"):
                after = after[:-1]
            
            if middle[2][4].startswith("i"):
                after = after[1:]  
            elif not middle[2][4].startswith("i"):
                after = after[:-1]
            
            if "." in before:
                before = before[before.index(".")+1:]
            if "." in after:
                after = after[:after.index(".")]
            for i in range(len(endsymbols)):
                if len(after) < 5:
                    after.insert(len(after), endsymbols[i])
            for i in range(len(endsymbols)):
                if len(before) < 5:
                    before.insert(i, startsymbols[i]) 
            
            allitems = []
            neclass = middle[0][4][2:]
            allitems.extend(before)
            allitems.extend(after)
            features = allitems
            instances.append(Instance(neclass, features))
    
    
    return instances


def reduce(matrix, dims=300):
    from sklearn.decomposition import TruncatedSVD
    sentence_names = matrix["classname"]
    matrix = matrix.drop(labels="classname", axis=1)
    
    tsvd = TruncatedSVD(n_components=dims)
    matrix = tsvd.fit_transform(matrix)
    
    dataframe = pd.DataFrame.from_records(matrix)
    dataframe.insert(0,"classname",sentence_names)
    return dataframe




# Code for part 3
def create_table(instances):
    
    wordlist = {}
    for eachlist in instances:
        for word in eachlist.features:
            if word not in wordlist:
                wordlist[word] = 1
            wordlist[word] += 1
    
    topwords_to_columns = sorted(wordlist, key=wordlist.get, reverse=True)
    
    listofsents = []
    listofclasses = []
    for eachlist in instances:
            one_sentence = {}
            for word in wordlist.keys():
                if word not in eachlist.features:
                    one_sentence[word] = 0
                else:
                    wordcount = eachlist.features.count(word)
                    one_sentence[word] = wordcount
            listofsents.append(one_sentence)
            listofclasses.append(eachlist.neclass)
    
    
    
    df = pd.DataFrame(data = listofsents, columns=topwords_to_columns)
    df.insert(0,"classname",listofclasses)
    
    
    df = reduce(df, dims=500)
    
    return df


def ttsplit(bigdf):
    
    dfgroups = bigdf.groupby("classname", group_keys=False)
    df_test = dfgroups.apply(lambda x: x.sample(frac=0.2))
    df_train = bigdf.drop(df_test.index, axis=0)
    
   # df_train = df_train.reset_index(drop=True)
   # df_test = df_test.reset_index(drop=True)

    
    return (df_train.drop('classname', axis=1).to_numpy(), 
           df_train['classname'], 
           df_test.drop('classname', axis=1).to_numpy(),
           df_test['classname'])
    

# Code for part 5
def confusion_matrix(truth, predictions):
    from sklearn.metrics import confusion_matrix
    
    cols = sorted(truth.unique())
    conmat =  confusion_matrix(truth, predictions, labels=cols)
    df = pd.DataFrame(conmat, index=cols, columns=cols)
    
    return df




#this function does the same thing as create_instances except for adding pos-tags to each word
def create_instances2(data):
    instances = []
    wordlist = [str(x[2]) + "-" + str(x[3]) for x in data]
    before = []
    after = []
    
    poslist = [x[3] for x in data]
    
    before_pos = [[a,b,c,d,e] for a,b,c,d,e in zip(poslist,poslist[1:],poslist[2:],poslist[3:],poslist[4:])]
    after_pos = [[a,b,c,d,e,f,g] for a,b,c,d,e,f,g in zip(poslist[6:],poslist[7:],poslist[8:],poslist[9:],poslist[10:],poslist[11:],poslist[12:])]
    
    before_words = [[a,b,c,d,e] for a,b,c,d,e in zip(wordlist,wordlist[1:],wordlist[2:],wordlist[3:],wordlist[4:])]
    list_from_five = [[a,b,c] for a,b,c in zip(data[5:], data[6:], data[7:])]
    after_words = [[a,b,c,d,e,f,g] for a,b,c,d,e,f,g in zip(wordlist[6:],wordlist[7:],wordlist[8:],wordlist[9:],wordlist[10:],wordlist[11:],wordlist[12:])]
    
    
    
    zipper = zip(before_words, list_from_five, after_words)
    
    startsymbols = [ "<S1>", "<S2>", "<S3>", "<S4>", "<S5>"]
    endsymbols = ["</S1>", "</S2>", "</S3>", "</S4>", "</S5>"]
    
    
    for before, middle, after in zipper:
         
        if middle[0][4].startswith("b"):
            if middle[1][4].startswith("i"):
                after = after[1:]
            elif not middle[1][4].startswith("i"):
                after = after[:-1]
            
            if middle[2][4].startswith("i"):
                after = after[1:]
            elif not middle[2][4].startswith("i"):
                after = after[:-1]
            
            if '.-.' in before:
                before = before[before.index('.-.')+1:]
            if '.-.' in after:
                after = after[:after.index('.-.')]
                
            
            
            for i in range(len(endsymbols)):
                if len(after) < 5:
                    after.insert(len(after), endsymbols[i])
            for i in range(len(endsymbols)):
                if len(before) < 5:
                    before.insert(i, startsymbols[i])
            
            
            allitems = []
            neclass = middle[0][4][2:]
            allitems.extend(before)
            allitems.extend(after)
            features = allitems
            instances.append(Instance(neclass, features))
    
    
    return instances


def bonusb(filename):
    from sklearn.svm import LinearSVC
    filename = open(filename, "r")
    file = preprocess(filename)
    filename.close()
    instances = create_instances2(file)
    bigdf = create_table(instances)
    train_X, train_y, test_X, test_y = ttsplit(bigdf)
    model = LinearSVC()
    model.fit(train_X, train_y)
    train_predictions = model.predict(train_X)
    test_predictions = model.predict(test_X)
    conmat = confusion_matrix(test_y, test_predictions)
    conmat2 = confusion_matrix(train_y, train_predictions)
    return print("test", conmat),print("train", conmat2)

