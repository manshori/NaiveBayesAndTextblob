import re
import csv
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_auc_score
#from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from wordcloud import WordCloud,STOPWORDS
import matplotlib.pyplot as plt

#start processing data twitter
def processTweet(tweet):
    # process the tweets
    #Convert to lower case
    tweet = tweet.lower()
    #Convert www.* or https?://* to URL
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)
    #Convert @username to AT_USER
    tweet = re.sub('@[^\s]+','AT_USER',tweet)
    #Remove additional white spaces
    tweet = re.sub('[\s]+', ' ', tweet)
    #Replace #word with word
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    #trim
    tweet = tweet.strip('\'"')
#     print(tweet)
#     print(' ')
    return tweet
#end

#initialize stopWords
stopWords = []

#start replaceTwoOrMore
def replaceTwoOrMore(s):
    #look for 2 or more repetitions of character and replace with the character itself
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    return pattern.sub(r"\1\1", s)
#end

#start getStopWordList
def getStopWordList(stopWordListFileName):
    #read the stopwords file and build a list
    stopWords = []
    stopWords.append('AT_USER')
    stopWords.append('URL')

    fp = open(stopWordListFileName, 'r')
    line = fp.readline()
    while line:
        word = line.strip()
        stopWords.append(word)
        line = fp.readline()
    fp.close()
    return stopWords
#end

#start getfeatureVector
def getFeatureVector(tweet):
    featureVector = []
    # tweet=stemmer.stem(tweet)
    #split tweet into words
    words = tweet.split()
    for w in words:
        #replace two or more with two occurrences
        w = replaceTwoOrMore(w)
        #strip punctuation
        w = w.strip('\'"?,.')
        #check if the word stats with an alphabet
        val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", w)
        #ignore if it is a stop word
        if(w in stopWords or val is None):
            continue
        else:
           featureVector.append(w.lower())
#     print(featureVector)
#     print(' ')
    return featureVector
#end

#start replace singkatan/kata alay
def translator(tweet):
    j = 0
    for kata in tweet:
        with open('singkatan.txt', 'r') as csvfile:
            # Reading file as CSV with delimiter as "=", so that abbreviation are stored in row[0] and phrases in row[1]
            dataFromFile = csv.reader(csvfile, delimiter="=")
            for row in dataFromFile:
                # Check if selected word matches short forms[LHS] in text file.
                if kata == row[0]:
                    # If match found replace it with its appropriate phrase in text file.
                    tweet[j] = row[1]
            csvfile.close()
        j = j + 1
#     print(' '.join(tweet))
#     print('')
    return tweet    
#end process

#Read the tweets one by one and process it
df = pd.read_csv('jok.csv', index_col = False, encoding = "ISO-8859-1" )
line = df['Mentions']
st = open('konjungsi.csv', 'r')
stopWords = getStopWordList('konjungsi.csv')
result_list = []

for x in range(0, len(line)):
    processedTweet = processTweet(line[x])
#     print(processedTweet)
    featureVector = getFeatureVector(processedTweet)
#     print(featureVector)
    replaceAbb = translator(featureVector)
#     print(replaceAbb)
    result_list.append(replaceAbb)   
#     result_list.append(temp)
#     line = fp.readline()
#end loop
print(result_list)

tempStr = []
for baris in range(0, len(result_list)):
    tempo=""
    for kolom in range(0,len(result_list[baris])):
        tempo = tempo+" "+result_list[baris][kolom]
    tempStr.append(tempo)
# print(type(tempStr))

tempData = df['Mentions']
for x in range(0,len(tempStr)):
    df['Mentions'][x] = tempStr[x]
#     tempData.replace(x,tempStr[x])

print(df['Mentions'])

df['Sentiment'] = df['Sentiment'].replace({'Positive': 1, 'Negative': -1,'Neutral': 0})

df.to_csv("out.csv", sep=',', encoding='utf-8', index=False)