import csv
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_auc_score
#from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn import svm

df = pd.read_csv('coba-training.csv', delimiter=',', index_col = False, encoding = "ISO-8859-1" )

# random untuk training data dan testing data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df['Mentions'], 
                                                    df['Sentiment'],
                                                    random_state=0)

# print('X_train first entry:\n\n', X_train.iloc[0])
# print('\n\nX_train shape: ', X_train.shape)

# pembobotan setiap term yang ada pada document menggunakan rumus tf-idf
# naive bayes tidak bisa membaca fitur dalam bentuk kata, dia bisa membaca dalam bentuk angka
vect = TfidfVectorizer(min_df=5, use_idf=True, ngram_range=(1,2)).fit(X_train)
# vect = TfidfVectorizer(min_df=5, use_idf=True).fit(X_train)
vect.get_feature_names()
# tf = TfidfVectorizer(token_pattern=u'(?u)\\b\\w+\\b')
# tfidf_matrix = tf.fit_transform(X_train)
# print(tfidf_matrix.toarray())
# tf.

X_train_vectorized = vect.transform(X_train)

mnb = MultinomialNB()

mnb.fit(X_train_vectorized, y_train)

predictions = mnb.predict(vect.transform(X_test))

# print('Confusion Metrix: ', confusion_matrix(y_test,predictions))
# print('ACC: ', accuracy_score(y_test, predictions))

# feature_names = np.array(vect.get_feature_names())

# sorted_tfidf_index = X_train_vectorized.max(0).toarray()[0].argsort()

# print('Smallest tfidf:\n{}\n'.format(feature_names[sorted_tfidf_index[:10]]))
# print('Largest tfidf: \n{}'.format(feature_names[sorted_tfidf_index[:-11:-1]]))

dftest = pd.read_csv('coba-tes.csv', delimiter=',', index_col = False, encoding = "ISO-8859-1" )
for index, row in dftest.iterrows():
    # mnb.predict(vect.transform([row['Mentions']]))
    dftest['Sentiment'] = dftest.apply (lambda row: mnb.predict(vect.transform([row['Mentions']])),axis=1)
    # print(hasil)

dftest['Sentiment'] = dftest['Sentiment'].replace({'[1]': 1, '[0]': 0, '[-1]': -1})
dftest.to_csv("coba-out.csv", index=False, encoding = "ISO-8859-1")