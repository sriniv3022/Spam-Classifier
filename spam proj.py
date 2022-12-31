# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd

msg= pd.read_csv('smsspamcollection\SMSSpamCollection', sep='\t', names= ["label","message"])

#Data Cleaning and Preprocessing
import re
import nltk
nltk.download('wordnet')
nltk.download('stopwords')

from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer
lem= WordNetLemmatizer()

corpus= []
for i in range(0, len(msg)):
    review = re.sub('[^a-zA-Z]', ' ', msg['message'][i])
    review = review.lower()
    review = review.split()
    
    review = [lem.lemmatize(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
    
#Bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=2500)
X = cv.fit_transform(corpus).toarray()

y=pd.get_dummies(msg['label'])
y=y.iloc[:,1].values

#Train Test Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

#Naive Bayes Classifier
from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(X_train, y_train)

y_pred = spam_detect_model.predict(X_test)

from sklearn.metrics import confusion_matrix
conf_m= confusion_matrix(y_test,y_pred)

from sklearn.metrics import accuracy_score
accuracy= accuracy_score(y_test,y_pred)

    
    