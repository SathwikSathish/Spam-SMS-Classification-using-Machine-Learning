# Spam-SMS-Classification-using-Machine-Learning

## DESCRIPTION

Spam classification is the process of identifying whether a given text message is spam or not. Classification algorithms :Naive Bayes,Decision-Tree,Random Forest classifiers are used to perform spam classification by training a model on a dataset of labeled text messages (i.e., a dataset that includes both spam and non-spam messages, where each message has been labeled as one or the other).

To train a machine learning model for spam classification, large dataset of labeled text messages is collected. This dataset includes a diverse range of messages, including both spam and non-spam messages. Once dataset is obtained,it can be used it to train a classification model using any of a variety of machine learning algorithms.

Once the model has been trained, it can be used to classify new text messages as either spam or non-spam. The model will make predictions based on the features (i.e., the characteristics or attributes) of the text messages that it has been trained on. For example, it may look at the words used in the message, the sender's address, or the message's subject line.

In general, the accuracy of a spam classification model will depend on several factors, including the quality of the training dataset, the choice of machine learning algorithm, and the features used to train the model. With careful tuning and optimization, it is possible to achieve very high levels of accuracy in spam classification using machine learning.

## DATASET

The dataset for this project has been obtained from the Kaggle.The link is given below:

https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

## PRE-REQUIRED LIBRARIES

  import numpy as np
  
  import pandas as pd
  
  from sklearn import preprocessing
  
  #### FOR VISUALIZATION
   import matplotlib.pyplot as plt
  
   import seaborn as sns

   %matplotlib inline
   
   #### FOR BALANCING THE DATASET
   from collections import Counter
   
from sklearn.datasets import make_classification

from imblearn.over_sampling import RandomOverSampler

#### IMPORTING ESSENTIAL LIBRARIES FOR PERFORMING NLP

import nltk

import re

nltk.download('stopwords')

nltk.download('wordnet')

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer

#### FOR VECTORIZING THE CORPUS

from sklearn.feature_extraction.text import TfidfVectorizer

#### FOR VALIDATION AND ACCURACY
from sklearn.model_selection import cross_val_score

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.model_selection import train_test_split

#### THE MACHINE LEARNING MODELS
from sklearn.naive_bayes import MultinomialNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier        

from sklearn.ensemble import VotingClassifier


