import os  #noqa
import sys  #noqa
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  #noqa

import numpy as np
import pandas as pd

from keras.layers import Dense
from keras.models import Sequential
from sklearn.feature_extraction.text import TfidfVectorizer
from utils.upload_data import DataUploader
from sklearn.model_selection import train_test_split

def fit(features, labels):
  Xtrain, Xtest, ytrain, ytest = train_test_split(features, labels, test_size=0.33)
  model = Sequential()
  model.add(Dense(100,
                  input_shape=(features.shape[1],),
                  activation='relu'))
  model.add(Dense(2, activation='softmax'))

  model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
  print(model.summary())
  model.fit(Xtrain, ytrain, epochs=100, batch_size=100, verbose=2)

  loss, acc = model.evaluate(Xtest, ytest, verbose=2)
  print('Test Accuracy: %f' % (acc*100))
  return model


def balance_data(text_data):
  texts = [x[1] for x in text_data]
  targets = [x[0] for x in text_data] 


  np_texts = np.array(texts)
  np_targets = np.array(targets)
  num_spam = len(np.where(np_targets == 'spam')[0])
  num_ham = len(np.where(np_targets == 'ham')[0])

  num_max = min(num_ham, num_spam)

  ham_indices = np.where(np_targets == 'ham')[0]
  np.random.shuffle(ham_indices)
  ham_indices = ham_indices[:num_max]

  spam_indices = np.where(np_targets == 'spam')[0]
  np.random.shuffle(spam_indices)
  spam_indices = spam_indices[:num_max]

  indices_to_keep = np.concatenate((ham_indices, spam_indices), axis=0)
  np.random.shuffle(indices_to_keep)

  texts = np_texts[indices_to_keep]
  targets = np_targets[indices_to_keep]

  return texts, targets


def preprocess_bow(texts, targets, one_hot=False):
  tv = TfidfVectorizer(min_df=1, stop_words='english', norm='l2')
  x_traintv = tv.fit_transform(texts)

  inputs = x_traintv.toarray()

  if one_hot is True:
    outputs = pd.get_dummies(targets).as_matrix()
  else:
    outputs = targets

  return inputs, outputs


if __name__ == "__main__":
  text_data = DataUploader.upload_and_read_from_url('http://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip',  #noqa
                'SMSSpamCollection',
                'spam_ham_data.csv')
  texts, targets = balance_data(text_data)
  features, labels = preprocess_bow(texts, targets, one_hot=True)

  model = fit(features[:100], labels[:100])
