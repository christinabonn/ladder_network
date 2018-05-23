import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer


"""
This class will take a single dataset, will convert it to
use the Dataset API and produce a semi-labeled dataset
"""

class Data(object):

  def __init__( self,
                Xtrain_ul,
                Xtrain_l,
                Xtest,
                ytrain_ul,
                ytrain_l,
                ytest, 
                num_labeled,
                batch_size, 
                shuffle=False
              ):

    # No need to do if labeled and unlabeled are already organized
    # Create a joined dataset as a tuple of inputs and outputs
    # [{(feature_a, feature_b, feature_c), (label)}]
    dx_train_ul = tf.data.Dataset.from_tensor_slices(Xtrain_ul)
    dy_train_ul = tf.data.Dataset.from_tensor_slices(tf.cast(ytrain_ul, dtype=tf.int64))
    train_dataset_ul = tf.data.Dataset.zip((dx_train_ul, dy_train_ul)).repeat()

    dx_train_l = tf.data.Dataset.from_tensor_slices(Xtrain_l)
    dy_train_l = tf.data.Dataset.from_tensor_slices(ytrain_l)
    train_dataset_l = tf.data.Dataset.zip((dx_train_l, dy_train_l)).repeat()

    dx_test = tf.data.Dataset.from_tensor_slices(Xtest)
    dy_train = tf.data.Dataset.from_tensor_slices(tf.cast(ytrain_ul, dtype=tf.int64))
    test_dataset = tf.data.Dataset.zip((dx_test, dy_train)).repeat()

    # Merge labeled and unlabeled datasets together for each batch
    # Used this instead of interleaving as we have to join two datasets for each get_next
    # as opposed to interleaving a labeled and then an unlabeled dataset together

    # For every single iteration step take batch size from the unlabeled dataset 
    # and batch size of the labeled dataset. If we have exhausted all examples in 
    # a particular dataset, it repeats and goes from the start again.
    datasets_to_zip = (train_dataset_l.batch(batch_size), train_dataset_ul.batch(batch_size))
    train_dataset = tf.data.Dataset.zip(datasets_to_zip).map(concat)

    self.num_labeled = num_labeled
    self.batch_size = batch_size

    self.Xtrain_ul = Xtrain_ul
    self.Xtrain_l = Xtrain_l
    self.Xtest = Xtest
    self.ytrain_ul = ytrain_ul
    self.ytrain_l = ytrain_l
    self.ytest = ytest

    self.train_dataset = train_dataset
    self.test_dataset = test_dataset

    self.train = train_dataset.make_one_shot_iterator()
    self.test = test_dataset.make_one_shot_iterator()

  def next_batch(self):
    # Returns an 
    # * array of images where 0,max_labeled_per_batch are images that have labels and 
    #   max_labeled_per_batch,end are images that have no labels.
    # * array of labels where 0,max_labeled_per_batch are labels that have 
    batch_size = self.batch_size
    num_labeled = self.num_labeled

    max_labeled_per_batch = num_labeled if num_labeled <= batch_size else batch_size

    x_label = self.train.get_next()
    x = x_label[0]
    y = x_label[1][:max_labeled_per_batch]
    return x, y


def split_data(X, y, num_labeled, test_size=0.5, shuffle=False):
  Xtrain, Xtest, ytrain, ytest = train_test_split(X,
                                                  y,
                                                  test_size=test_size)

  Xtrain = np.ndarray.astype(Xtrain, dtype=np.int64)
  Xtest = np.ndarray.astype(Xtest, dtype=np.int64)
  ytrain = np.ndarray.astype(ytrain, dtype=np.int64)
  ytest = np.ndarray.astype(ytest, dtype=np.int64)

  Xtrain_l = Xtrain[:num_labeled]
  ytrain_l = ytrain[:num_labeled]

  return Xtrain, Xtrain_l, Xtest, ytrain, ytrain_l, ytest


def concat(*ds_elements):
  #Concatenate each component list
  lists = map(list, zip(*ds_elements))
  return tuple(tf.concat(l, axis=0) for l in list(lists))

