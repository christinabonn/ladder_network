import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import tensorflow as tf
import unittest

from keras.utils.np_utils import to_categorical
from sklearn.datasets import fetch_mldata
from utils import input_data


mnist = fetch_mldata('MNIST original')

BATCH_SIZE = 100
N_LABELED = 100

X = np.multiply(mnist.data, 1.0 / 255.0)
y = to_categorical(mnist.target, num_classes=10)


class InputDataTest(tf.test.TestCase):

  def test_split_data(self):
    Xtrain, Xtrain_l, Xtest, ytrain, ytrain_l, ytest = input_data.split_data(X, y, N_LABELED)

    total_length_X = len(Xtrain) + len(Xtest)
    total_length_y = len(ytrain) + len(ytest)

    self.assertEqual(total_length_X, len(mnist.data))
    self.assertEqual(total_length_y, len(mnist.target))
    self.assertEqual(len(Xtrain_l), BATCH_SIZE)
    self.assertEqual(len(ytrain_l), BATCH_SIZE)

    self.assertEqual(Xtrain.dtype, np.float32)
    self.assertEqual(Xtest.dtype, np.float32)
    self.assertEqual(ytrain.dtype, np.int64)
    self.assertEqual(ytest.dtype, np.int64)

  def test_dataset(self):
    Xtrain_ul, Xtrain_l, Xtest, ytrain_ul, ytrain_l, ytest = input_data.split_data(X, y, N_LABELED)
    dataset = input_data.Data(Xtrain_ul, Xtrain_l, Xtest,
                              ytrain_ul, ytrain_l, ytest,
                              N_LABELED, BATCH_SIZE)
    with self.test_session():
      features, labels = dataset.next_batch()

      sess_feat = features.eval()
      sess_labels = labels.eval()

      self.assertAllEqual(sess_feat.shape, [BATCH_SIZE * 2, 784])
      self.assertAllEqual(sess_labels.shape, [BATCH_SIZE, 10])

      concatted_X = np.concatenate(
        (dataset.Xtrain_l[:BATCH_SIZE], dataset.Xtrain_ul[:BATCH_SIZE])
      )

      self.assertAllEqual(
        np.argmax(sess_labels, axis=1),
        np.argmax(dataset.ytrain_l[:BATCH_SIZE], axis=1)
      )
      self.assertAllEqual(
        sess_feat[:BATCH_SIZE * 2],
        concatted_X[:BATCH_SIZE * 2]
      )

unittest.main()
