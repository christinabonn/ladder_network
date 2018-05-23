from __future__ import division, print_function
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import math
import numpy as np

import tensorflow as tf
from utils import input_data

from collections import namedtuple
from keras.utils.np_utils import to_categorical
from sklearn.datasets import fetch_mldata
from tensorflow import logging
from tensorflow.python import debug as tf_debug

tf.logging.set_verbosity(tf.logging.INFO)


LAYERS = [784, 1000, 500, 250, 250, 250, 10]
DENOISING_COST_WEIGHTS = [1000.0, 10.0, 0.10, 0.10, 0.10, 0.10, 0.10]
L = len(LAYERS) - 1
LEARNING_RATE = 0.02
NOISE_STD = 0.3
NUM_EXAMPLES = 60000
NUM_EPOCHS = 150
NUM_LABELED = 100
BATCH_SIZE = 100

num_steps = (NUM_EXAMPLES/BATCH_SIZE) * NUM_EPOCHS

dims_encoder = zip(LAYERS[:-1], LAYERS[1:])
dims_decoder = zip(LAYERS[1:], LAYERS[:-1])

Encoding = namedtuple('Encoding', 'z m_unlabeled v_unlabeled')
ema = tf.train.ExponentialMovingAverage(decay=0.99)  # to calculate the moving averages of mean and variance
moving_average_updates = []  # this list stores the updates to be made to average mean and variance

class LadderNetwork(object):
  def __init__(self):
    self.mean = [tf.Variable(0.0 * tf.ones([l]), trainable=False) for l in LAYERS[1:]]
    self.var = [tf.Variable(1.0 * tf.ones([l]), trainable=False) for l in LAYERS[1:]]

  def predict(self, inputs):
    with tf.variable_scope('ladder_network', reuse=tf.AUTO_REUSE):
      encodings = self.encoder(inputs, 0.0, False)
      logits = encodings[L].z
      return self.classifier(logits), self.var[-1]

  def loss(self, inputs, labels, noise_std=0.0):
    with tf.variable_scope('ladder_network', reuse=tf.AUTO_REUSE):
      clean = self.encoder(inputs, 0.0, True)
      corr = self.encoder(inputs, noise_std, True)
      
      logits_corr = corr[len(LAYERS) - 1].z
      logits_clean = clean[len(LAYERS) - 1].z
      y_preds = get_labeled(self.classifier(logits_corr))

      classification_cost = -tf.reduce_mean(tf.reduce_sum(tf.cast(labels, dtype=tf.float32) *
                                                                  tf.log(y_preds + 1e-10), 1))
      reconstruction_cost = self.decoder(clean, corr)

      loss = classification_cost + reconstruction_cost

      return loss, self.var[-1], y_preds, classification_cost, reconstruction_cost

  def encoder(self, inputs, noise_std, is_training=False):
    with tf.variable_scope('beta', reuse=tf.AUTO_REUSE):
      betas = [tf.get_variable(str(i), initializer=0.0 * tf.ones([l])) for i, l in enumerate(LAYERS[1:])]

    with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
      encoder_weights = [tf.get_variable(str(i), initializer=tf.random_normal(dim, stddev=(1 / math.sqrt(dim[0])))) for i, dim in enumerate(dims_encoder)]

      noise = tf.random_normal(tf.shape(inputs), stddev=noise_std)
      h = tf.cast(inputs, tf.float32) + noise
      encodings = {}
      encodings[0] = Encoding(h, 0, 1-1e-10)

      for l in range(1, L + 1):
        z_pre = tf.matmul(h, encoder_weights[l-1])  # pre-activation
        m, v = tf.nn.moments(get_unlabeled(z_pre), axes=[0])

        if is_training is True:
          if noise_std > 0:
            noise = tf.random_normal(tf.shape(z_pre), stddev=noise_std)
            z = batch_normalization(z_pre, is_training_and_clean=False) + noise
          else:
            z = batch_normalization(z_pre, l=l, is_training_and_clean=True, mean_ma=self.mean, var_ma=self.var)
        else:
          mean = ema.average(self.mean[l-1])
          var = ema.average(self.var[l-1])
          z = batch_normalization(z_pre, mean, var, is_training_and_clean=False)

        encodings[l] = Encoding(z, m, v)
        h = tf.nn.relu(z + betas[l-1])
      return encodings


  def decoder(self, clean, corr):
    with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
      decoder_weights = [tf.get_variable(str(i), initializer=tf.random_normal(dim, stddev=(1 / math.sqrt(dim[0])))) for i, dim in enumerate(dims_decoder)]
      # Decoder
      z_est = {}
      d_cost = []  # to store the denoising cost of all LAYERS
      for l in range(L, -1, -1):
        z = get_unlabeled(clean[l].z)
        z_c = get_unlabeled(corr[l].z)
        m, v = clean[l].m_unlabeled, clean[l].v_unlabeled

        if l == len(LAYERS) - 1:
          u = get_unlabeled(self.classifier(corr[l].z))
        else:
          u = tf.matmul(z_est[l+1], decoder_weights[l])

        u = batch_normalization(u)
        z_est[l] = gaussian_denoiser(z_c, u, LAYERS[l])
        z_est_bn = (z_est[l] - m) / v

        mse = tf.losses.mean_squared_error(z_est_bn, z)
        d_cost.append(mse * DENOISING_COST_WEIGHTS[l])

      u_cost = tf.add_n(d_cost, name="u_cost")

      return u_cost

  def classifier(self, z):
    with tf.variable_scope('beta', reuse=tf.AUTO_REUSE):
      beta = tf.get_variable(str(L), shape=LAYERS[L])

    with tf.variable_scope('gamma', reuse=tf.AUTO_REUSE):
      gamma = tf.get_variable(str(L), initializer=tf.Variable(1.0 * tf.ones([LAYERS[L]])))

    return tf.nn.softmax(gamma * (z + beta))

def model_fn(features, labels, mode, params):
  ln = LadderNetwork()
  if mode == tf.estimator.ModeKeys.TRAIN:
    loss, mm, y_preds, classification_cost, reconstruction_cost = ln.loss(features, labels, params.noise_std)
    global_step = tf.train.get_global_step()

    preds = tf.argmax(y_preds, 1)
    true = tf.argmax(labels, 1)
    casted = tf.cast(tf.equal(tf.cast(preds, tf.int64), true), tf.int64)
    accuracy = tf.reduce_sum(casted) / BATCH_SIZE

    
    # print out logging hook 
    hooks = [tf.train.LoggingTensorHook({
      "accuracy": accuracy,
      "preds": preds[:10],
      "labels": true[:10],
      "casted": casted,
      "classification_cost": classification_cost,
      "reconstruction_cost": reconstruction_cost,
      }, every_n_iter=10)]


    # with tf.control_dependencies([mm]):
    #   #hooks = tf.train.LoggingTensorHook({"moving_mean": mm}, every_n_iter=100)

    train_step = tf.train.AdamOptimizer(params.learning_rate).minimize(loss, global_step=global_step)

    with tf.control_dependencies([train_step]):
        train_step = tf.group(tf.group(*moving_average_updates))

    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, train_op=train_step, training_hooks=hooks])

  if mode == tf.estimator.ModeKeys.PREDICT:
    y_preds, mm = ln.predict(features)
    hooks = tf.train.LoggingTensorHook({"moving_mean": mm}, every_n_iter=1)
    return tf.estimator.EstimatorSpec(mode, predictions=tf.argmax(y_preds, 1), prediction_hooks=[hooks])


def get_labeled(x):
  return x[:NUM_LABELED, :] if NUM_LABELED <= BATCH_SIZE else x[:BATCH_SIZE, :]


def get_unlabeled(x):
  return x[NUM_LABELED:, :] if NUM_LABELED <= BATCH_SIZE else x[BATCH_SIZE:, :]


def batch_normalization(batch, mean=None, var=None, is_training_and_clean=False, l=-1, mean_ma=None, var_ma=None):
  if mean is None or var is None:
    mean, var = tf.nn.moments(batch, axes=[0])

  if is_training_and_clean is True:
    assign_mean = mean_ma[l-1].assign(mean)
    assign_var = var_ma[l-1].assign(var)
    moving_average_updates.append(ema.apply([mean_ma[l-1], var_ma[l-1]]))
    with tf.control_dependencies([assign_mean, assign_var]):
      return tf.nn.batch_normalization(batch, mean, var, None, None, 1e-10)
  else:
    return tf.nn.batch_normalization(batch, mean, var, None, None, 1e-10)

def gaussian_denoiser(z_noisy, u, size):
  """
  Approximate optimal denoising function z[l + 1]
  the reconstruction from the previous layer z[l + 1]
  proposed by page 7 of https://arxiv.org/pdf/1507.02672.pdf
  mu
  """

  a1 = tf.Variable(0. * tf.ones([size]), name='a1')
  a2 = tf.Variable(1. * tf.ones([size]), name='a2')
  a3 = tf.Variable(0. * tf.ones([size]), name='a3')
  a4 = tf.Variable(0. * tf.ones([size]), name='a4')
  a5 = tf.Variable(0. * tf.ones([size]), name='a5')

  # v
  a6 = tf.Variable(0. * tf.ones([size]), name='a6')
  a7 = tf.Variable(1. * tf.ones([size]), name='a7')
  a8 = tf.Variable(0. * tf.ones([size]), name='a8')
  a9 = tf.Variable(0. * tf.ones([size]), name='a9')
  a10 = tf.Variable(0. * tf.ones([size]), name='a10')

  mu = a1 * tf.sigmoid(a2 * u + a3) + a4 * u + a5
  v = a6 * tf.sigmoid(a7 * u + a8) + a9 * u + a10

  z = (z_noisy - mu) * v + mu
  return z



estimator = tf.estimator.Estimator(model_fn)

def get_input_fn_training(Xtrain_ul, Xtrain_l, Xtest, ytrain_ul, ytrain_l, ytest, batch_size, num_labeled):
  """
  Organize training data.
  Returns a tuple with images of size batch_size * 2 and labels of size batch_size
  """
  dataset = input_data.Data(Xtrain_ul,
                            Xtrain_l,
                            Xtest,
                            ytrain_ul,
                            ytrain_l,
                            ytest,
                            num_labeled, 
                            batch_size, 
                            shuffle=True)
  return dataset.next_batch()

if __name__ == "__main__":
  mnist = fetch_mldata('MNIST original')
  X = np.multiply(mnist.data, 1.0 / 255.0)
  y = to_categorical(mnist.target, num_classes=10)
  Xtrain_ul, Xtrain_l, Xtest, ytrain_ul, ytrain_l, ytest = input_data.split_data(X, y, NUM_LABELED, shuffle=True)

  params = dict({
    "model_type": 'ladder_network',
    "save_dir": '/tmp/checkpoints6',
    "learning_rate": LEARNING_RATE,
    "noise_std": NOISE_STD,
    "num_steps": num_steps,
  })

  opt = tf.contrib.training.HParams(**params)

  trainer = tf.estimator.Estimator(model_fn, params=opt, model_dir="../checkpoints")

  trainer.train(input_fn=lambda: get_input_fn_training(
    Xtrain_ul, Xtrain_l, Xtest, ytrain_ul, ytrain_l, ytest,
    BATCH_SIZE, NUM_LABELED
  ), steps=2)

  y_preds = np.fromiter(trainer.predict(tf.estimator.inputs.numpy_input_fn(
         x = Xtest,
         y = ytest,
         batch_size = BATCH_SIZE,
         shuffle = False)), np.int64)

  y_true = np.argmax(ytest, axis=1)
  accuracy = np.mean(y_preds == y_true)

  logging.info('accuracy=%0.3f', accuracy)
