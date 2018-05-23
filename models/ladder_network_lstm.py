from __future__ import division, print_function
import os
import sys
__file__ = "__file__"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import math

import numpy as np
import pandas as pd
import tensorflow as tf

from utils import input_data
from utils.upload_data import DataUploader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from collections import namedtuple
from keras.utils.np_utils import to_categorical
from tensorflow import logging
from utils.preprocess_glove import get_vars
from utils.preprocess_glove import load_glove_embeddings
from tensorflow.python import debug as tf_debug

tf.logging.set_verbosity(tf.logging.INFO)

moving_average_updates = []  # this list stores the updates to be made to average mean and variance

LAYERS = [10, 2]
DENOISING_COST_WEIGHTS = [15.0, 0.10]
L = len(LAYERS) - 1
LEARNING_RATE = 0.2
NOISE_STD = 0.9

NUM_EXAMPLES = 25000
NUM_EPOCHS = 10
BATCH_SIZE = 100
NUM_LABELED = 25000

num_steps = (NUM_EXAMPLES/BATCH_SIZE) * NUM_EPOCHS

dims_encoder = zip(LAYERS[:-1], LAYERS[1:])
dims_decoder = zip(LAYERS[1:], LAYERS[:-1])

Encoding = namedtuple('Encoding', 'z m_unlabeled v_unlabeled')

def get_labeled(x):
  return x[:NUM_LABELED, :] if NUM_LABELED <= BATCH_SIZE else x[:BATCH_SIZE, :]


def get_unlabeled(x):
  return x[NUM_LABELED:, :] if NUM_LABELED <= BATCH_SIZE else x[BATCH_SIZE:, :]


def batch_normalization(batch, mean=None, var=None, is_training_and_clean=False, l=-1, moving_averages=None):
  if mean is None or var is None:
    mean, var = tf.nn.moments(batch, axes=[0])

  if is_training_and_clean is True:
    assign_mean = moving_averages['mean'][l-1].assign(mean)
    assign_var = moving_averages['var'][l-1].assign(var)
    moving_average_updates.append(ema.apply([moving_averages['mean'][l-1], moving_averages['var'][l-1]]))
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


ema = tf.train.ExponentialMovingAverage(decay=0.99)  # to calculate the moving averages of mean and variance

def forward(inputs, is_training=False, noise_std=0.0):
  with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
    weights = {
      'encoder': [tf.Variable(tf.random_normal(dim, stddev=(1 / math.sqrt(dim[0])))) for dim in dims_encoder],
      'decoder': [tf.Variable(tf.random_normal(dim, stddev=(1 / math.sqrt(dim[0])))) for dim in dims_decoder],
      'beta': [tf.Variable(0.0 * tf.ones([l])) for l in LAYERS[1:]],
      'gamma': [tf.Variable(1.0 * tf.ones([l])) for l in LAYERS[1:]]
    }

    moving_averages = {
      'mean': [tf.Variable(0.0 * tf.ones([l]), trainable=False, name="running_mean") for l in LAYERS[1:]],
      'var': [tf.Variable(1.0 * tf.ones([l]), trainable=False, name="running_var") for l in LAYERS[1:]]
    }

    def encoder(inputs, noise_std, z_func=None):
      with tf.variable_scope("encoder"):
        noise = tf.random_normal(tf.shape(inputs), stddev=noise_std)
        h = inputs + noise
        encodings = {}
        encodings[0] = Encoding(h, 0, 1-1e-10)

        for l in range(1, L + 1):
          z_pre = tf.matmul(h, weights['encoder'][l-1])  # pre-activation
          m, v = tf.nn.moments(get_unlabeled(z_pre), axes=[0])

          if is_training is True:
            if noise_std > 0:
              noise = tf.random_normal(tf.shape(z_pre), stddev=noise_std)
              z = batch_normalization(z_pre, is_training_and_clean=False) + noise
            else:
              z = batch_normalization(z_pre, l=l, is_training_and_clean=True, moving_averages=moving_averages)
          else:
            mean = ema.average(moving_averages['mean'][l-1])
            var = ema.average(moving_averages['var'][l-1])
            z = batch_normalization(z_pre, mean, var, is_training_and_clean=False)

          encodings[l] = Encoding(z, m, v)
          h = tf.nn.relu(z + weights["beta"][l-1])
        return encodings


    def decoder(clean, corr):
      # Decoder
      with tf.variable_scope("decoder"):
        z_est = {}
        d_cost = []  # to store the denoising cost of all LAYERS
        for l in range(L, -1, -1):
          z = get_unlabeled(clean[l].z)
          z_c = get_unlabeled(corr[l].z)
          m, v = clean[l].m_unlabeled, clean[l].v_unlabeled

          if l == len(LAYERS) - 1:
            u = get_unlabeled(classifier(corr[l].z))
          else:
            u = tf.matmul(z_est[l+1], weights['decoder'][l])

          u = batch_normalization(u)
          z_est[l] = gaussian_denoiser(z_c, u, LAYERS[l])
          z_est_bn = (z_est[l] - m) / v

          mse = tf.losses.mean_squared_error(z_est_bn, z)
          d_cost.append(mse * DENOISING_COST_WEIGHTS[l])

        u_cost = tf.add_n(d_cost, name="u_cost")

        return u_cost

    def classifier(z):
      return weights['gamma'][-1] * (z + weights["beta"][-1])

    clean = encoder(inputs, 0.0)
    corr = encoder(inputs, noise_std)
    reconstruction_cost = decoder(clean, corr)
    
    #logits_corr = get_labeled(corr[len(LAYERS) - 1].z)
    logits_corr = corr[len(LAYERS) - 1].z
    logits_clean = clean[len(LAYERS) - 1].z

    if is_training is True:
      return get_labeled(classifier(logits_corr)), reconstruction_cost
    else:
      return classifier(logits_clean)

def model_fn(features, labels, mode, params):
  #sparse_out = tf.layers.dense(features, 10)
  head = tf.contrib.estimator.binary_classification_head()
  if mode == tf.estimator.ModeKeys.TRAIN:
    features = get_labeled(features)
  inputs = tf.contrib.layers.embed_sequence(
      get_labeled(features), 5000, 50,
      initializer=params.embedding_initializer)

  lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(100)
  _, final_states = tf.nn.dynamic_rnn(
      lstm_cell, inputs, dtype=tf.float32)
  outputs = final_states.h

  logits = tf.layers.dense(outputs, 1)

  if mode == tf.estimator.ModeKeys.TRAIN:
    # add a full sparse layer here
    global_step = tf.train.get_global_step()
    if labels is not None:
        labels = tf.reshape(labels, [-1, 1])

    #y_preds, reconstruction_cost = forward(logits, is_training=True, noise_std=params.noise_std)
    #classification_cost = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_preds, labels=tf.cast(labels, tf.float32)))
    #loss = classification_cost
    loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.cast(labels, tf.float32)))
    train_step = tf.train.AdamOptimizer().minimize(loss, global_step=tf.train.get_global_step())
    preds = tf.reshape(tf.round(tf.nn.sigmoid(logits)), [-1])
    labels = tf.reshape(labels, [-1])
    casted = tf.cast(tf.equal(tf.cast(preds, tf.int64), labels), tf.int64)

    accuracy = tf.reduce_sum(casted) / BATCH_SIZE

    #train_step = tf.train.AdamOptimizer(params.learning_rate).minimize(loss, global_step=global_step)

    #with tf.control_dependencies([train_step]):
    #    train_step = tf.group(tf.group(*moving_average_updates))

    # print out logging hook 
    hooks = [tf.train.LoggingTensorHook({"accuracy": accuracy, "preds": preds[:10], "labels": labels[:10], "casted": casted}, every_n_iter=10)]

    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, train_op=train_step, training_hooks=hooks)

  if mode == tf.estimator.ModeKeys.PREDICT:
    #y_preds = forward(logits, is_training=False)
    return tf.estimator.EstimatorSpec(mode, predictions=tf.round(tf.nn.sigmoid(logits)))


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
  features, labels = get_vars()
  #labels = to_categorical(labels)
  Xtrain_ul, Xtrain_l, Xtest, ytrain_ul, ytrain_l, ytest = input_data.split_data(features, labels, NUM_LABELED, shuffle=True)
  print("This is Xtrain shape", Xtrain_ul.shape)

  #embedding_matrix = load_glove_embeddings('../data/glove.6B.50d.txt')

  def my_initializer(shape=None, dtype=tf.float32, partition_info=None):
      assert dtype is tf.float32
      return embedding_matrix

  params = dict({
    "model_type": 'ladder_network',
    "save_dir": '/tmp/checkpoints6',
    "learning_rate": LEARNING_RATE,
    "noise_std": NOISE_STD,
    "num_steps": 2500,
    "embedding_initializer": my_initializer
  })

  opt = tf.contrib.training.HParams(**params)

  my_checkpointing_config = tf.estimator.RunConfig(
      save_checkpoints_secs = 1*60,  # Save checkpoints every 20 minutes.
      keep_checkpoint_max = 10,       # Retain the 10 most recent checkpoints.
  )
  trainer = tf.estimator.Estimator(model_fn, params=opt, model_dir="/tmp/lstm_ladder", config=my_checkpointing_config)

  trainer.train(input_fn=lambda: get_input_fn_training(
    Xtrain_ul, Xtrain_l, Xtest, ytrain_ul, ytrain_l, ytest,
    BATCH_SIZE, NUM_LABELED
  ), steps=opt.num_steps)

  Xtrain_ul, Xtrain_l, Xtest, ytrain_ul, ytrain_l, ytest = input_data.split_data(features, labels, NUM_LABELED, shuffle=True)

  y_preds = np.fromiter(trainer.predict(tf.estimator.inputs.numpy_input_fn(
         x = Xtest,
         y = ytest,
         batch_size = BATCH_SIZE,
         shuffle = False)), np.int64)


  y_true = ytest
  accuracy = np.mean(y_preds == y_true)

  logging.info('accuracy=%0.3f', accuracy)
