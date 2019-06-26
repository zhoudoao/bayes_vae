# Copyright 2019, zhoudoao@gmail.com.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Train a neural network to classify MNIST dataset. The result is regarded
as benchmark of Bayesian neural network.

The architecture is 2-layer neural network.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

import tensorflow_probability as tfp

tfd = tfp.distributions


# MNIST datasets
INPUT_SHAPE = 784
OUTPUT_SHAPE = 10
BATCH_SIZE = 64
TRAIN_SIZE = 60000
TESTING_SIZE = 10000
EPOCHS = 5

(train_images, train_labels), (test_images, test_labels) \
    = tf.keras.datasets.mnist.load_data()

train_images = tf.cast(train_images[:TRAIN_SIZE].reshape(-1, 784),
    tf.float32) / 255.
test_images = tf.cast(test_images.reshape(-1, 784), tf.float32) / 255.

train_labels = tf.one_hot(train_labels[:TRAIN_SIZE], OUTPUT_SHAPE)
test_labels = tf.one_hot(test_labels, OUTPUT_SHAPE)

train_dataset = tf.compat.v1.data.Dataset.from_tensor_slices((train_images, 
    train_labels)).repeat(EPOCHS).batch(BATCH_SIZE).shuffle(TRAIN_SIZE)
test_dataset = tf.compat.v1.data.Dataset.from_tensor_slices((test_images, 
    test_labels)).repeat(1).batch(BATCH_SIZE)

iterater = tf.compat.v1.data.Iterator.from_structure(
    output_types = train_dataset.output_types,
    output_shapes = train_dataset.output_shapes)

train_iterater_initializer = iterater.make_initializer(train_dataset)
test_iterater_initializer = iterater.make_initializer(test_dataset)
images, onehot_labels = iterater.get_next()
labels = tf.argmax(onehot_labels, axis=1)


def bayesian_nn(hidden_layers, prob_layer=tfp.layers.DenseReparameterization, 
    activation='relu'):
    """Define a Bayesian neural network.
    
    Args:
        hidden_layers: List. Numbers of each hidden layer.
        prob_layer: Probabilistic layer. 
        activation: Activation function.
    
    Returns:
        A sequential model with tfp.layers.
    """
    bnn = tf.keras.Sequential()
    
    for hidden_layer in hidden_layers:
        bnn.add(prob_layer(hidden_layer, activation=activation))
    
    bnn.add(prob_layer(OUTPUT_SHAPE))
    return bnn


def train(model, loss, optimizer, metrics):
    pass

lr = 0.001
hidden_layers = [400, 400]

# images = tf.compat.v1.placeholder(tf.float32, [None, INPUT_SHAPE])
# onehot_labels = tf.compat.v1.placeholder(tf.float32, [None, OUTPUT_SHAPE,])


bnn_2layers = bayesian_nn(hidden_layers)

logits = bnn_2layers(images)
labels_distribution = tfd.Categorical(logits=logits)

neg_log_lik = -tf.reduce_mean(labels_distribution.log_prob(labels))
kl = sum(bnn_2layers.losses) / TRAIN_SIZE
loss = neg_log_lik + kl

predictions = tf.argmax(logits, axis=1)
accuracy, accuracy_update_op = tf.compat.v1.metrics.accuracy(
    labels=labels, predictions=predictions)

optimizer = tf.compat.v1.train.AdamOptimizer(lr)
train_op = optimizer.minimize(loss)

init_op = tf.group(tf.compat.v1.global_variables_initializer(),
    tf.compat.v1.local_variables_initializer())

with tf.compat.v1.Session() as sess:
    sess.run(init_op)
    sess.run(train_iterater_initializer)
    i = 0
    # for images_batch, labels_batch in train_iterator:
    try:
        while True:
            _ = sess.run([train_op, accuracy_update_op])
            
            if i % 100 == 0:
                loss_value, accuracy_value = sess.run([loss, accuracy]) 
                print('Step: {:>3d} Loss: {:.3f} Accuracy: {:.3f}'.format(
                    i, loss_value, accuracy_value))
        
            i += 1
    except tf.errors.OutOfRangeError:
        pass


# Todo: Plot loss

# Todo: Plot weight distribution





exit
