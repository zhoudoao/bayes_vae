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

import tensorflow as tf
import tensorflow_probability as tfp

import matplotlib.pyplot as plt
import numpy as np


# MNIST datasets
INPUT_SHAPE = 784
OUTPUT_SHAPE = 10
BATCH_SIZE = 64
TRAIN_SIZE = 60000

(train_images, train_labels), (test_images, test_labels) \
    = tf.keras.datasets.mnist.load_data()

train_images = tf.cast(train_images[:TRAIN_SIZE].reshape(-1, 784),
    tf.float32) / 255.
test_images = tf.cast(test_images.reshape(-1, 784), tf.float32) / 255.

train_labels = tf.one_hot(train_labels[:TRAIN_SIZE], OUTPUT_SHAPE)
test_labels = tf.one_hot(test_labels, OUTPUT_SHAPE)


def neural_network(hidden_layers, activation='relu'):
    """Define a neural network.
    
    Args:
        hidden_layers: List. Numbers of each hidden layer.
    
    Returns:
        A sequential model.
    """

    nn = tf.keras.Sequential()
    for hidden_layer in hidden_layers:
        nn.add(tf.keras.layers.Dense(hidden_layer, activation=activation))
    
    nn.add(tf.keras.layers.Dense(OUTPUT_SHAPE, activation='softmax'))
    return nn


def train(model):
    """Training model.
    
    Args:
        Model or sequential model.
    
    Returns:
        History of the model training.
    """    
    loss = tf.keras.losses.categorical_crossentropy
    metrics = ['accuracy']
    optimizer = tf.keras.optimizers.Adam(lr)

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    history = model.fit(train_images, train_labels, epochs=epochs,
        batch_size=BATCH_SIZE, validation_data=(test_images, test_labels), 
        verbose=1)
    return history


lr = 0.01
epochs = 20
nn_2layers = neural_network([400, 400])
nn_2l_history = train(nn_2layers)


# Metrics
def plot_history(history, title=None):
    """Plot the loss and accuracy of the model during training process. """

    fig = plt.figure(figsize=(16, 10))
    # fig, ax = plt.subplots(1, 2)
    
    x = history.epoch
    
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    train_accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    
    # plot loss
    ax0 = fig.add_subplot(1, 2, 1)
    loss_fig = ax0.plot(x, train_loss, '--',
        label='Train loss')
    ax0.plot(x, val_loss, color=loss_fig[0].get_color(),
        label='Val loss')
    
    ax1 = fig.add_subplot(1, 2, 2)
    acc_fig = ax1.plot(x, train_accuracy, '--',
        label='Train accuracy')
    ax1.plot(x, val_accuracy, color=acc_fig[0].get_color(),
        label='Val accuracy')
    
    ax0.set_xlabel('Epoch')
    ax0.set_ylabel('Loss')
    ax0.legend()
    ax0.set_xlim([0, max(history.epoch)])

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.set_xlim([0, max(history.epoch)])

    if title is not None:
        fig.suptitle(title)
    
    
plot_history(nn_2l_history, title='2 layers neural network')


# Todo
# def plot_weights(model, num_layer):
#     # num = num_layer * 2
#     weights = model.trainable_weights

#     fig = plt.figure()
#     for i in range(num_layer):
#         ax_left = fig.add_subplot(num_layer, 2, 2*i)
#         ax_left.hist()


    

# Todo
# Save model
    
exit    







