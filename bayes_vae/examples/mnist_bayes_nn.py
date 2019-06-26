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

"""Train a Bayesian neural network to classify MNIST dataset. The result is 
regarded as benchmark of Bayesian neural network.

The architecture is 2-layer neural network.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_probability as tfp

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import pickle
import functools

tfd = tfp.distributions

home = os.path.expanduser('~')

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
    """Training model.
    
    Args:
        model: Model or sequential model.
        loss: Loss function. Such as categorical_crossentropy, 
            mean_squared_error.
        optimizer: Optimizer method such as Adam, SGD.
        metrics: List. A list of names of metrics. Such as ['mse'], ['acc'].
    
    Returns:
        History of the model training.
    """    
    
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    history = model.fit(train_images, train_labels, epochs=epochs,
        batch_size=BATCH_SIZE, validation_data=(test_images, test_labels), 
        verbose=1)
    return history


lr = 0.01
epochs = 5
hidden_layers = [400, 400]
# loss_fn = functools.partial(tf.keras.losses.categorical_crossentropy,
#     from_logits=True)
# loss_fn = tf.keras.losses.categorical_crossentropy
neg_log_lik = lambda labels, logits: \
    -tfd.OneHotCategorical(logits=logits).log_prob(labels)

metrics = ['accuracy']
optimizer = tf.keras.optimizers.Adam(lr)
bnn_2layers = bayesian_nn(hidden_layers)

def elbo(model):
    def loss_fn(labels, logits):
        return tf.reduce_mean(neg_log_lik(labels, logits)) + \
            sum(model.losses) / TRAIN_SIZE
    return loss_fn

# def loss_fn(labels, logits):
#     kl = sum(bnn_2layers.losses) / TRAIN_SIZE

# loss_fn = neg_log_lik + bnn_2layers.losses
bnn_2l_history = train(bnn_2layers, elbo(bnn_2layers), optimizer, metrics)
# for i in range(50):
#     train(bnn_2layers, elbo(bnn_2layers), optimizer, metrics)
#     logits = bnn_2layers(train_images)
#     print(logits)


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
    
    
plot_history(bnn_2l_history, title='2 layers bayesian neural network')


def plot_weights(model, num_layer):
    """Plot weights of the model.
    
    Args:
        model: Model or Sequential model.
        num_layer: Number of the hidden layers.
    """
    weights = model.trainable_weights

    fig = plt.figure()
    for i in range(num_layer):
        kernel_posterior_loc = weights[3*i].numpy().flatten()
        kernel_posterior_scale = weights[3*i+1].numpy().flatten()
        bias_posterior_loc = weights[3*i+2].numpy()

        ax_left = fig.add_subplot(num_layer, 3, 3*i+1)
        sns.distplot(kernel_posterior_loc, ax=ax_left)
        ax_left.legend()
        ax_left.set_title('kernel {}'.format(i))
        
        ax_middle = fig.add_subplot(num_layer, 3, 3*i+2)
        sns.distplot(kernel_posterior_scale, ax=ax_middle)
        ax_middle.legend()
        ax_middle.set_title('kernel posterior unstranformed scale {}'.format(i))

        ax_right = fig.add_subplot(num_layer, 3, 3*i+3)
        sns.distplot(bias_posterior_loc, ax=ax_right)
        ax_right.legend()
        ax_right.set_title('bias {}'.format(i))


num_layer = len(hidden_layers)    
plot_weights(bnn_2layers, num_layer)



exit