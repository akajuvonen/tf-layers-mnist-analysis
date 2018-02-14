import numpy as np
import tensorflow as tf


def cnn_model(features, labels, mode):
    """CNN model function. TODO: Not done yet.
    """

    # Have to reshape the inputs for tf
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

    # The first convolutional layers
    conv1 = tf.layers.conv2d(inputs=input_layer, filters=16,
                             kernel_size=[3, 3], padding="same",
                             activation=tf.nn.relu)
    conv2 = tf.layers.conv2d(inputs=conv1, filters=16, kernel_size=[3, 3],
                             padding="same", activation=tf.nn.relu)
    # 2D pooling layer
    pool1 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # The second set of convolutional and pooling layers
    conv3 = tf.layers.conv2d(inputs=pool1, filters=32, kernel_size=[3, 3],
                             padding="same", activation=tf.nn.relu)
    conv4 = tf.layers.conv2d(inputs=conv3, filters=32, kernel_size=[3, 3],
                             padding="same", activation=tf.nn.relu)
    # 2D pooling layer
    pool2 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)

def main(dummy):
    # Load MNIST data
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")

    classifier = tf.estimator.Estimator(model_fn=cnn_model,
                                        model_dir="/cnn_model")

if __name__ == '__main__':
    tf.app.run()
