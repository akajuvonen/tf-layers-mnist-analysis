import numpy as np
import tensorflow as tf


def cnn_model(features, labels, mode):
    """CNN model function.
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

    # Flatten the pool2 outputs
    pool2width = 7
    pool2height = 7
    pool2channels = 32
    p2flat = tf.reshape(pool2,
                            [-1, pool2width * pool2height * pool2channels])
    # Dense layer
    dense = tf.layers.dense(inputs=p2flat, units=1024,
                            activation=tf.nn.relu)
    # Maybe add a dropout later
    # Logits Layer
    logits = tf.layers.dense(inputs=dense, units=10)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
      optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
      train_op = optimizer.minimize(
          loss=loss,
          global_step=tf.train.get_global_step())
      return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)



def main(dummy):
    # Load MNIST data
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images  # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)

    classifier = tf.estimator.Estimator(model_fn=cnn_model,
                                        model_dir="cnn_model")

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=10,
        shuffle=True)
    classifier.train(input_fn=train_input_fn)

if __name__ == '__main__':
    tf.app.run()
