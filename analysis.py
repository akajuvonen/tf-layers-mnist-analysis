import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix


def cnn_model(features, labels, mode):
    """CNN model function.
    Arguments:
    features -- Batch features from input function
    labels -- Batch labels from input function
    mode -- train, eval, predict, instance of tf.estimator.Modekeys
    Returns:
    The estimator spec depending on the chosen mode
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
    p2flat = tf.reshape(pool2, [-1, pool2width * pool2height * pool2channels])
    # Dense layer
    dense = tf.layers.dense(inputs=p2flat, units=1024, activation=tf.nn.relu)

    # Output layer
    output = tf.layers.dense(inputs=dense, units=10)

    # Predictions for eval and predict
    predictions = {"class": tf.argmax(input=output, axis=1),
                   "probabilities": tf.nn.softmax(output,
                                                  name="softmax_tensor")}

    # If predicting, return early
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate loss, global step, train op, predictions, acc etc.
    # Needed for different modes
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=output)
    global_step = tf.train.get_global_step()
    train_op = tf.train.AdamOptimizer(1e-2).minimize(loss, global_step)
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=predictions["class"])
    eval_metric_ops = {"accuracy": accuracy}

    # Return the model spec
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op,
                                      eval_metric_ops=eval_metric_ops,
                                      predictions=predictions)


def main(dummy):
    # Load MNIST data
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    # Training data
    train_data = mnist.train.images
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    # Evaluation data
    eval_data = mnist.test.images
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    # Create the estimator
    estimator = tf.estimator.Estimator(model_fn=cnn_model,
                                       model_dir='cnn_model')

    # Train the model
    train_input = tf.estimator.inputs.numpy_input_fn(x={"x": train_data},
                                                     y=train_labels,
                                                     batch_size=100,
                                                     num_epochs=10,
                                                     shuffle=True)
    estimator.train(train_input)

    # Evaluate
    eval_input = tf.estimator.inputs.numpy_input_fn(x={"x": eval_data},
                                                    y=eval_labels,
                                                    num_epochs=1,
                                                    shuffle=False)
    results = estimator.evaluate(input_fn=eval_input)
    print(results)

    # Predictions
    predict_input = tf.estimator.inputs.numpy_input_fn(x={"x": eval_data},
                                                          num_epochs=1,
                                                          shuffle=False)
    predictions = list(estimator.predict(input_fn=predict_input))
    pred_classes = [x['class'] for x in predictions]
    pred_classes = np.asarray(pred_classes)

    # Confusion matrix
    cm = confusion_matrix(eval_labels, pred_classes)
    print(cm)

if __name__ == '__main__':
    tf.app.run()
