# tf-layers-mnist-analysis
MNIST data analysis using TensorFlow Layers API. Attempting to recreate the same network architecture as found in my [other repo][keras] which uses Keras.

CNN model architecture:
- Convolutional layer 1
- Conv layer 2
- Max max pooling
- Conv layer 3
- Conv Layer 4
- Max pooling 2
- Fully connected nn layer with 1024 neurons
- Output layer

## Dependencies

This code only needs:
- TensorFlow
- Numpy

Tested on Python 3.6.

## Running

```
python analysis.py
```

Prints out accuracy and loss:

```
{'accuracy': 0.9755, 'loss': 0.09953127, 'global_step': 11000}
```

[keras]: https://github.com/akajuvonen/keras-mnist-analysis
