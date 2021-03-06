import theano
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from neupy import algorithms, layers, environment


environment.reproducible()
theano.config.floatX = 'float32'


def reduce_dimension(network, data):
    """ Function minimize input data dimention using
    pre-trained autoencoder.
    """
    minimized_data = network.input_layer.output(data)
    return minimized_data.eval()


mnist = datasets.fetch_mldata('MNIST original')

data = mnist.data / 255.
features_mean = data.mean(axis=0)
data = (data - features_mean).astype(np.float32)

np.random.shuffle(data)
x_train, x_test = data[:60000], data[60000:]

autoencoder = algorithms.Momentum(
    [
        layers.Dropout(proba=0.5),
        layers.Sigmoid(784),
        layers.Sigmoid(100),
        layers.Output(784),
    ],
    step=0.25,
    verbose=True,
    momentum=0.99,
    nesterov=True,
)
autoencoder.train(x_train, x_train, x_test, x_test, epochs=100)

n_samples = 4
image_vectors = x_test[:n_samples, :]
images = (image_vectors + features_mean) * 255.
predicted_images = autoencoder.predict(image_vectors)
predicted_images = (predicted_images + features_mean) * 255.

# Compare real and reconstructed images
fig, axes = plt.subplots(4, 2, figsize=(12, 8))
iterator = zip(axes, images, predicted_images)

for (left_ax, right_ax), real_image, predicted_image in iterator:
    real_image = real_image.reshape((28, 28))
    predicted_image = predicted_image.reshape((28, 28))

    left_ax.imshow(real_image, cmap=plt.cm.binary)
    right_ax.imshow(predicted_image, cmap=plt.cm.binary)

plt.show()
