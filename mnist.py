import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from polyaxon_client.tracking import Experiment, get_outputs_path, get_data_paths

tf.logging.set_verbosity(tf.logging.DEBUG)

image_size = 28
labels_size = 10
hidden_size = 1024


def input_fn(dataset):
    features = dataset.images
    labels = dataset.labels.astype(np.int32)
    return features, labels


def main():
    experiment = Experiment()

    # Read in the MNIST dataset
    data_path = list(get_data_paths().values())[0]
    mnist = input_data.read_data_sets("{}/MNIST_data/".format(data_path), one_hot=False)

    # Define the Estimator
    feature_columns = [tf.contrib.layers.real_valued_column("", dimension=image_size*image_size)]
    classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                                hidden_units=[hidden_size],
                                                n_classes=labels_size,
                                                optimizer=tf.train.AdamOptimizer())

    # Fit the model
    features, labels = input_fn(mnist.train)
    classifier.fit(x=features, y=labels, batch_size=100, steps=1000)

    # Evaluate the model on the test data
    features, labels = input_fn(mnist.test)
    metrics = classifier.evaluate(x=features, y=labels, steps=1)
    test_accuracy = metrics['accuracy']
    experiment.log_metrics(**metrics)

    print("\nTest accuracy: %g %%"%(test_accuracy*100))

    # Predict the new examples and compare with the onderlying values
    features = mnist.validation.images[:10]
    labels = mnist.validation.labels[:10].astype(np.int32)
    predictions = list(classifier.predict(x=features))

    print("\nPredicted labels from validation set: %s"%predictions)
    print("Underlying values: %s"%list(labels))


if __name__ == "__main__":
    main()
