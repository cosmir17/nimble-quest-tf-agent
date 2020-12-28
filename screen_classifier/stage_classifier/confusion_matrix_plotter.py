import itertools
import matplotlib.pyplot as plt
import numpy as np
import io
import tensorflow as tf


# Confusion matrix part is copied from:
# https://colab.research.google.com/drive/1pdzZ2MB2g6CT_-bT0D0bO2IKyghOhlM_
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    figure = plt.figure(figsize=(8, 8))

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
      plt.text(j, i, format(cm[i, j], fmt),
               horizontalalignment="center",
               color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure


def plot_mismatched_images(p, x, y, class_names):
    misclassified_idices = np.where(p != y)[0]
    imgs = []
    for i in misclassified_idices:
        figure = plt.figure(figsize=(6, 3))
        img = x[i]
        plt.imshow(img, cmap='gray')
        plt.title("True label: %s __ Predicted: %s" % (class_names[y[i]], class_names[p[i]]))
        figure_image = plot_figure_to_image(figure)
        imgs.append(figure_image[0])
    return imgs


# This method is copied from https://www.tensorflow.org/tensorboard/image_summaries
def plot_figure_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image
